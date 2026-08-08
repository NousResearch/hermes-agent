"""Tests for hermes_cli.copilot_auth — Copilot token validation and resolution."""

import json
import os
import stat
import sys
import time

import pytest
from unittest.mock import patch


class TestTokenValidation:
    """Token type validation."""

    def test_classic_pat_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghp_abcdefghijklmnop1234")
        assert valid is False
        assert "Classic Personal Access Tokens" in msg
        assert "ghp_" in msg


class TestResolveToken:
    """Token resolution with env var priority."""


    def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_gh_second"
        assert source == "GH_TOKEN"




    def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_classic"):
            with pytest.raises(ValueError, match="classic PAT"):
                resolve_copilot_token()

    def test_invalid_env_var_skips_gh_cli_fallback(self, monkeypatch):
        """When an env var is set but holds an unsupported classic PAT,
        resolve_copilot_token must NOT fall back to ``gh auth token``.

        The user explicitly exported a token; silently substituting one
        from the gh CLI credential store is surprising and the subprocess
        call adds up to 5s of latency on Windows cold starts (#60800).
        Only fall back to the CLI when NO Copilot env var is set at all.
        """
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_classic_pat_nope")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()

    def test_all_env_vars_invalid_skips_gh_cli_fallback(self, monkeypatch):
        """All three env vars set to classic PATs → no gh CLI call."""
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_one")
        monkeypatch.setenv("GH_TOKEN", "ghp_two")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_three")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()


class TestRequestHeaders:
    """Copilot API header generation."""

    def test_default_headers_include_openai_intent(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert headers["Openai-Intent"] == "conversation-edits"
        assert headers["User-Agent"] == "HermesAgent/1.0"
        assert "Editor-Version" in headers


    def test_no_vision_header_by_default(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert "Copilot-Vision-Request" not in headers


class TestCopilotDefaultHeaders:
    """The models.py copilot_default_headers uses copilot_auth."""


    def test_agent_turn_explicit(self):
        """Explicitly passing is_agent_turn=True sets x-initiator to 'agent'."""
        from hermes_cli.models import copilot_default_headers
        headers = copilot_default_headers(is_agent_turn=True)
        assert headers["x-initiator"] == "agent"

    def test_param_passthrough_both_values(self):
        """is_agent_turn param correctly maps to x-initiator for both True and False."""
        from hermes_cli.models import copilot_default_headers
        for is_agent, expected in [(True, "agent"), (False, "user")]:
            headers = copilot_default_headers(is_agent_turn=is_agent)
            assert headers["x-initiator"] == expected, (
                f"is_agent_turn={is_agent} should produce x-initiator={expected!r}, "
                f"got {headers['x-initiator']!r}"
            )


class TestApiModeSelection:
    """API mode selection matching opencode's shouldUseCopilotResponsesApi."""

    def test_gpt5_uses_responses(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5.4") is True
        assert _should_use_copilot_responses_api("gpt-5.4-mini") is True
        assert _should_use_copilot_responses_api("gpt-5.3-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2") is True
        assert _should_use_copilot_responses_api("gpt-5.1-codex-max") is True

    def test_gpt5_mini_excluded(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5-mini") is False


class TestEnvVarOrder:
    """PROVIDER_REGISTRY has correct env var order."""

    def test_copilot_env_vars_include_copilot_github_token(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert "COPILOT_GITHUB_TOKEN" in copilot.api_key_env_vars
        # COPILOT_GITHUB_TOKEN should be first
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"


# ---------------------------------------------------------------------------
# On-disk exchanged-JWT store writers (<HERMES_HOME>/.copilot_jwt.json)
# ---------------------------------------------------------------------------

_OWNER_ONLY = stat.S_IRUSR | stat.S_IWUSR


def _seed_store(path, entries):
    """Write an existing JWT store at 0o600 so the writer takes its merge path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")
    os.chmod(path, _OWNER_ONLY)


def _spying_os_open(observed):
    """Record every ``os.open`` call so the creation mode can be asserted."""
    real_os_open = os.open

    def spy(path, flags, mode=0o777, *args, **kwargs):
        observed.append((str(path), flags, mode))
        return real_os_open(path, flags, mode, *args, **kwargs)

    return spy


def _assert_temp_created_owner_only(observed):
    tmp_opens = [
        entry for entry in observed if ".copilot_jwt.json.tmp" in entry[0]
    ]
    assert tmp_opens, (
        "os.open was never called for the JWT store temp file — the writer still "
        f"creates it at the process umask; observed={observed!r}"
    )
    for path, flags, mode in tmp_opens:
        assert flags & os.O_CREAT, f"temp open missing O_CREAT: {path}"
        assert flags & os.O_EXCL, (
            f"temp open missing O_EXCL — a concurrent writer could be clobbered: {path}"
        )
        assert mode == _OWNER_ONLY, (
            f"temp open mode 0o{mode:o} != 0o600 — umask applies and the live "
            f"Copilot JWT is briefly world-readable: {path}"
        )


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits and symlinks are not enforced on Windows",
)
class TestJwtStoreDiskWrites:
    """``.copilot_jwt.json`` holds a live Copilot API token.

    Both writers — ``_save_jwt_to_disk`` and ``evict_cached_exchanged_token`` —
    used to create the temp file with ``Path.write_text`` and only ``chmod`` it
    to 0o600 afterward (world-readable at the process umask in between), then
    finish with a bare ``os.replace`` that detaches a symlinked store. They now
    share ``_write_jwt_store_atomically``: ``os.open(O_EXCL, 0o600)`` + ``fsync``
    + ``utils.atomic_replace``. Mirrors #19673 / #21148 and
    ``tests/hermes_cli/test_auth_toctou_file_modes.py``.
    """

    def test_save_creates_store_owner_only(self, tmp_path, monkeypatch):
        """``_save_jwt_to_disk`` must create the store at 0o600, never 0o644."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli import copilot_auth

        observed: list = []
        old_umask = os.umask(0o022)  # make the race observable if it regresses
        try:
            with patch.object(os, "open", _spying_os_open(observed)):
                copilot_auth._save_jwt_to_disk(
                    "fp-save", "tid=abc;exp=1", time.time() + 600, None
                )
        finally:
            os.umask(old_umask)

        store_path = tmp_path / ".copilot_jwt.json"
        assert store_path.exists(), "JWT store was not written"
        mode = stat.S_IMODE(store_path.stat().st_mode)
        assert mode == 0o600, f"JWT store mode 0o{mode:o} != 0o600"
        assert json.loads(store_path.read_text())["fp-save"]["api_token"] == "tid=abc;exp=1"
        _assert_temp_created_owner_only(observed)

    def test_save_preserves_symlinked_store(self, tmp_path, monkeypatch):
        """A ``.copilot_jwt.json`` symlinked into a managed profile must survive."""
        home = tmp_path / "home"
        home.mkdir()
        real = tmp_path / "tracked" / "copilot_jwt.json"
        _seed_store(real, {})
        link = home / ".copilot_jwt.json"
        link.symlink_to(real)
        monkeypatch.setenv("HERMES_HOME", str(home))
        from hermes_cli import copilot_auth

        copilot_auth._save_jwt_to_disk(
            "fp-save", "tid=abc;exp=1", time.time() + 600, None
        )

        assert link.is_symlink(), (
            "the symlinked JWT store was replaced by a regular file — managed "
            "profile packages silently detach (#16743)"
        )
        assert os.path.realpath(link) == os.path.realpath(real)
        assert json.loads(real.read_text())["fp-save"]["api_token"] == "tid=abc;exp=1"
        real_mode = stat.S_IMODE(real.stat().st_mode)
        assert real_mode == 0o600, f"symlink target mode 0o{real_mode:o} != 0o600"

    def test_evict_creates_store_owner_only(self, tmp_path, monkeypatch):
        """``evict_cached_exchanged_token`` must rewrite the store at 0o600."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli import copilot_auth

        drop_raw, keep_raw = "gho_drop_me", "gho_keep_me"
        drop_fp = copilot_auth._token_fingerprint(drop_raw)
        keep_fp = copilot_auth._token_fingerprint(keep_raw)
        expires_at = time.time() + 600
        store_path = tmp_path / ".copilot_jwt.json"
        _seed_store(
            store_path,
            {
                drop_fp: {"api_token": "stale", "expires_at": expires_at, "base_url": None},
                keep_fp: {"api_token": "fresh", "expires_at": expires_at, "base_url": None},
            },
        )

        observed: list = []
        old_umask = os.umask(0o022)
        try:
            with patch.object(os, "open", _spying_os_open(observed)):
                copilot_auth.evict_cached_exchanged_token(drop_raw)
        finally:
            os.umask(old_umask)

        data = json.loads(store_path.read_text())
        assert drop_fp not in data, "evicted fingerprint survived the rewrite"
        assert data[keep_fp]["api_token"] == "fresh", "eviction dropped an unrelated entry"
        mode = stat.S_IMODE(store_path.stat().st_mode)
        assert mode == 0o600, f"JWT store mode 0o{mode:o} != 0o600"
        _assert_temp_created_owner_only(observed)

    def test_evict_preserves_symlinked_store(self, tmp_path, monkeypatch):
        """Eviction must not detach a symlinked store either."""
        home = tmp_path / "home"
        home.mkdir()
        real = tmp_path / "tracked" / "copilot_jwt.json"
        monkeypatch.setenv("HERMES_HOME", str(home))
        from hermes_cli import copilot_auth

        drop_raw, keep_raw = "gho_drop_me", "gho_keep_me"
        drop_fp = copilot_auth._token_fingerprint(drop_raw)
        keep_fp = copilot_auth._token_fingerprint(keep_raw)
        expires_at = time.time() + 600
        _seed_store(
            real,
            {
                drop_fp: {"api_token": "stale", "expires_at": expires_at, "base_url": None},
                keep_fp: {"api_token": "fresh", "expires_at": expires_at, "base_url": None},
            },
        )
        link = home / ".copilot_jwt.json"
        link.symlink_to(real)

        copilot_auth.evict_cached_exchanged_token(drop_raw)

        assert link.is_symlink(), (
            "eviction replaced the symlinked JWT store with a regular file (#16743)"
        )
        assert os.path.realpath(link) == os.path.realpath(real)
        data = json.loads(real.read_text())
        assert drop_fp not in data
        assert data[keep_fp]["api_token"] == "fresh"
        real_mode = stat.S_IMODE(real.stat().st_mode)
        assert real_mode == 0o600, f"symlink target mode 0o{real_mode:o} != 0o600"

