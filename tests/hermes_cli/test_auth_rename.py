"""Tests for `hermes auth rename` — symmetric with test_auth_commands.py."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.credential_pool import PooledCredential, load_pool, write_credential_pool
from hermes_cli import auth_commands
from hermes_cli.auth import _load_auth_store, _save_auth_store


PROVIDER = "openai-codex"


@pytest.fixture(autouse=True)
def isolated_auth_store(tmp_path, monkeypatch):
    """Every test writes into a throwaway HERMES_HOME.

    The repo-wide `_isolate_env` autouse in `tests/conftest.py` already redirects
    HERMES_HOME and HOME to per-test tempdirs; we point HERMES_HOME at a
    subdirectory that intentionally does NOT collide with the patched
    `~/.hermes/auth.json` seat-belt in `hermes_cli.auth._auth_file_path`.
    """
    hermes_home = tmp_path / "sandbox-hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))


def _entry_dict(id_: str, label: str, priority: int = 0) -> dict:
    return {
        "id": id_,
        "label": label,
        "auth_type": "oauth",
        "priority": priority,
        "source": "manual:device_code",
        "access_token": f"access-{id_}",
        "refresh_token": f"refresh-{id_}",
    }


def _seed(entries: list[dict], provider: str = PROVIDER) -> None:
    rows = [PooledCredential.from_dict(provider, e).to_dict() for e in entries]
    write_credential_pool(provider, rows)


def _labels(provider: str = PROVIDER) -> list[str]:
    return [e.label for e in load_pool(provider).entries()]


class TestRenameHappyPath:
    def test_rename_by_index(self):
        _seed([_entry_dict("aaa111", "codex-1"), _entry_dict("bbb222", "codex-2", priority=1)])
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="1", new_label="work"))
        assert _labels() == ["work", "codex-2"]

    def test_rename_by_entry_id(self):
        _seed([_entry_dict("aaa111", "codex-1"), _entry_dict("bbb222", "codex-2", priority=1)])
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="bbb222", new_label="personal"))
        assert _labels() == ["codex-1", "personal"]

    def test_rename_by_current_label(self):
        _seed([_entry_dict("aaa111", "codex-1"), _entry_dict("bbb222", "codex-2", priority=1)])
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="codex-1", new_label="renamed"))
        assert _labels() == ["renamed", "codex-2"]

    def test_rename_trims_whitespace(self):
        _seed([_entry_dict("aaa111", "codex-1")])
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="1", new_label="  padded  "))
        assert _labels() == ["padded"]

    def test_no_op_rename_to_same_label_succeeds(self):
        _seed([_entry_dict("aaa111", "codex-1")])
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="1", new_label="codex-1"))
        assert _labels() == ["codex-1"]


class TestRenameValidation:
    @pytest.mark.parametrize(
        "bad_label",
        ["", "   ", "\t", "\n", "line\nbreak", "with\x00null",
         "with\x1bescape", "with\x07bell"])
    def test_rejects_invalid_label(self, bad_label):
        _seed([_entry_dict("aaa111", "codex-1")])
        with pytest.raises(SystemExit):
            auth_commands.auth_rename_command(
                SimpleNamespace(provider=PROVIDER, target="1", new_label=bad_label))
        # Pool must be untouched after the rejection.
        assert _labels() == ["codex-1"]

    def test_rejects_none_label(self):
        _seed([_entry_dict("aaa111", "codex-1")])
        with pytest.raises(SystemExit):
            auth_commands.auth_rename_command(
                SimpleNamespace(provider=PROVIDER, target="1", new_label=None))
        assert _labels() == ["codex-1"]

    def test_rejects_duplicate_label_within_provider(self):
        _seed([_entry_dict("aaa111", "codex-1"), _entry_dict("bbb222", "codex-2", priority=1)])
        with pytest.raises(SystemExit, match="already in use"):
            auth_commands.auth_rename_command(
                SimpleNamespace(provider=PROVIDER, target="1", new_label="codex-2"))
        assert _labels() == ["codex-1", "codex-2"]

    def test_rejects_duplicate_label_case_insensitive(self):
        _seed([_entry_dict("aaa111", "codex-1"), _entry_dict("bbb222", "codex-2", priority=1)])
        with pytest.raises(SystemExit, match="already in use"):
            auth_commands.auth_rename_command(
                SimpleNamespace(provider=PROVIDER, target="1", new_label="CODEX-2"))
        assert _labels() == ["codex-1", "codex-2"]

    def test_unknown_selector_exits_cleanly(self):
        _seed([_entry_dict("aaa111", "codex-1")])
        with pytest.raises(SystemExit):
            auth_commands.auth_rename_command(
                SimpleNamespace(provider=PROVIDER, target="does-not-exist", new_label="whatever"))
        assert _labels() == ["codex-1"]

    def test_empty_provider_exits(self):
        _seed([_entry_dict("aaa111", "codex-1")])
        with pytest.raises(SystemExit):
            auth_commands.auth_rename_command(
                SimpleNamespace(provider="", target="1", new_label="whatever"))


class TestLegacyLabelMirror:
    """`nous` and `openai-codex` still write providers.<x>.label as a mirror.
    Rename must keep it in sync when the mirror matches the old label, and must
    NOT clobber a value the user has already diverged."""

    def _write_provider_label_mirror(self, provider: str, label: str) -> None:
        store = _load_auth_store()
        providers = store.setdefault("providers", {})
        providers.setdefault(provider, {})["label"] = label
        _save_auth_store(store)

    def _read_provider_label_mirror(self, provider: str):
        return (_load_auth_store().get("providers") or {}).get(provider, {}).get("label")

    def test_updates_mirror_for_openai_codex_when_it_matches_old_label(self):
        _seed([_entry_dict("aaa111", "codex-primary")])
        self._write_provider_label_mirror(PROVIDER, "codex-primary")
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="1", new_label="codex-renamed"))
        assert self._read_provider_label_mirror(PROVIDER) == "codex-renamed"

    def test_updates_mirror_for_nous(self, monkeypatch):
        """`nous` pool filters seeded entries (free-tier heuristics), so we can't
        exercise the full CLI happy path against it in a hermetic test. Instead
        assert the legacy-mirror helper the CLI calls does the right thing for
        `nous`, and separately that `nous` is in the audited allowlist."""
        from hermes_cli import auth_commands
        provider = "nous"
        assert provider in auth_commands._LEGACY_LABEL_MIRROR_PROVIDERS
        self._write_provider_label_mirror(provider, "nous-primary")
        auth_commands._mirror_legacy_label(provider, "nous-primary", "nous-renamed")
        assert self._read_provider_label_mirror(provider) == "nous-renamed"

    def test_leaves_mirror_alone_if_user_already_diverged(self):
        _seed([_entry_dict("aaa111", "codex-primary")])
        self._write_provider_label_mirror(PROVIDER, "user-hand-edited")
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="1", new_label="codex-renamed"))
        # Pool row is renamed, mirror is preserved.
        assert _labels() == ["codex-renamed"]
        assert self._read_provider_label_mirror(PROVIDER) == "user-hand-edited"

    def test_does_not_touch_mirror_for_providers_that_do_not_use_it(self, monkeypatch):
        """A provider not in ``_LEGACY_LABEL_MIRROR_PROVIDERS`` must not gain a
        ``providers.<provider>.label`` mirror after a rename. Test the helper
        directly to avoid providers that auto-detect credentials from the
        environment (copilot pulls from `gh auth`, openrouter from env, etc.)."""
        from hermes_cli import auth_commands
        provider = "anthropic"
        assert provider not in auth_commands._LEGACY_LABEL_MIRROR_PROVIDERS
        # Even if a mirror value happens to already exist, the helper is a no-op.
        self._write_provider_label_mirror(provider, "should-not-change")
        auth_commands._mirror_legacy_label(provider, "should-not-change", "would-change-if-buggy")
        assert self._read_provider_label_mirror(provider) == "should-not-change"


class TestRenameOutput:
    def test_prints_old_and_new_label(self, capsys):
        _seed([_entry_dict("aaa111", "codex-1")])
        auth_commands.auth_rename_command(
            SimpleNamespace(provider=PROVIDER, target="1", new_label="work"))
        out = capsys.readouterr().out
        assert "codex-1" in out
        assert "work" in out
        assert PROVIDER in out
