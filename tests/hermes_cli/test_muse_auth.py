"""Tests for `muse login` session reuse (hermes_cli.muse_auth + resolver hook).

Hermes' meta-ai provider family previously resolved credentials from explicit
API-key env vars and the credential pool only, i.e. pay-as-you-go billing
even for users with a Muse Code Power Usage subscription (`muse login`).
The last-resort fallback in `_resolve_api_key_provider_secret` now reuses the
local `muse login` provisioned key (macOS keychain) when nothing explicit is
configured.

Rules under test (see hermes_cli/muse_auth.py):
- explicit env beats pool beats muse-login; muse-login never fires for
  non-Meta providers and never fires when explicit config exists (no stray
  `security` subprocess, cf. #60800);
- the login `access_token` is ignored (401s on /v1/models); only the stable
  provisioned `api_key` is used;
- secret material is never logged;
- stdlib + pytest + unittest.mock only; no live network/keychain access.

The `LLM|...` fixtures below are synthetic values in the documented shape,
never real credentials.
"""

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

LOGIN_KEY = "LLM|1403246500870000|synthetic-test-key-material"
PAYG_KEY = "LLM_2272589396850547 synthetic-test-payg-material"


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return home


@pytest.fixture(autouse=True)
def _fresh_login_cache():
    from hermes_cli import muse_auth

    muse_auth._reset_login_cache()
    yield
    muse_auth._reset_login_cache()


def _write_env_file(home: Path, **kwargs) -> None:
    lines = [f"{k}={v}" for k, v in kwargs.items()]
    (home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mock_pool(*entries):
    pool = MagicMock()
    pool.has_credentials.return_value = bool(entries)
    pool.peek.return_value = entries[0] if entries else None
    pool.entries.return_value = list(entries)
    return pool


def _pool_entry(token):
    e = MagicMock()
    e.access_token = token
    e.runtime_api_key = ""
    return e


def _meta_pconfig():
    from hermes_cli.auth import ProviderConfig

    return ProviderConfig(
        id="meta-ai",
        name="Meta Model API",
        auth_type="api_key",
        api_key_env_vars=("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"),
    )


def _keychain_payload(**overrides):
    payload = {
        "secret_schema_version": 1,
        "api_key": LOGIN_KEY,
        "access_token": "dca-synthetic",
    }
    payload.update(overrides)
    return payload


class TestReadMuseLoginKey:
    def test_valid_payload_returns_key_and_source(self, monkeypatch):
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/security")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(_keychain_payload()),
            stderr="",
        )
        with patch.object(muse_auth.subprocess, "run", return_value=completed) as run:
            key, source = muse_auth.read_muse_login_key()
        assert key == LOGIN_KEY
        assert source == muse_auth.MUSE_LOGIN_SOURCE
        run.assert_called_once()

    def test_missing_api_key_returns_empty(self, monkeypatch):
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/security")
        payload = _keychain_payload()
        del payload["api_key"]
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        with patch.object(muse_auth.subprocess, "run", return_value=completed):
            assert muse_auth.read_muse_login_key() == ("", "")

    def test_non_json_payload_returns_empty(self, monkeypatch):
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/security")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        with patch.object(muse_auth.subprocess, "run", return_value=completed):
            assert muse_auth.read_muse_login_key() == ("", "")

    def test_keychain_miss_returns_empty(self, monkeypatch):
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/security")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=44,
            stdout="",
            stderr="not found",
        )
        with patch.object(muse_auth.subprocess, "run", return_value=completed):
            assert muse_auth.read_muse_login_key() == ("", "")

    def test_non_darwin_never_spawns_subprocess(self, monkeypatch):
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "linux")
        with patch.object(muse_auth.subprocess, "run") as run:
            assert muse_auth.read_muse_login_key() == ("", "")
        run.assert_not_called()

    def test_missing_security_helper_returns_empty(self, monkeypatch):
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda _name: None)
        with patch.object(muse_auth.subprocess, "run") as run:
            assert muse_auth.read_muse_login_key() == ("", "")
        run.assert_not_called()

    def test_miss_is_cached(self, monkeypatch):
        """A keychain miss must not re-spawn `security` on every resolve."""
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/security")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=44,
            stdout="",
            stderr="not found",
        )
        with patch.object(muse_auth.subprocess, "run", return_value=completed) as run:
            assert muse_auth.read_muse_login_key() == ("", "")
            assert muse_auth.read_muse_login_key() == ("", "")
        run.assert_called_once()

    def test_secret_never_logged(self, monkeypatch, caplog):
        from hermes_cli import muse_auth

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/security")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(_keychain_payload()),
            stderr="",
        )
        with patch.object(muse_auth.subprocess, "run", return_value=completed):
            with caplog.at_level(logging.DEBUG):
                muse_auth.read_muse_login_key()
        for record in caplog.records:
            assert LOGIN_KEY not in record.getMessage()


class TestResolverFallbackOrder:
    def _resolve(self, provider_id="meta-ai", pconfig=None, pool=None):
        from hermes_cli.auth import _resolve_api_key_provider_secret

        with patch("agent.credential_pool.load_pool", return_value=pool):
            return _resolve_api_key_provider_secret(
                provider_id=provider_id,
                pconfig=pconfig or _meta_pconfig(),
            )

    def test_env_beats_muse_login(self, isolated_hermes_home):
        _write_env_file(isolated_hermes_home, MODEL_API_KEY=PAYG_KEY)
        with patch(
            "hermes_cli.muse_auth.read_muse_login_key",
            return_value=(LOGIN_KEY, "muse-login:keychain"),
        ) as login:
            key, source = self._resolve(pool=_mock_pool())
        assert key == PAYG_KEY
        assert source == "MODEL_API_KEY"
        login.assert_not_called()

    def test_pool_beats_muse_login(self, isolated_hermes_home):
        (isolated_hermes_home / ".env").write_text("", encoding="utf-8")
        with patch(
            "hermes_cli.muse_auth.read_muse_login_key",
            return_value=(LOGIN_KEY, "muse-login:keychain"),
        ) as login:
            key, source = self._resolve(pool=_mock_pool(_pool_entry("pool-key")))
        assert key == "pool-key"
        assert source == "credential_pool:meta-ai"
        login.assert_not_called()

    def test_muse_login_is_last_resort(self, isolated_hermes_home):
        (isolated_hermes_home / ".env").write_text("", encoding="utf-8")
        with patch(
            "hermes_cli.muse_auth.read_muse_login_key",
            return_value=(LOGIN_KEY, "muse-login:keychain"),
        ):
            key, source = self._resolve(pool=_mock_pool())
        assert key == LOGIN_KEY
        assert source == "muse-login:keychain"

    def test_muse_login_failure_returns_empty(self, isolated_hermes_home):
        (isolated_hermes_home / ".env").write_text("", encoding="utf-8")
        with patch("hermes_cli.muse_auth.read_muse_login_key", return_value=("", "")):
            assert self._resolve(pool=_mock_pool()) == ("", "")

    def test_non_meta_provider_never_consults_login(self, isolated_hermes_home):
        from hermes_cli.auth import ProviderConfig

        (isolated_hermes_home / ".env").write_text("", encoding="utf-8")
        with patch(
            "hermes_cli.muse_auth.read_muse_login_key",
            return_value=(LOGIN_KEY, "muse-login:keychain"),
        ) as login:
            key, source = self._resolve(
                provider_id="openrouter",
                pconfig=ProviderConfig(
                    id="openrouter",
                    name="OpenRouter",
                    auth_type="api_key",
                    api_key_env_vars=("OPENROUTER_API_KEY",),
                ),
                pool=_mock_pool(),
            )
        assert (key, source) == ("", "")
        login.assert_not_called()

    def test_login_key_passes_prefix_check(self):
        """The meta family declares no key prefix (fail-open): the `LLM|`-shaped
        login key must not be rejected as malformed."""
        from hermes_cli.auth import _secret_matches_declared_prefix

        for provider_id in (
            "meta-ai",
            "meta",
            "muse",
            "muse-spark",
            "model-api",
            "msl",
        ):
            assert _secret_matches_declared_prefix(provider_id, LOGIN_KEY) is True
