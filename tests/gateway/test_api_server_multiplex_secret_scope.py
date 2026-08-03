"""Regression for #61276: api_server agent entry under multiplex isolation.

When gateway.multiplex_profiles is on, get_secret fails closed without a
profile secret scope. Requests with a ``/p/<profile>/`` prefix are scoped by
``_profile_scope(profile)``, but plain requests on the default listener used
to get ``nullcontext()`` — so agent runs crashed with UnscopedSecretError on
their first credential read (e.g. OPENROUTER_BASE_URL). ``_profile_scope``
now enters the DEFAULT profile's runtime scope when multiplex is active and
no profile was requested.

Adapted from PR #61283 by @giggling-ginger (originally targeting a
pre-``_profile_scope`` helper); no live gateway or network.
"""

from __future__ import annotations

import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


class TestProfileScopeDefaultFallback:
    def test_noop_when_multiplex_off(self, adapter, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://from-environ.example/v1")
        with adapter._profile_scope(None):
            # Legacy single-profile path: unscoped get_secret reads os.environ.
            assert ss.get_secret("OPENROUTER_BASE_URL") == "https://from-environ.example/v1"
        assert ss.current_secret_scope() is None

    def test_default_scope_installed_under_multiplex(self, adapter, tmp_path, monkeypatch):
        """No /p/ prefix + multiplex active → default profile scope, not nullcontext."""
        (tmp_path / ".env").write_text(
            "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://leak.example/v1")
        ss.set_multiplex_active(True)

        with adapter._profile_scope(None):
            assert ss.current_secret_scope() is not None
            # Profile .env wins; process env must not leak through.
            assert ss.get_secret("OPENROUTER_BASE_URL") == "https://openrouter.ai/api/v1"

        # Scope torn down; fail-closed behavior restored outside.
        assert ss.current_secret_scope() is None
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("OPENROUTER_BASE_URL")

    def test_named_profile_scope_still_wins(self, adapter, tmp_path, monkeypatch):
        """A /p/<profile>/ request keeps resolving that profile's scope."""
        profile_home = tmp_path / "profiles" / "worker"
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text(
            "OPENROUTER_BASE_URL=https://worker.example/v1\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir", lambda name: profile_home
        )
        ss.set_multiplex_active(True)

        with adapter._profile_scope("worker"):
            assert ss.get_secret("OPENROUTER_BASE_URL") == "https://worker.example/v1"
        assert ss.current_secret_scope() is None
