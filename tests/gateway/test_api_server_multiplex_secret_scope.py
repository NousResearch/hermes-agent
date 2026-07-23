"""Regression for #61276: api_server agent entry under multiplex isolation.

When gateway.multiplex_profiles is on, get_secret fails closed without a
profile secret scope. Requests with a ``/p/<profile>/`` prefix are scoped by
``_profile_scope(profile)``, while plain requests must be scoped to the
process-primary profile that owns the shared listener.

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


class TestProfileScopePrimaryFallback:
    def test_noop_when_multiplex_off(self, adapter, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://from-environ.example/v1")
        with adapter._profile_scope(None):
            # Legacy single-profile path: unscoped get_secret reads os.environ.
            assert ss.get_secret("OPENROUTER_BASE_URL") == "https://from-environ.example/v1"
        assert ss.current_secret_scope() is None

    def test_primary_scope_installed_under_multiplex(self, adapter, tmp_path, monkeypatch):
        """No /p/ prefix + multiplex active → primary profile scope, not nullcontext."""
        (tmp_path / ".env").write_text(
            "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir", lambda name: tmp_path
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

    def test_unprefixed_named_primary_uses_primary_scope(self, tmp_path, monkeypatch):
        """The unprefixed shared listener must use its named primary's secrets."""
        default_home = tmp_path
        coder_home = tmp_path / "profiles" / "coder"
        for home, url in (
            (default_home, "https://default.example/v1"),
            (coder_home, "https://coder.example/v1"),
        ):
            home.mkdir(parents=True, exist_ok=True)
            (home / ".env").write_text(
                f"OPENROUTER_BASE_URL={url}\n", encoding="utf-8"
            )

        process_root = tmp_path
        with pytest.MonkeyPatch.context() as scoped:
            scoped.setattr(
                "hermes_constants.get_process_hermes_home", lambda: coder_home
            )
            scoped.setattr(
                "hermes_constants.get_default_hermes_root", lambda: process_root
            )
            adapter = APIServerAdapter(PlatformConfig(enabled=True))
        try:
            assert adapter._primary_profile == "coder"
            monkeypatch.setattr(
                "hermes_cli.profiles.get_profile_dir",
                lambda name: {"default": default_home, "coder": coder_home}[name],
            )
            ss.set_multiplex_active(True)

            with adapter._profile_scope(None):
                assert ss.get_secret("OPENROUTER_BASE_URL") == "https://coder.example/v1"
        finally:
            adapter._response_store.close()
