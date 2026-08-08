"""#70652 — default-profile adapter construction must run under a secret scope.

When ``gateway.multiplex_profiles`` is on, ``get_secret`` fails closed on an
unscoped read (``UnscopedSecretError``). Secondary profiles build their adapters
inside ``_profile_runtime_scope``, but the default/primary profile used to call
``_create_adapter`` bare — so any adapter resolving an optional secret in its
constructor (e.g. Weixin's ``WEIXIN_CDN_BASE_URL``) aborted gateway startup.

``GatewayRunner._create_default_profile_adapter`` now enters the default
profile's runtime scope when multiplexing is on, mirroring the secondary path,
and calls through directly when it is off (legacy ``os.environ`` resolution
unchanged). These tests assert the secret resolves from the default profile's
isolated ``.env`` scope — installed by the runner's own code, not by the test.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


@pytest.fixture(autouse=True)
def _reset_multiplex_flag():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _runner(multiplex: bool) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    return runner


def _point_default_home(monkeypatch, home) -> None:
    """Redirect get_hermes_home() (as seen by the runner) at the temp home."""
    import hermes_constants as hc
    from gateway import run as run_mod

    monkeypatch.setattr(hc, "get_hermes_home", lambda: home)
    monkeypatch.setattr(run_mod, "get_hermes_home", lambda: home)


class TestDefaultProfileAdapterSecretScope:
    def test_reads_secret_from_default_profile_env_under_multiplex(
        self, tmp_path, monkeypatch
    ):
        """Adapter ctor reads the default profile's .env, not process env."""
        (tmp_path / ".env").write_text(
            "WEIXIN_CDN_BASE_URL=https://cdn.from-default-profile.example/c2c\n",
            encoding="utf-8",
        )
        _point_default_home(monkeypatch, tmp_path)
        # A cross-profile value in os.environ must NOT leak through.
        monkeypatch.setenv("WEIXIN_CDN_BASE_URL", "https://leak.example/c2c")
        ss.set_multiplex_active(True)

        runner = _runner(multiplex=True)

        def _fake_create(platform, config):
            # Mirror WeixinAdapter.__init__: resolve an optional secret. This is
            # exactly the read that raised UnscopedSecretError before the fix.
            return SimpleNamespace(
                platform=platform,
                cdn_base_url=ss.get_secret("WEIXIN_CDN_BASE_URL", "default-fallback"),
                scope_seen=ss.current_secret_scope() is not None,
            )

        monkeypatch.setattr(runner, "_create_adapter", _fake_create)

        # The runner installs the scope itself — the test does NOT pre-install one.
        assert ss.current_secret_scope() is None
        adapter = runner._create_default_profile_adapter(
            Platform.WEIXIN, PlatformConfig(enabled=True, token="t")
        )

        assert adapter.scope_seen is True
        assert adapter.cdn_base_url == "https://cdn.from-default-profile.example/c2c"
        # Scope torn down after construction; fail-closed restored.
        assert ss.current_secret_scope() is None

    def test_bare_create_adapter_still_fails_closed_under_multiplex(
        self, tmp_path, monkeypatch
    ):
        """Guards the bug: the unscoped path raises, the helper is what fixes it."""
        (tmp_path / ".env").write_text(
            "WEIXIN_CDN_BASE_URL=https://cdn.from-default-profile.example/c2c\n",
            encoding="utf-8",
        )
        _point_default_home(monkeypatch, tmp_path)
        ss.set_multiplex_active(True)

        runner = _runner(multiplex=True)

        def _fake_create(platform, config):
            return SimpleNamespace(
                platform=platform,
                cdn_base_url=ss.get_secret("WEIXIN_CDN_BASE_URL", "default-fallback"),
            )

        monkeypatch.setattr(runner, "_create_adapter", _fake_create)

        # Directly (no scope) → fail closed. This is the pre-#70652 behavior.
        with pytest.raises(ss.UnscopedSecretError):
            runner._create_adapter(
                Platform.WEIXIN, PlatformConfig(enabled=True, token="t")
            )
        # Through the helper → scope installed, no raise.
        adapter = runner._create_default_profile_adapter(
            Platform.WEIXIN, PlatformConfig(enabled=True, token="t")
        )
        assert adapter.cdn_base_url == "https://cdn.from-default-profile.example/c2c"

    def test_non_multiplex_path_reads_os_environ_unchanged(
        self, tmp_path, monkeypatch
    ):
        """Multiplex off: no scope installed, get_secret reads os.environ as before."""
        (tmp_path / ".env").write_text(
            "WEIXIN_CDN_BASE_URL=https://cdn.should-not-be-used.example/c2c\n",
            encoding="utf-8",
        )
        _point_default_home(monkeypatch, tmp_path)
        monkeypatch.setenv("WEIXIN_CDN_BASE_URL", "https://cdn.from-environ.example/c2c")
        # multiplex stays inactive (autouse fixture default)

        runner = _runner(multiplex=False)

        def _fake_create(platform, config):
            return SimpleNamespace(
                platform=platform,
                cdn_base_url=ss.get_secret("WEIXIN_CDN_BASE_URL", "default-fallback"),
                scope_seen=ss.current_secret_scope() is not None,
            )

        monkeypatch.setattr(runner, "_create_adapter", _fake_create)

        adapter = runner._create_default_profile_adapter(
            Platform.WEIXIN, PlatformConfig(enabled=True, token="t")
        )
        # No profile scope was installed, and os.environ is authoritative.
        assert adapter.scope_seen is False
        assert adapter.cdn_base_url == "https://cdn.from-environ.example/c2c"


class TestRealWeixinAdapterUnderMultiplex:
    """End-to-end: the real WeixinAdapter constructor under the helper."""

    def test_real_weixin_adapter_constructs_without_unscoped_error(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("aiohttp")
        pytest.importorskip("cryptography")
        from gateway.platforms.weixin import check_weixin_requirements

        if not check_weixin_requirements():
            pytest.skip("Weixin runtime requirements not met")

        (tmp_path / ".env").write_text(
            "WEIXIN_CDN_BASE_URL=https://cdn.real-default-profile.example/c2c\n",
            encoding="utf-8",
        )
        _point_default_home(monkeypatch, tmp_path)
        # hermes_constants.get_hermes_home is also read inside WeixinAdapter
        # (already patched by _point_default_home).
        monkeypatch.setenv("WEIXIN_CDN_BASE_URL", "https://leak.example/c2c")
        ss.set_multiplex_active(True)

        runner = _runner(multiplex=True)
        adapter = runner._create_default_profile_adapter(
            Platform.WEIXIN,
            PlatformConfig(enabled=True, token="weixin-token"),
        )

        assert adapter is not None
        assert adapter._cdn_base_url == "https://cdn.real-default-profile.example/c2c"
        assert ss.current_secret_scope() is None
