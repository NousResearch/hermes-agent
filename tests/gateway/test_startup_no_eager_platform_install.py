"""Regression tests: ``_apply_env_overrides`` must not lazy-install platform
SDKs for platforms the user has not configured.

Historically ``PlatformEntry.check_fn`` doubled as the lazy-installer
(it pip-installed the platform SDK as a side effect).  The enablement sweep
in ``_apply_env_overrides`` used to call ``check_fn`` for *every* registered
plugin platform unconditionally, so a single ``load_gateway_config()`` —
which the desktop/dashboard readiness probe (``GET /api/status``) awaits
synchronously — pip-installed Discord, Telegram, Slack, Feishu and Dingtalk
even with ``platforms: none``.  That blocked startup until every install
finished and made the desktop app time out and boot-loop (stuck at 94%).

Two layers of protection now exist:
1. The sweep consults the cheap ``is_connected`` credential check FIRST and
   only reaches the dependency check for platforms that are already enabled
   or actually configured (this file pins that contract).
2. ``check_fn`` is now defined as a PASSIVE probe; the ACTIVE installer
   lives on ``ensure_deps_fn`` and only runs from
   ``platform_registry.create_adapter()`` (#79812).
"""

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platform_registry import PlatformEntry, platform_registry


@pytest.fixture
def isolated_registry():
    """Run with a registry containing only the entries the test registers."""
    original = dict(platform_registry._entries)
    platform_registry._entries.clear()
    try:
        # ``_apply_env_overrides`` calls ``discover_plugins()`` (idempotent),
        # which would re-register the real bundled platforms and clobber the
        # fakes below.  Neutralize it so the test controls the registry.
        with patch("hermes_cli.plugins.discover_plugins", lambda *a, **k: None):
            yield platform_registry
    finally:
        platform_registry._entries.clear()
        platform_registry._entries.update(original)


def _register_fake_platform(name, *, check_fn, is_connected):
    platform_registry.register(
        PlatformEntry(
            name=name,
            label=name.title(),
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=check_fn,
            is_connected=is_connected,
            source="plugin",
        )
    )


def test_unconfigured_platform_is_not_probed_for_install(isolated_registry):
    # is_connected reports "no credentials" → the platform must be skipped
    # without ever calling check_fn (which would lazy-install the SDK).
    check_fn = MagicMock(return_value=True)
    _register_fake_platform(
        "discord", check_fn=check_fn, is_connected=lambda cfg: False
    )

    config = GatewayConfig()
    _apply_env_overrides(config)

    check_fn.assert_not_called()
    assert not config.platforms.get(Platform.DISCORD, PlatformConfig()).enabled


