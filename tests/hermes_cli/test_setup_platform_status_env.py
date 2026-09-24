"""The gateway setup picker must render env-configured plugin platforms as configured (#120870).

``_platform_status`` is what fills the "(not configured)" suffix in the Messaging Platforms menu.
For plugin platforms it calls the registered ``is_connected`` with a synthetic
``PlatformConfig(enabled=True)``, so a platform whose credentials live in ``.env`` only depends on
that checker resolving the env rung.
"""

from __future__ import annotations

import pytest

from gateway.platform_registry import platform_registry
from hermes_cli.plugins import discover_plugins

_CREDENTIAL_KEYS = (
    "FEISHU_APP_ID", "FEISHU_APP_SECRET",
    "WECOM_BOT_ID", "WECOM_SECRET",
    "WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET",
)


@pytest.fixture(autouse=True)
def _clean_credentials(monkeypatch):
    for key in _CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)


def _picker_status(name: str) -> str:
    """The status string the setup menu renders for plugin platform ``name``."""
    from hermes_cli.gateway_setup_wizard import _platform_status

    # The registry is profile-scoped, and the test home is not the one an earlier
    # discovery ran under — force re-registration so the lookup sees this scope.
    discover_plugins(force=True)
    entry = next((e for e in platform_registry.all_entries() if e.name == name), None)
    assert entry is not None, f"platform {name!r} not registered"
    return _platform_status({
        "key": name, "label": entry.label, "emoji": entry.emoji, "_registry_entry": entry,
    })


@pytest.mark.parametrize(
    ("platform", "credentials"),
    (
        ("feishu", {"FEISHU_APP_ID": "cli_env", "FEISHU_APP_SECRET": "secret_env"}),
        ("wecom", {"WECOM_BOT_ID": "bot_env", "WECOM_SECRET": "secret_env"}),
        ("wecom_callback", {"WECOM_CALLBACK_CORP_ID": "corp_env", "WECOM_CALLBACK_CORP_SECRET": "secret_env"}),
    ),
)
def test_env_configured_platform_reads_as_configured(platform, credentials, monkeypatch):
    for key, value in credentials.items():
        monkeypatch.setenv(key, value)
    assert _picker_status(platform) == "configured"


def test_platform_without_credentials_reads_as_not_configured():
    assert _picker_status("feishu") == "not configured"


def test_half_configured_platform_reads_as_not_configured(monkeypatch):
    """A missing secret cannot connect, so the picker must not advertise it as ready."""
    monkeypatch.setenv("FEISHU_APP_ID", "cli_env")
    assert _picker_status("feishu") == "not configured"
