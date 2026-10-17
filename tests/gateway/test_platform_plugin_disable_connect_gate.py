"""``plugins.disabled`` is authoritative for platform adapters (#68367).

The deny-list is enforced when plugins are *discovered* (``gate_manifest``), but
nothing consulted it once a platform entry was registered: the env-enable pass
(``_enable_plugin_platforms_from_env``) re-enabled an already-registered plugin
platform from inherited credentials on every config reload, and
``platform_registry.create_adapter()`` happily built the adapter — so a platform
the user had just disabled in ``plugins.disabled`` kept connecting (the duplicate
responder on the same Urbit moon in #68367).

Every other surface enforces the deny-list at request time (the dashboard's
``_plugin_api_runtime_gate``, ``providers/__init__``, ``tools/skills_tool_plugin``);
platform connections now do too.

These tests drive the real code paths: a real ``plugins.disabled`` in a temp
HERMES_HOME, a real registry entry, real ``load_gateway_config()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gateway import config_env
from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platform_registry import PlatformEntry, platform_registry

PLATFORM = "tlon"
MANIFEST_NAME = "tlon-platform"
MANIFEST_KEY = "platforms/tlon"
CREDENTIAL_ENV = "TLON_SHIP_CODE"
ALLOWED_CONFIG = "plugins: {}\n"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME whose config.yaml the test writes itself."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def fake_platform():
    """A registered plugin platform, exactly as ``PluginContext.register_platform`` leaves it."""
    entry_kwargs = dict(
        name=PLATFORM,
        label="Tlon",
        adapter_factory=lambda cfg: MagicMock(name=f"{PLATFORM}-adapter"),
        check_fn=lambda: True,
        is_connected=lambda cfg: True,
        env_enablement_fn=lambda: {"ship_code": "inherited-from-parent-env"},
        source="plugin",
        plugin_name=MANIFEST_NAME,
    )
    # ``plugin_key`` is the path-derived spelling ``plugins.disabled`` is written with
    # (``platforms/tlon``); build the entry without it too, so the contract is proven
    # behaviourally on base rather than dying at fixture setup.
    if "plugin_key" in PlatformEntry.__dataclass_fields__:
        entry_kwargs["plugin_key"] = MANIFEST_KEY
    entry = PlatformEntry(**entry_kwargs)
    platform_registry.register(entry)
    yield entry
    platform_registry.unregister(PLATFORM)


def _write_disabled_config(home, disabled=(MANIFEST_KEY, MANIFEST_NAME)):
    lines = ["plugins:", "  disabled:"]
    lines.extend(f"    - {key}" for key in disabled)
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_plugins_disabled_beats_env_enable_and_reverts_without_restart(
    hermes_home, fake_platform, monkeypatch, caplog
):
    """Contract: the deny-list wins over the env-enable pass, announces itself, and is re-read
    on the next config reload — so a disabled plugin platform goes dark without a restart."""
    monkeypatch.setenv(CREDENTIAL_ENV, "moon-code")

    # Precondition: with the plugin allowed the same entry enables from the same env credential,
    # so the refusal below is the deny-list's doing and not a dead registration.
    (hermes_home / "config.yaml").write_text(ALLOWED_CONFIG, encoding="utf-8")
    precondition = load_gateway_config().platforms.get(Platform(PLATFORM))
    assert precondition is not None and precondition.enabled is True

    # Add the plugin to plugins.disabled at runtime — no restart, no registry change. Key-only
    # spelling (``platforms/tlon``), the one ``hermes plugins disable`` writes for a platform plugin.
    _write_disabled_config(hermes_home, disabled=(MANIFEST_KEY,))
    warned = getattr(config_env, "_PLUGIN_DISABLE_WARNED", None)
    if warned is not None:  # one-time warning set; cleared so the assertion can't be vacuous
        warned.clear()
    with caplog.at_level("WARNING", logger="gateway.config"):
        config = load_gateway_config()
    cfg = config.platforms.get(Platform(PLATFORM))
    assert cfg is None or cfg.enabled is False, (
        "plugins.disabled must beat env-presence auto-enable: the gateway would "
        "otherwise connect a platform the profile explicitly disabled (#68367)"
    )
    assert any(
        MANIFEST_KEY in rec.getMessage() or MANIFEST_NAME in rec.getMessage()
        for rec in caplog.records
    ), "the operator must learn why the platform went dark"

    # Control: drop the deny-list and the very next reload enables it again.
    (hermes_home / "config.yaml").write_text(ALLOWED_CONFIG, encoding="utf-8")
    allowed = load_gateway_config().platforms.get(Platform(PLATFORM))
    assert allowed is not None and allowed.enabled is True


def test_create_adapter_refuses_a_platform_whose_plugin_is_disabled(
    hermes_home, fake_platform
):
    """Contract: ``create_adapter`` is the single funnel every connect path uses, so it refuses a
    platform whose plugin is in ``plugins.disabled`` — even when config.yaml enables the platform."""
    platform_cfg = GatewayConfig().platforms.get(Platform(PLATFORM))

    # Control first: without the deny-list the registered entry builds an adapter.
    (hermes_home / "config.yaml").write_text(ALLOWED_CONFIG, encoding="utf-8")
    assert platform_registry.create_adapter(PLATFORM, platform_cfg) is not None

    _write_disabled_config(hermes_home)
    assert platform_registry.create_adapter(PLATFORM, platform_cfg) is None, (
        "create_adapter must refuse a platform whose plugin is in plugins.disabled (#68367)"
    )

    # config.yaml enabling the platform cannot resurrect it either.
    with (hermes_home / "config.yaml").open("a", encoding="utf-8") as fh:
        fh.write(f"platforms:\n  {PLATFORM}:\n    enabled: true\n")
    yaml_enabled = load_gateway_config().platforms.get(Platform(PLATFORM))
    assert yaml_enabled is not None and yaml_enabled.enabled is True  # YAML intent still loads...
    assert platform_registry.create_adapter(PLATFORM, yaml_enabled) is None, (
        "a YAML enabled: true must not resurrect a plugin the deny-list refuses (#68367)"
    )
