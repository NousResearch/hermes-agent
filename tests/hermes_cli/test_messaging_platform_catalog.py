"""Messaging-catalog metadata for plugin platforms.

Regression coverage for the enum-shadowing bug: a configured plugin platform
gets dynamically added to the ``Platform`` enum (``Platform._missing_`` via
``load_gateway_config``), so ``_messaging_platform_catalog()`` sees it in BOTH
the enum and the plugin registry.  Building its entry from the bare enum
member dropped ``required_env`` / ``description`` / ``docs_url`` — the
messaging settings card then rendered "no token needed" with every field
optional and reported the platform as configured (``all(())`` is True).
"""

import pytest

from gateway.platform_registry import PlatformEntry, platform_registry


PLATFORM_ID = "catalogtestplat"


@pytest.fixture
def registered_plugin_platform():
    entry = PlatformEntry(
        name=PLATFORM_ID,
        label="Catalog Test Platform",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        required_env=["CATALOGTESTPLAT_URL", "CATALOGTESTPLAT_TOKEN"],
        install_hint="pip install catalogtestplat",
        description="A test platform for catalog coverage.",
        docs_url="https://example.com/setup",
        source="plugin",
    )
    platform_registry.register(entry)
    try:
        yield entry
    finally:
        platform_registry.unregister(PLATFORM_ID)
        # Drop the pseudo enum member a test may have created so state
        # doesn't leak across tests.
        from gateway.config import Platform

        member = Platform._value2member_map_.pop(PLATFORM_ID, None)
        if member is not None:
            Platform._member_map_.pop(member._name_, None)


def _catalog_entry():
    from hermes_cli.web_server import _catalog_lookup

    entry = _catalog_lookup(PLATFORM_ID)
    assert entry is not None, "plugin platform missing from messaging catalog"
    return entry


def test_plugin_entry_metadata_reaches_catalog(registered_plugin_platform):
    entry = _catalog_entry()
    assert entry["required_env"] == ("CATALOGTESTPLAT_URL", "CATALOGTESTPLAT_TOKEN")
    assert entry["name"] == "Catalog Test Platform"
    assert entry["description"] == "A test platform for catalog coverage."
    assert entry["docs_url"] == "https://example.com/setup"
    # required env vars must lead the card's field list
    assert entry["env_vars"][:2] == (
        "CATALOGTESTPLAT_URL",
        "CATALOGTESTPLAT_TOKEN",
    )


def test_enum_extended_plugin_platform_keeps_metadata(registered_plugin_platform):
    """The actual regression: platform present in the enum AND the registry."""
    from gateway.config import Platform

    # Simulate load_gateway_config() materializing the pseudo enum member for
    # a configured plugin platform.
    member = Platform(PLATFORM_ID)
    assert member.value == PLATFORM_ID
    assert PLATFORM_ID in [m.value for m in Platform.__members__.values()]

    entry = _catalog_entry()
    assert entry["required_env"] == ("CATALOGTESTPLAT_URL", "CATALOGTESTPLAT_TOKEN")
    assert entry["description"] == "A test platform for catalog coverage."
    assert entry["docs_url"] == "https://example.com/setup"


def test_description_falls_back_to_install_hint(registered_plugin_platform):
    registered_plugin_platform.description = ""
    entry = _catalog_entry()
    assert entry["description"] == "pip install catalogtestplat"
