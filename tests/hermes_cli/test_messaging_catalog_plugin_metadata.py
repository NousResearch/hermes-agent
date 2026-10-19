"""A platform plugin's PlatformEntry is the single source of its dashboard card and write checks."""

import pytest
from fastapi import HTTPException

from gateway.platform_registry import PlatformEntry, platform_registry


def test_plugin_entry_declares_card_metadata_and_env_value_rules():
    from hermes_cli.web_routers.messaging import _env_value_rules, _validate_messaging_env_value
    from hermes_cli.web_server_messaging import _build_catalog_entry

    entry = PlatformEntry(
        name="cardtest", label="Card Test", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        required_env=["CARDTEST_TOKEN", "CARDTEST_HOST"], install_hint="pip install cardtest",
        dashboard_label="Card Test (beta)", dashboard_description="Talk to Hermes over Card Test.",
        docs_url="https://example.invalid/cardtest", dashboard_env_vars=("CARDTEST_TOKEN",),
        dashboard_required_env=("CARDTEST_TOKEN",),
        env_value_rules={"CARDTEST_TOKEN": (lambda v: v.startswith("ct-"), "must start with ct-")},
    )
    platform_registry.register(entry, scope=platform_registry.current_scope_key())
    try:
        card = _build_catalog_entry("cardtest", entry)
        assert card["name"] == "Card Test (beta)"
        assert card["description"] == "Talk to Hermes over Card Test."
        assert card["docs_url"] == "https://example.invalid/cardtest"
        assert card["env_vars"][0] == "CARDTEST_TOKEN"
        assert card["required_env"] == ("CARDTEST_TOKEN",)

        rules = _env_value_rules("cardtest")
        _validate_messaging_env_value(rules, "CARDTEST_TOKEN", "ct-123")
        with pytest.raises(HTTPException) as exc:
            _validate_messaging_env_value(rules, "CARDTEST_TOKEN", "nope")
        assert exc.value.status_code == 400 and exc.value.detail == "must start with ct-"
    finally:
        platform_registry.unregister("cardtest", scope=platform_registry.current_scope_key())
