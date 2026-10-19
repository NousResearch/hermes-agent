"""Tests for ``_display_provider_name`` — a NAMED custom provider must not print as ``custom``.

The billed route stays ``custom`` (price tables, context lookup); only the /status label changes,
mirroring /model (config key) and the /new notice (``requested_provider``).
"""

from types import SimpleNamespace
from unittest.mock import patch

from gateway.slash_commands_status import _display_provider_name


def test_live_agent_reports_its_requested_provider():
    agent = SimpleNamespace(requested_provider="engy")
    assert _display_provider_name("custom", agent, {}, {}, {}) == "engy"


def test_override_wins_when_agent_is_gone():
    assert _display_provider_name("custom", None, {"provider": "engy"}, {}, {}) == "engy"


def test_config_name_used_only_on_matching_endpoint():
    cfg = {"model": {"provider": "engy", "base_url": "https://api.engy.ai/v1"}}
    with patch("gateway.run._load_gateway_config", return_value=cfg):
        assert _display_provider_name(
            "custom", None, {}, {}, {"billing_base_url": "https://api.engy.ai/v1/"}) == "engy"
        # A /model switch to another custom endpoint must keep the honest generic class.
        assert _display_provider_name(
            "custom", None, {}, {}, {"billing_base_url": "https://api.deepinfra.com/v1"}) == "custom"


def test_non_custom_provider_is_untouched():
    assert _display_provider_name("nous", None, {}, {}, {}) == "nous"
    assert _display_provider_name("custom", None, {"provider": "custom"}, {}, {}) == "custom"