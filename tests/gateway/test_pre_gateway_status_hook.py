"""Tests for the ``pre_gateway_status`` plugin hook (#116455).

``gateway/run.py`` has no plugin hook site on the outbound status-callback path:
``_prepare_gateway_status_message`` re-emits the same provider-outage error text on every
retry tick with no way for a plugin to de-duplicate it. ``pre_gateway_status`` fires before
the built-in noise filtering, once per candidate status message, and lets a plugin suppress it.
"""

from unittest.mock import patch

from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from hermes_cli.plugins import VALID_HOOKS


def test_pre_gateway_status_in_valid_hooks():
    """Plugins must be able to subscribe to the new hook without warnings."""
    assert "pre_gateway_status" in VALID_HOOKS


@patch("hermes_cli.plugins.invoke_hook")
def test_suppress_action_drops_the_message(mock_invoke_hook):
    """A plugin returning ``{"action": "suppress"}`` drops the status entirely, even for a
    message that would otherwise pass the built-in filter untouched."""
    mock_invoke_hook.return_value = [{"action": "suppress"}]

    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", "still on it") is None


@patch("hermes_cli.plugins.invoke_hook")
def test_hook_receives_platform_event_type_and_message(mock_invoke_hook):
    """The hook must see the exact platform/event_type/message triple from the caller."""
    mock_invoke_hook.return_value = []

    _prepare_gateway_status_message(Platform.TELEGRAM, "warn", "  provider outage  ")

    mock_invoke_hook.assert_called_once_with(
        "pre_gateway_status", platform=Platform.TELEGRAM, event_type="warn", message="provider outage",
    )


@patch("hermes_cli.plugins.invoke_hook")
def test_non_suppress_result_falls_through_to_normal_filtering(mock_invoke_hook):
    """Any result other than an explicit suppress directive leaves the built-in
    noise filter in charge — a plugin can observe without changing behavior."""
    mock_invoke_hook.return_value = [{"action": "allow"}]

    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", "still on it") == "still on it"
    # The noise filter still swallows retry chatter when the hook doesn't suppress.
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM, "warn", "⏳ Retrying in 4.2s (attempt 1/3)...",
    ) is None


@patch("hermes_cli.plugins.invoke_hook")
def test_hook_dispatch_failure_fails_open(mock_invoke_hook):
    """A misbehaving plugin must not block status delivery."""
    mock_invoke_hook.side_effect = RuntimeError("plugin exploded")

    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", "still on it") == "still on it"
