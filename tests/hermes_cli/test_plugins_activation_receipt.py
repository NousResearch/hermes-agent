"""Regression for #119502 — the ``reload-plugins`` receipt may only claim callbacks are active when
the gateway confirmed the native handler re-wire.

``reload_plugins_verb`` answers ``adapters_rewired=None`` when its read of the event loop times out,
and a ``register_platform_handler`` factory may raise; neither outcome may reach the user as
``active in the running gateway now``. Both have to say the wiring is unconfirmed / failed and hand
back a retry-or-restart instruction. A confirmed answer keeps the message it has today.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.plugins_activation import activate_plugin_now, activation_hint

_DEMO = {"name": "demo", "key": "demo",
         "activated_now": {"callbacks": ["telegram"]}, "deferred": {}}


def _activate(answer: dict) -> dict:
    # `in_process=False` asks a running serve/Dashboard backend to do the in-process half — patch it
    # out so the receipt under test comes only from the control-socket answer.
    with patch("gateway.control_socket.reload_gateway_plugins", return_value=answer), \
            patch("hermes_cli.plugins_activation.notify_serve_backend", return_value=None):
        return activate_plugin_now("demo", in_process=False)


@pytest.mark.parametrize("answer", [
    pytest.param(
        {"reloaded": True, "adapters_rewired": None, "activations": [_DEMO]},
        id="rewire-timeout"),
    pytest.param(
        {"reloaded": True, "adapters_rewired": 1, "handler_wiring_failures": ["demo"],
         "activations": [_DEMO]},
        id="raising-handler-factory"),
])
def test_unconfirmed_or_failed_wiring_never_claims_callbacks_active(answer: dict):
    result = _activate(answer)
    # The rescan itself is acknowledged — only the wiring claim is withheld.
    assert result["gateway_reloaded"] is True
    assert result["restart_required"] is False
    assert result["handler_wiring"] != "confirmed"
    hint = activation_hint(result)
    assert "active in the running gateway now" not in hint
    assert "restart" in hint.lower()


def test_confirmed_wiring_keeps_the_existing_active_now_receipt():
    result = _activate({"reloaded": True, "adapters_rewired": 2, "activations": [_DEMO]})
    assert result["handler_wiring"] == "confirmed"
    assert result["handler_wiring_failures"] == []
    assert activation_hint(result) == (
        "Gateway reloaded plugins — active in the running gateway now: callbacks.")
