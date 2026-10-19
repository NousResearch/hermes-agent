"""``/v1/capabilities`` advertises ``turn_idempotency_v1``.

The mobile/desktop client flips its idempotent-turn path on only when the authenticated capability
payload declares this feature, so the flag is the app's feature gate for the ``client_turn_id`` ack
and ``turn.status``. It is unconditional: the code that implements it ships in the same build.
"""

from __future__ import annotations

import asyncio
import json
import types

from gateway.platforms.api_server import APIServerAdapter


def _adapter_stub() -> types.SimpleNamespace:
    """The slice of the adapter ``_handle_capabilities`` reads (auth is bypassed by ``__wrapped__``)."""
    return types.SimpleNamespace(
        _model_name="hermes", _cors_origins=[], _api_key="",
        _browser_control_enabled=lambda: False,
        _browser_control_developer_mode=lambda: False,
        _run_idempotency_store=types.SimpleNamespace(durable=True),
        gateway_runner=None)


def _capabilities() -> dict:
    response = asyncio.run(
        APIServerAdapter._handle_capabilities.__wrapped__(_adapter_stub(), None))
    return json.loads(response.text)


def test_capabilities_declare_turn_idempotency():
    features = _capabilities()["features"]

    assert features["turn_idempotency_v1"] is True
    # Additive: the flags the client already depends on keep their values.
    assert features["chat_completions"] is True
    assert features["chat_completions_streaming"] is True
    assert features["runs_idempotency"]["supported"] is True
