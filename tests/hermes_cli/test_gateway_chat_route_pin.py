"""The CLI clients send ``pin_route`` (no profile fallback chain for the session) only for a typed ``-m``,
and only to a gateway whose ``session.create`` contract lists the field."""
import argparse
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("args, env, parameters, expected", [
    ({"model": "m"}, {}, ["cwd", "model", "request_id", "source", "pin_route"], True),
    ({"model": "m"}, {}, ["cwd", "model", "request_id", "source"], None),  # an older gateway: unpinned, accepted
    ({}, {"HERMES_MODEL": "envm", "HERMES_INFERENCE_MODEL": "envm"},
     ["cwd", "model", "request_id", "source", "pin_route"], None),  # an inherited model is no typed -m
    ({"provider": "p"}, {}, ["cwd", "model", "provider", "request_id", "source", "pin_route"], None),
])
async def test_the_thin_client_pins_only_a_typed_model(monkeypatch, tmp_path, args, env, parameters, expected):
    """``hermes -z`` / ``chat -q`` through the transport client (``run_gateway_chat``)."""
    from contextlib import asynccontextmanager
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView
    created = []

    class Peer:
        async def rpc(self, method, **params):
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli", "oneshot"], "parameters": parameters}}
            created.append(params)
            return {"stored_session_id": "stored"}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        return 0
    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(GatewayChatView, "run", rendered)
    monkeypatch.chdir(tmp_path)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert await gateway_chat.run_gateway_chat(argparse.Namespace(query="q", quiet=True, **args)) == 0
    assert created[0].get("pin_route") is expected, created[0]


def test_the_classic_clis_new_drops_the_pin():
    """``hermes chat -m m`` pins its session; ``/new`` starts on the profile's default route, unpinned."""
    from hermes_cli.cli_init_mixin import _ROUTE_FLAGS, _gateway_launch
    from hermes_cli.gateway_session_store import GatewaySessionStore

    class Connection:
        def __init__(self):
            self.calls = []

        def call(self, method, **params):
            self.calls.append((method, params))
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli"],
                                           "parameters": ["cwd", "source", "row_cwd", "model", "pin_route"]}}
            return {"stored_session_id": "sid"}

        def state(self, sid):
            return SimpleNamespace(adopt_snapshot=lambda snapshot: None)
    cli = SimpleNamespace(_explicit_model_override=True, model="m", _explicit_provider=None)
    connection = Connection()
    store = GatewaySessionStore(connection)
    store.create(_gateway_launch(cli))
    store.create({k: v for k, v in _gateway_launch(cli).items() if k not in _ROUTE_FLAGS}, record_cwd=False)
    launched, renewed = [params for method, params in connection.calls if method == "session.create"]
    assert launched.get("pin_route") is True and launched["model"] == "m"
    assert "pin_route" not in renewed and "model" not in renewed
