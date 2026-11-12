"""The public control-socket contract, pinned by the JSON vectors clients in any language replay.

``apps/shared/src/node/control-socket.vectors.json`` is the single source: each vector builds a
gateway control server, sends one request line and compares the one reply line. The
``session-ticket`` vectors describe the verb as it already is (they pass unchanged against a tree
without ``resolve-profile``); the ``resolve-profile`` vectors pin the lookup verb.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

from hermes_agent.gateway.control_socket import GatewayControlServer
from hermes_agent.gateway.runtime_bootstrap import TicketStore

pytestmark = pytest.mark.platforms("posix")

_VECTORS_FILE = Path(__file__).resolve().parents[2] / "apps" / "shared" / "src" / "node" / "control-socket.vectors.json"
_DOC = json.loads(_VECTORS_FILE.read_text(encoding="utf-8"))
_BOOT_SHA = "1" * 40
# Reply-only wildcards: values the server generates.
_WILDCARDS = {
    "$TICKET": lambda value: isinstance(value, str) and 0 < len(value) <= 256,
    "$INSTALL_SHA": lambda value: value is None or (isinstance(value, str) and bool(value)),
    "$VERBS": lambda value: isinstance(value, list) and all(isinstance(v, str) for v in value),
}


def _layout(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "mux" / ".hermes"
    (root / "profiles" / "work").mkdir(parents=True)
    (root / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    solo = tmp_path / "solo"
    solo.mkdir()
    return {"$OTHER_INSTANCE": "instance-before-restart", "$INSTANCE": "instance-now", "$BOOT_SHA": _BOOT_SHA,
            "$FOREIGN": str((tmp_path / "elsewhere").resolve()), "$HOME": str(solo.resolve()),
            "$ROOT": str(root.resolve()), "$SEC": str((root / "profiles" / "work").resolve())}


def _fill(value, names: dict[str, str]):
    if isinstance(value, str):
        if value in _WILDCARDS:
            return value
        for name in sorted(names, key=len, reverse=True):
            value = value.replace(name, names[name])
        return value
    if isinstance(value, list):
        return [_fill(v, names) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, names) for k, v in value.items()}
    return value


def _matches(expected, actual) -> bool:
    if isinstance(expected, str) and expected in _WILDCARDS:
        return _WILDCARDS[expected](actual)
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(expected) == set(actual)
                and all(_matches(expected[k], actual[k]) for k in expected))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_matches(e, a) for e, a in zip(expected, actual)))
    return type(expected) is type(actual) and expected == actual


def _server(spec: dict) -> GatewayControlServer:
    server = GatewayControlServer(Path(spec["home"]))
    server.ticket_store = TicketStore(spec["instance_id"], frozenset(spec["profile_ids"]))
    if "identify" in spec and "resolve-profile" not in spec.get("without_verbs", []):
        from hermes_agent.gateway.control_resolve import resolve_profile_handler
        server.resolve_handler = resolve_profile_handler(lambda: dict(spec["identify"]))
    if spec.get("update_marker"):
        # A marker still being written fences for its grace period (update_lock.EMPTY_MARKER_GRACE_SECONDS).
        (Path(spec["home"]) / ".hermes-update-in-progress").touch()
    return server


def test_the_vectors_file_is_well_formed():
    assert _DOC["format"] == 1
    names = [v["name"] for v in _DOC["vectors"]]
    assert len(names) == len(set(names))
    assert {n.split(":", 1)[0] for n in names} == {"ticket", "envelope", "resolve", "legacy"}
    for vector in _DOC["vectors"]:
        assert ("request" in vector) != ("request_raw" in vector), vector["name"]
        outcome = vector.get("client", {}).get("outcome")
        assert outcome is None or outcome in {"grant", "endpoint", "not_attachable", "legacy"} or outcome.startswith("refused:")
        request = vector.get("request")
        # The public forms: no forwarded subject and no host scope on session-ticket.
        if isinstance(request, dict) and request.get("verb") == "session-ticket" and isinstance(request.get("params"), dict):
            assert not {"subject", "scope"} & set(request["params"]), vector["name"]


@pytest.mark.parametrize("vector", _DOC["vectors"], ids=[v["name"] for v in _DOC["vectors"]])
def test_vector(vector, tmp_path):
    names = _layout(tmp_path)
    spec = _fill(vector["server"], names)
    server = _server(spec)
    raw = (_fill(vector["request_raw"], names) if "request_raw" in vector
           else json.dumps(_fill(vector["request"], names)))
    peer = f"uid:{os.getuid()}" if spec["peer"] == "same-user" else None  # windows-footgun: ok — posix-marked module
    line = server.handle_request_line(raw.encode("utf-8"), peer)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    reply = json.loads(line)
    expected = _fill(vector["reply"], names)
    prefix = expected.pop("error_prefix", None)
    if prefix is not None:
        assert isinstance(reply.get("error"), str) and reply["error"].startswith(prefix)
        reply.pop("error")
    assert _matches(expected, reply), json.dumps(reply, indent=1)


def test_a_resolved_endpoint_mints_one_single_use_ticket_over_the_real_socket(tmp_path):
    """The client sequence end to end: resolve-profile, then session-ticket with the answer's
    identity on the socket that answered, then the WebSocket admission's one redeem."""
    from hermes_agent.gateway.control_resolve import resolve_profile_handler
    from hermes_agent.hermes_cli.gateway_client import private_control_query

    names = _layout(tmp_path)
    home = names["$HOME"]
    identify = _fill(next(v for v in _DOC["vectors"] if v["name"] == "resolve: a ready standalone owner")["server"]["identify"], names)
    server = GatewayControlServer(Path(home))
    server.ticket_store = store = TicketStore(names["$INSTANCE"], frozenset({home}))
    server.resolve_handler = resolve_profile_handler(lambda: dict(identify))

    def ask(verb, params):
        line = json.dumps({"protocol": 1, "id": 1, "verb": verb, "params": params}).encode() + b"\n"
        return json.loads(private_control_query(Path(home), line))

    async def scenario():
        assert await server.start()
        try:
            resolved = await asyncio.to_thread(ask, "resolve-profile", {"home": home})
            endpoint = resolved["result"]["endpoint"]
            granted = await asyncio.to_thread(ask, "session-ticket", {
                "profile_id": endpoint["profile_id"], "instance_id": endpoint["instance_id"], "purpose": "interactive"})
            ticket = granted["result"]["ticket"]
            # The WebSocket admission redeems it once (stop() revokes whatever is left).
            first = store.redeem(ticket, profile_id=endpoint["profile_id"], purpose="interactive")
            with pytest.raises(PermissionError, match="invalid bootstrap ticket"):
                store.redeem(ticket, profile_id=endpoint["profile_id"], purpose="interactive")
            return resolved, first
        finally:
            await server.stop()

    resolved, grant = asyncio.run(scenario())
    assert resolved["ok"] is True and resolved["result"]["attachable"] is True
    assert resolved["result"]["endpoint"]["profile_id"] == grant["profile_id"] == home
