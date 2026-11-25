"""``hermes peer`` reads a peer's "no such profile" 404 by its typed code, not by its English sentence.

The ``/p/<profile>/`` prefix middleware answers a profile the gateway does not have with
``{"error": "Unknown or unconfigured profile", "code": "profile_unknown"}`` (that body is pinned in
``tests/gateway/test_multiplex_api_server_routing.py``). Clients key on the code, so rewording the
sentence cannot change where ``box/hermes`` goes, and another code beside the old sentence is not the
verdict. A peer older than the code sends the sentence alone, which still counts.
"""

import threading
from http.server import HTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_cmd
from tests.hermes_cli.test_peer_cmd import _ProfilePeer

_SENTENCE = "Unknown or unconfigured profile"
_AMBIGUOUS = "box/default"
_RELAYED = "rejected the request (HTTP 404)"


@pytest.fixture
def refusing_peer(monkeypatch):
    """A loopback peer serving no profile at all: every ``/p/<profile>/`` request gets its 404 body."""
    monkeypatch.setattr(_ProfilePeer, "served", set())
    monkeypatch.setattr(_ProfilePeer, "hits", [])
    server = HTTPServer(("127.0.0.1", 0), _ProfilePeer)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"box": {"url": f"http://127.0.0.1:{server.server_port}"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "secret-key-123456")
    try:
        yield
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("body,verdict", [
    # Typed verdict: the code decides, whatever the sentence says.
    ({"error": "No profile by that name here", "code": "profile_unknown"}, _AMBIGUOUS),
    ({"error": _SENTENCE, "code": "profile_unknown"}, _AMBIGUOUS),
    # The old sentence beside another code is not the verdict.
    ({"error": _SENTENCE, "code": "route_disabled"}, _RELAYED),
    # A peer older than the code: the exact sentence alone still counts.
    ({"error": _SENTENCE}, _AMBIGUOUS),
    ({"error": "Unknown profile"}, _RELAYED),
])
def test_hermes_handle_reads_the_peers_typed_unknown_profile_verdict(
        refusing_peer, monkeypatch, capsys, body, verdict):
    monkeypatch.setattr(_ProfilePeer, "refusal", body)
    rc = peer_cmd.cmd_peer(SimpleNamespace(peer_action="dm", target="box/hermes", message="hi", json=True))
    err = capsys.readouterr().err
    assert rc == 1
    other = _RELAYED if verdict == _AMBIGUOUS else _AMBIGUOUS
    assert verdict in err and other not in err, err
    # Every request addressed the profile asked for: nothing is ever rerouted to /p/default.
    assert {hit.split("/")[2] for hit in _ProfilePeer.hits} == {"hermes"}
