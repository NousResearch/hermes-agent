"""A forwarded mutation acts as the dashboard's authenticated principal, not the socket UID (#138915).

The standalone dashboard's owner-forward mints its ticket over the private control socket, whose
POSIX/pipe peer gate trusts the same user. The minted grant must carry the dashboard's
authenticated subject when one is named — ``authorize_history`` compares that subject against a
cold row's stored owner — while an unverified peer, or a malformed subject, still fails closed.
"""

import json

from gateway.control_socket import GatewayControlServer
from gateway.runtime_bootstrap import TicketStore

FORWARDED = 'auth:v1:["basic","","wweic"]'


def _server(tmp_path):
    server = GatewayControlServer(tmp_path)
    server.ticket_store = TicketStore("inst-1", frozenset({str(tmp_path)}))
    return server


def _mint_line(profile, subject=None, *, with_subject=True):
    params = {
        "profile_id": str(profile),
        "instance_id": "inst-1",
        "purpose": "interactive",
    }
    if with_subject:
        params["subject"] = subject
    return json.dumps(
        {"protocol": 1, "id": 7, "verb": "session-ticket", "params": params}
    ).encode()


def test_a_verified_peer_mints_for_the_named_dashboard_principal(tmp_path):
    server = _server(tmp_path)
    response = json.loads(
        server.handle_request_line(
            _mint_line(tmp_path, FORWARDED), peer_subject="uid:501"
        )
    )
    assert response["ok"] is True
    grant = server.ticket_store.redeem(
        response["result"]["ticket"], profile_id=None, purpose="interactive"
    )
    assert grant["subject"] == FORWARDED
    # authorize_history's capability gate: the forwarded actor must still carry it.
    assert "session:create" in grant["capabilities"]


def test_an_unverified_peer_cannot_name_a_principal(tmp_path):
    response = json.loads(
        _server(tmp_path).handle_request_line(
            _mint_line(tmp_path, FORWARDED), peer_subject=None
        )
    )
    assert response["ok"] is False
    assert "authenticated runtime bootstrap unavailable" in response["error"]


def test_a_malformed_forwarded_subject_fails_closed(tmp_path):
    for bad in ("", "x" * 257, 501, None):
        response = json.loads(
            _server(tmp_path).handle_request_line(
                _mint_line(tmp_path, bad), peer_subject="uid:501"
            )
        )
        assert response["ok"] is False, bad
        assert "invalid bootstrap subject" in response["error"]


def test_a_forwarded_subject_matching_the_row_owner_takes_the_mutation(tmp_path):
    """The reviewer-repro shape: a stored ``desktop`` row owned by a dashboard account is
    deletable by a forwarded actor acting as that account, and refused for any other."""
    from gateway.session_contract import Principal
    from hermes_state import SessionDB
    from hermes_state_mutation_binding import authorize_history

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("cold-desktop", source="desktop", user_id=FORWARDED)
        same = Principal(
            FORWARDED,
            str(tmp_path),
            frozenset({"session:read", "session:create"}),
            "native-http:x",
        )
        other = Principal(
            'auth:v1:["basic","","someone-else"]',
            str(tmp_path),
            frozenset({"session:read", "session:create"}),
            "native-http:y",
        )
        with db._read_ctx() as conn:
            assert authorize_history(conn, same, "cold-desktop") is True
            try:
                authorize_history(conn, other, "cold-desktop")
            except Exception as exc:
                assert getattr(exc, "reason", None) == "permission_denied"
            else:
                raise AssertionError("a foreign principal must not take the mutation")
    finally:
        db.close()
