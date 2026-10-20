from types import SimpleNamespace

from gateway.session_persistence import SessionPersistenceMixin


def test_stale_route_warning_does_not_claim_the_gateway_crashed(caplog):
    manager = SessionPersistenceMixin.__new__(SessionPersistenceMixin)
    entry = SimpleNamespace(session_id="ended-session", origin=None)

    verdict = manager._stale_entry_verdict(
        "agent:main:telegram:group:1", entry, {"end_reason": "webhook_complete"}
    )

    assert verdict == "prune"
    assert "left by an earlier gateway process" in caplog.text
    assert "crashed gateway" not in caplog.text
