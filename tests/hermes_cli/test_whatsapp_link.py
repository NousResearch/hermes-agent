"""WhatsApp link decisions follow the bridge, not the session directory."""

from hermes_cli.whatsapp_link import (
    discard_whatsapp_session,
    whatsapp_onboarding_link_decision,
)


def test_live_bridge_connection_is_the_only_linked_state():
    assert whatsapp_onboarding_link_decision(True, {"status": "connected"}) == "linked"
    assert whatsapp_onboarding_link_decision(False, {"status": "connected"}) == "watch"
    assert whatsapp_onboarding_link_decision(True, None) == "pair"
    assert whatsapp_onboarding_link_decision(True, {"status": "needs_qr", "qr": "abc"}) == "watch"
    assert whatsapp_onboarding_link_decision(True, {"status": "disconnected"}) == "watch"
    assert whatsapp_onboarding_link_decision(True, {"status": "logged_out"}) == "clear_and_pair"


def test_discard_whatsapp_session_deletes_without_backup(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "creds.json").write_text("{}", encoding="utf-8")
    (session / "pre-key-1.json").write_text("{}", encoding="utf-8")

    discard_whatsapp_session(session)

    assert not session.exists()
    assert not any(path.name.startswith("session") for path in tmp_path.iterdir())


def test_discard_missing_session_is_a_no_op(tmp_path):
    discard_whatsapp_session(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
