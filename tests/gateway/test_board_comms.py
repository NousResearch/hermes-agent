import json

from gateway.board_comms import evaluate_send, is_ack_only, log_suppression


def test_ack_only_detection():
    assert is_ack_only("Acknowledged.")
    assert is_ack_only("Prime Hermes — standing by.")
    assert not is_ack_only("Status: Foundry Hermes is degraded; cleanup routed.")


def test_non_board_chat_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_BOARD_COMMS_STATE_DIR", str(tmp_path))
    decision = evaluate_send(chat_id="123", content="Acknowledged.", metadata={})
    assert decision.allow


def test_targeted_other_bot_stop_kill_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_BOARD_COMMS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_BOARD_CHAT_IDS", "-1003817293915")
    decision = evaluate_send(
        chat_id="-1003817293915",
        content="Acknowledged. Prime Hermes standing by.",
        metadata={
            "thread_id": "1",
            "board_context": {
                "inbound_text": "@foundry_hermes_bot stop kill",
                "is_bot": False,
            },
        },
    )
    assert not decision.allow
    assert decision.reason == "SUPPRESS_TARGETED_OTHER_BOT"


def test_bot_origin_ack_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_BOARD_COMMS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_BOARD_CHAT_IDS", "-1003817293915")
    decision = evaluate_send(
        chat_id="-1003817293915",
        content="Received. Standing by.",
        metadata={"board_context": {"inbound_text": "Acknowledged.", "is_bot": True}},
    )
    assert not decision.allow
    assert decision.reason == "SUPPRESS_BOT_TO_BOT_ACK"


def test_board_ack_only_suppressed_and_logged(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_BOARD_COMMS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_BOARD_CHAT_IDS", "-1003817293915")
    metadata = {"thread_id": "1", "board_context": {"inbound_text": "Ping", "is_bot": False}}
    decision = evaluate_send(chat_id="-1003817293915", content="Acknowledged.", metadata=metadata, now=1000)
    assert not decision.allow
    assert decision.reason == "SUPPRESS_ACK_ONLY"

    log_suppression(chat_id="-1003817293915", content="Acknowledged.", reason=decision.reason, metadata=metadata)
    log_path = tmp_path / "suppression.jsonl"
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["reason"] == "SUPPRESS_ACK_ONLY"
    assert row["thread_id"] == "1"
