import json
from pathlib import Path

import pytest

from scripts import feishu_inbound_backfill_watchdog as watchdog


def _write_log(home: Path, text: str) -> None:
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "gateway.log").write_text(text, encoding="utf-8")


def _message(mid: str, text: str, *, sender_type: str = "user", create_time: str = "1700000000000") -> dict:
    return {
        "message_id": mid,
        "create_time": create_time,
        "msg_type": "text",
        "sender": {"sender_type": sender_type},
        "body": {"content": json.dumps({"text": text})},
    }


def test_reports_missing_user_message_and_appends_queue(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_test")
    monkeypatch.setattr(watchdog.time, "time", lambda: 1_700_000_100)
    monkeypatch.setattr(watchdog, "_tenant_token", lambda: "token")
    monkeypatch.setattr(
        watchdog,
        "_list_messages",
        lambda *args, **kwargs: [_message("om_missing", "hello from feishu")],
    )
    _write_log(tmp_path, "")

    assert watchdog.run(["--window-seconds", "300", "--grace-seconds", "10"]) == 0
    output = capsys.readouterr().out
    assert "suspected missed message" in output
    assert "om_missing" in output

    queue = tmp_path / "state" / "feishu_inbound_backfill_queue.jsonl"
    records = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "chat_id": "oc_test",
            "create_time": 1_700_000_000,
            "detected_at": 1_700_000_100,
            "message_id": "om_missing",
            "msg_type": "text",
            "preview": "hello from feishu",
            "sender_type": "user",
        }
    ]


def test_ignores_messages_seen_in_rotated_gateway_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_test")
    monkeypatch.setattr(watchdog.time, "time", lambda: 1_700_000_100)
    monkeypatch.setattr(watchdog, "_tenant_token", lambda: "token")
    monkeypatch.setattr(
        watchdog,
        "_list_messages",
        lambda *args, **kwargs: [_message("om_seen", "already processed")],
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "gateway.log.1").write_text("Received raw message type=text message_id=om_seen", encoding="utf-8")

    assert watchdog.run(["--window-seconds", "300", "--grace-seconds", "10", "--dry-run"]) == 0
    assert "OK" in capsys.readouterr().out
    assert not (tmp_path / "state" / "feishu_inbound_backfill_queue.jsonl").exists()


def test_filters_recalled_and_bot_messages_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_test")
    monkeypatch.setattr(watchdog.time, "time", lambda: 1_700_000_100)
    monkeypatch.setattr(watchdog, "_tenant_token", lambda: "token")
    recalled = _message("om_recalled", "This message was recalled")
    bot = _message("om_bot", "bot echo", sender_type="app")
    monkeypatch.setattr(watchdog, "_list_messages", lambda *args, **kwargs: [recalled, bot])
    _write_log(tmp_path, "")

    assert watchdog.run(["--window-seconds", "300", "--grace-seconds", "10", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "checked=0" in output
    assert not (tmp_path / "state" / "feishu_inbound_backfill_queue.jsonl").exists()


def test_suppresses_repeated_alerts_with_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_test")
    monkeypatch.setattr(watchdog.time, "time", lambda: 1_700_000_100)
    monkeypatch.setattr(watchdog, "_tenant_token", lambda: "token")
    monkeypatch.setattr(
        watchdog,
        "_list_messages",
        lambda *args, **kwargs: [_message("om_once", "only alert once")],
    )
    _write_log(tmp_path, "")

    assert watchdog.run(["--window-seconds", "300", "--grace-seconds", "10"]) == 0
    assert watchdog.run(["--window-seconds", "300", "--grace-seconds", "10", "--dry-run"]) == 0
    queue = tmp_path / "state" / "feishu_inbound_backfill_queue.jsonl"
    assert len(queue.read_text(encoding="utf-8").splitlines()) == 1


def test_validates_required_chat_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FEISHU_HOME_CHANNEL", raising=False)
    with pytest.raises(RuntimeError, match="chat id missing"):
        watchdog.run(["--window-seconds", "300", "--grace-seconds", "10"])
