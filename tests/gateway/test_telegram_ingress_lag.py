from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gateway.platforms.telegram import TelegramAdapter


def _msg(*, date, message_id=123):
    return SimpleNamespace(
        date=date,
        message_id=message_id,
        chat=SimpleNamespace(id=441846490),
        from_user=SimpleNamespace(id=441846490),
    )


def _update(update_id=987):
    return SimpleNamespace(update_id=update_id)


def test_telegram_ingress_lag_logs_fresh_update_at_info(caplog, monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_INGRESS_LAG_WARN_SECONDS", "30")
    adapter = object.__new__(TelegramAdapter)
    msg = _msg(date=datetime.now(timezone.utc) - timedelta(seconds=1))

    with caplog.at_level("INFO", logger="gateway.platforms.telegram"):
        adapter._log_ingress_lag(_update(), msg, kind="command")

    records = [r for r in caplog.records if "Ingress command" in r.message]
    assert records
    assert records[-1].levelname == "INFO"
    assert "update_id=987" in records[-1].message
    assert "message_id=123" in records[-1].message
    assert "lag=" in records[-1].message


def test_telegram_ingress_lag_warns_for_stale_update(caplog, monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_INGRESS_LAG_WARN_SECONDS", "30")
    adapter = object.__new__(TelegramAdapter)
    msg = _msg(date=datetime.now(timezone.utc) - timedelta(seconds=65))

    with caplog.at_level("WARNING", logger="gateway.platforms.telegram"):
        adapter._log_ingress_lag(_update(update_id=654), msg, kind="text")

    records = [r for r in caplog.records if "Ingress text" in r.message]
    assert records
    assert records[-1].levelname == "WARNING"
    assert "update_id=654" in records[-1].message
    assert "chat_id=441846490" in records[-1].message


def test_telegram_message_datetime_treats_naive_dates_as_utc():
    naive = datetime(2026, 6, 20, 0, 49, 40)

    parsed = TelegramAdapter._telegram_message_datetime(_msg(date=naive))

    assert parsed == naive.replace(tzinfo=timezone.utc)
