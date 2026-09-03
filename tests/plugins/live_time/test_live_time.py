"""Unit tests for the live-time plugin."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from plugins.live_time import _on_pre_llm_call, register


def test_on_pre_llm_call_returns_live_timestamp_context() -> None:
    out = _on_pre_llm_call()
    assert out is not None
    text = out["context"]
    assert "[LIVE-TIME] Now:" in text

    m = re.search(r"Now: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    assert m, f"missing timestamp in {text!r}"
    parsed = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

    # Compare in UTC using the offset reported in the text, so the assertion
    # holds regardless of which timezone the plugin resolved to.
    mo = re.search(r"UTC([+-]\d+)", text)
    assert mo, f"missing UTC offset in {text!r}"
    offset_h = int(mo.group(1))
    parsed_utc = parsed - timedelta(hours=offset_h)
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((now_utc - parsed_utc).total_seconds()) < 30


def test_context_marks_itself_authoritative() -> None:
    text = _on_pre_llm_call()["context"]
    assert "Use THIS as the authoritative current time" in text
    assert "Conversation started" in text


def test_register_hooks_pre_llm_call() -> None:
    ctx = MagicMock()
    register(ctx)
    ctx.register_hook.assert_called_once_with("pre_llm_call", _on_pre_llm_call)


def test_env_timezone_is_respected(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
    text = _on_pre_llm_call()["context"]
    assert "TZ America/New_York" in text
    # The stamped time must actually be New York time, not system-local.
    from zoneinfo import ZoneInfo

    m = re.search(r"Now: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    assert m, f"missing timestamp in {text!r}"
    parsed = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    now_ny = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    assert abs((now_ny - parsed).total_seconds()) < 30


def test_config_timezone_is_respected(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text('timezone: "Asia/Tokyo"\n', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    text = _on_pre_llm_call()["context"]
    assert "TZ Asia/Tokyo" in text
    # Tokyo is +1h from the test machine (China Standard Time) — the stamped
    # timestamp must follow the configured zone, not the system one.
    from zoneinfo import ZoneInfo

    m = re.search(r"Now: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    assert m, f"missing timestamp in {text!r}"
    parsed = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    now_tokyo = datetime.now(ZoneInfo("Asia/Tokyo")).replace(tzinfo=None)
    assert abs((now_tokyo - parsed).total_seconds()) < 30
