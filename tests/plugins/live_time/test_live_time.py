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


def test_weekday_label_is_locale_neutral() -> None:
    text = _on_pre_llm_call()["context"]
    m = re.search(r"\(\s*Weekday \d/7, ([A-Za-z]{3})\)", text)
    assert m and m.group(1) in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}, (
        f"weekday label not locale-neutral: {text!r}"
    )


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
    # Inline comment + quoted value must not defeat parsing.
    cfg.write_text('timezone: "Asia/Tokyo"  # trailing comment\n', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    text = _on_pre_llm_call()["context"]
    assert "TZ Asia/Tokyo" in text


def test_nested_timezone_key_is_ignored(monkeypatch, tmp_path) -> None:
    """A timezone key nested under another section must not be picked up."""
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "model: deepseek\n"
        "plugins:\n"
        "  entries:\n"
        "    time-gap:\n"
        "      timezone: Europe/Berlin\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.live_time import _resolve_timezone_name

    assert _resolve_timezone_name() == ""


def test_invalid_timezone_falls_back_to_system_local(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    monkeypatch.setenv("HERMES_TIMEZONE", "Mars/Olympus_Mons")
    out = _on_pre_llm_call()
    assert out is not None  # invalid tz must not raise
    text = out["context"]
    assert "TZ Mars/Olympus_Mons" not in text
    assert re.search(r"TZ UTC[+-]\d+", text), f"no system-local fallback in {text!r}"
