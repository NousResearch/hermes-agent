"""Regression coverage for the compact auto-reset notification."""

from pathlib import Path


RUN_PY = Path(__file__).resolve().parents[2] / "gateway" / "run.py"


def test_auto_reset_notice_is_compact_and_omits_session_diagnostics():
    source = RUN_PY.read_text(encoding="utf-8")

    assert 'notice = f"◐ 已开启新会话（{reason_text}）；继续旧话题请用 /resume。"' in source
    assert "Session automatically reset" not in source
    assert "Conversation history cleared" not in source
    assert "Adjust reset timing in config.yaml under session_reset" not in source

    notice_block = source[source.index("if should_notify:") : source.index("await adapter.send(", source.index("if should_notify:"))]
    assert "_reset_notice_session_info" not in notice_block
