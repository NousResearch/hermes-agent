"""Static contract tests for the WebUI session handoff action."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHAT_PAGE = ROOT / "web" / "src" / "pages" / "ChatPage.tsx"


def test_chat_page_exposes_context_buttons_that_send_shared_commands():
    source = CHAT_PAGE.read_text(encoding="utf-8")

    assert "handleCompress" in source
    assert "handleHandoff" in source
    assert "handleMove" in source
    assert 'sendSlashCommand("/compress")' in source
    assert 'sendSlashCommand("/handoff")' in source
    assert 'sendSlashCommand("/move")' in source
    assert "handleTerminalHandoff" in source
    assert "새 Hermes CLI/터미널 세션" in source
    assert "세션 압축" in source
    assert "새 세션 이어가기 안내 만들기" in source
    assert "이동 준비" in source
    assert "인계문 복사" in source
    assert "터미널용 복사" in source
    assert "세션 이동" in source
