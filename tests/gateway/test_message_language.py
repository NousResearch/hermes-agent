"""Display language follows a message across concurrent tasks and restored origins."""
import asyncio

import pytest

from agent.display import get_tool_verb
from agent.i18n import get_language, t, use_language
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.LOCAL)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="reply")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_concurrent_and_drained_messages_render_own_language(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    adapter = CaptureAdapter()
    entered = {lang: asyncio.Event() for lang in ("ko", "en")}
    observed = {}

    async def render(event):
        lang = event.source.language
        entered[lang].set()
        await entered["en" if lang == "ko" else "ko"].wait()
        # These are the real progress label and worker-context paths, not translated fixtures.
        observed[lang] = (get_language(), await asyncio.to_thread(get_tool_verb, "terminal"))

    adapter.set_message_handler(render)
    sources = [SessionSource(Platform.LOCAL, lang, language=lang) for lang in ("ko", "en")]
    with use_language("ja"):
        await asyncio.wait_for(asyncio.gather(*(
            adapter._process_message_background(MessageEvent(text="work", source=s), build_session_key(s))
            for s in sources)), timeout=10)
        assert get_language() == "ja"
        # A drained message must rebind rather than inherit the preceding task's Japanese.
        await adapter._process_message_background(
            MessageEvent(text="again", source=sources[0]), build_session_key(sources[0]))
        assert get_language() == "ja"
    assert observed == {lang: (lang, t("display.verb.terminal", lang=lang)) for lang in ("ko", "en")}
    assert get_language() == "en"


def test_restored_language_and_invalid_locale_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    source = SessionSource(Platform.LOCAL, "chat", language="ko-KR")
    restored = SessionSource.from_dict(source.to_dict())
    with use_language(restored.language):
        assert get_language() == "ko"
        assert t("display.verb.terminal") == t("display.verb.terminal", lang="ko")
        assert t("display.verb.terminal", lang="en") != t("display.verb.terminal")
        with use_language("not-a-catalog"):
            assert get_language() == "fr"
        with use_language(None):
            assert get_language() == "fr"
        assert get_language() == "ko"
    assert get_language() == "fr"
    assert "language" not in SessionSource(Platform.LOCAL, "old").to_dict()
