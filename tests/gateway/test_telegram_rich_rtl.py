"""Rich messages carry InputRichMessage.is_rtl for predominantly RTL text (issue #78715)."""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter

AR = "مرحبا بك في الخدمة، كيف يمكنني مساعدتك اليوم؟"
HE = "שלום, איך אפשר לעזור לך היום?"
MIXED = "لتثبيت الحزمة شغّل الأمر pip install requests ثم أعد تشغيل الخادم بعد ذلك"
CODE = "```python\n" + "print('hello world')\n" * 50 + "```"


@pytest.fixture()
def adapter():
    return object.__new__(TelegramAdapter)


@pytest.mark.parametrize("text", [AR, HE, MIXED, AR + "\n\n" + CODE])
def test_rtl_sets_is_rtl(adapter, text):
    assert adapter._rich_message_payload(text)["is_rtl"] is True


@pytest.mark.parametrize("text", ["Hello, how can I help you today?", 'The word "مرحبا" means hello in this long English sentence.'])
def test_ltr_omits_is_rtl(adapter, text):
    assert "is_rtl" not in adapter._rich_message_payload(text)


def test_needs_rich_rendering(adapter):
    assert adapter._needs_rich_rendering(AR)
    assert not adapter._needs_rich_rendering("Hello, how can I help you today?")


def test_markdown_normalization_still_applied(adapter):
    payload = adapter._rich_message_payload(AR + "\n" + AR)
    assert payload["markdown"] == AR + "  \n" + AR
