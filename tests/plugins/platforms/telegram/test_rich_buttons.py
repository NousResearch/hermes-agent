"""Bot API 10.3 native buttons inside rich messages (opt-in ``rich_buttons``).

``sendRichMessage`` takes no ``reply_markup`` and MarkdownV2 has no button syntax, so before this
a rich reply that needed buttons had to degrade or spend a second message. With the extra on, a
```buttons fence is rewritten into ``<tg-button-row>``/``<tg-button>`` rows by the rich grammar.
With the extra absent the payload must stay byte-identical to the pre-10.3 behaviour (the fence
stays a plain code block), which the upstream payloads pin.
"""
from __future__ import annotations

from telegram import Bot

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.rich_buttons import has_button_fence, render_rich_buttons

FENCE = """**Esito**

```buttons
Approva | success | callback:approve_42
Rifiuta | danger | callback:reject_42
Copia | | copy:testo
Spento | | disabled
```
"""


def _make_adapter(**extra) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=extra))
    adapter._bot = Bot("123:ABC")
    adapter.format_message = lambda content, **kwargs: content
    return adapter


# ---------------------------------------------------------------------------
# fence -> tags
# ---------------------------------------------------------------------------


def test_no_fence_is_untouched():
    text = "solo testo, nessun bottone"
    assert render_rich_buttons(text) == text
    assert has_button_fence(text) is False


def test_fence_renders_one_row_per_line():
    out = render_rich_buttons(FENCE)
    assert "```" not in out
    assert '<tg-button-row align="left"><tg-button type="callback_data" style="success" data="approve_42">Approva</tg-button></tg-button-row>' in out
    assert '<tg-button type="callback_data" style="danger" data="reject_42">Rifiuta</tg-button>' in out
    assert '<tg-button type="copy_text" text="testo">Copia</tg-button>' in out
    assert '<tg-button type="disabled">Spento</tg-button>' in out
    assert out.startswith("**Esito**\n")
    assert out.count("<tg-button-row") == 4


def test_plus_prefix_shares_the_previous_row():
    out = render_rich_buttons("```buttons\nA | | url:https://a.example\n+ B | | url:https://b.example\n```")
    assert out.count("<tg-button-row") == 1
    assert out.count("<tg-button ") == 2


def test_url_button_carries_style_and_url():
    out = render_rich_buttons("```buttons\nDocs | link | url:https://core.telegram.org/bots/api\n```")
    assert '<tg-button type="url" style="link" url="https://core.telegram.org/bots/api">Docs</tg-button>' in out


def test_malformed_lines_are_dropped_not_rendered():
    out = render_rich_buttons(
        "```buttons\n"
        "ok | | disabled\n"
        "manche | solo-etichetta\n"
        "stile-cattivo | sparkle | callback:x\n"
        "schema-cattivo | | url:javascript:alert(1)\n"
        "no-label | | \n"
        "```"
    )
    assert out.count("<tg-button ") == 1
    assert "javascript:" not in out
    assert "stile-cattivo" not in out


def test_callback_over_64_bytes_is_rejected():
    out = render_rich_buttons(f"```buttons\nlungo | | callback:{'x' * 65}\n```")
    assert out == ""


def test_fence_without_valid_button_disappears():
    out = render_rich_buttons("prima\n```buttons\nspazzatura\n```\ndopo")
    assert out == "prima\ndopo"


def test_unclosed_fence_leaves_text_untouched():
    text = "```buttons\nA | | disabled\nsenza chiusura"
    assert render_rich_buttons(text) == text


def test_labels_and_attributes_are_escaped():
    out = render_rich_buttons("```buttons\n<click> & co | | copy:<a>&\"b\"\n```")
    assert "&lt;click&gt; &amp; co" in out
    assert '<tg-button' in out and "<click>" not in out
    assert 'text="&lt;a&gt;&amp;&quot;b&quot;"' in out


def test_fence_inside_longer_message_keeps_surrounding_text():
    out = render_rich_buttons("intro\n\n```buttons\nA | | disabled\n```\n\noutro")
    assert out.startswith("intro\n")
    assert out.rstrip().endswith("outro")
    assert "```" not in out


# ---------------------------------------------------------------------------
# adapter wiring
# ---------------------------------------------------------------------------


def test_default_off_payload_keeps_the_fence_and_rich_is_not_forced():
    adapter = _make_adapter()
    assert adapter._rich_buttons_enabled is False
    payload = adapter._rich_message_payload(FENCE)
    assert plugin_fence_in(payload["markdown"])
    assert adapter._needs_rich_rendering(FENCE) is False


def test_enabled_payload_carries_button_rows():
    adapter = _make_adapter(rich_buttons=True)
    assert adapter._rich_buttons_enabled is True
    payload = adapter._rich_message_payload(FENCE)
    assert "<tg-button-row" in payload["markdown"]
    assert plugin_fence_in(payload["markdown"]) is False
    assert adapter._needs_rich_rendering(FENCE) is True


def test_enabled_flag_is_string_coerced():
    assert _make_adapter(rich_buttons="true")._rich_buttons_enabled is True
    assert _make_adapter(rich_buttons="off")._rich_buttons_enabled is False


def plugin_fence_in(markdown: str) -> bool:
    """The fence survives as a plain code block when the extra is off."""
    return "```buttons" in markdown
