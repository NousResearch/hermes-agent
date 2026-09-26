"""Honcho turn guard: what a user turn contributes to memory is bounded.

A ``@file:`` attachment inlines its whole body into the user turn (``--- Attached Context ---``).
Mirrored verbatim, a 1 MB JSON became 47 messages of 25k chars in one Honcho session; one of
them overflowed the deriver's output cap and parked the session's derivation, the summarizer
blew the model context, and a single over-long chunk sank the embedding batch so the session
was unsearchable. The file body is agent-facing context, not memory about the user. Binary
attachments were never affected (they are stubbed upstream) — this makes text behave the same.
"""
from plugins.memory.honcho import turn_guard as tg
from plugins.memory.honcho.client import HonchoClientConfig


def _turn(sentence: str, *blocks: str) -> str:
    return sentence + tg.ATTACHED_MARKER + "\n\n".join(blocks)


BODY = '{"a": 1,\n' + '"b": "x",\n' * 40_000 + '"z": 0}'
TEXT_BLOCK = f"📄 @file:probe.json (17026 tokens)\n```json\n{BODY}\n```"
BINARY_BLOCK = ("📎 @file:photo.png (image/png, 2.1 MB) — binary file, not inlined as text. "
                "It is available on disk at `/x/photo.png`. Use your tools to work with it.")


def test_plain_turn_passes_through_unchanged():
    s = "how do I make VMs migratable so I can patch the host?"
    assert tg.guard_user_turn(s) == s


def test_attachment_body_is_replaced_by_a_stub_and_the_sentence_survives():
    out = tg.guard_user_turn(_turn("attached is the session that triaged your outage", TEXT_BLOCK))
    assert out.startswith("attached is the session that triaged your outage")
    assert "[attached: @file:probe.json, ~17,026 tokens, contents not stored in memory]" in out
    assert '"b": "x"' not in out
    assert len(out) < len(TEXT_BLOCK) // 100


def test_upstream_binary_stub_keeps_its_description_and_drops_guidance():
    out = tg.guard_user_turn(_turn("photo", BINARY_BLOCK))
    assert out.splitlines()[-1] == "📎 @file:photo.png (image/png, 2.1 MB)"
    assert "Use your tools" not in out


def test_big_paste_is_capped_with_an_honest_marker():
    paste = "log line\n" * 5000
    out = tg.guard_user_turn(paste, turn_max_chars=8000)
    assert len(out) < len(paste)
    assert out.endswith("more characters not stored in memory]")
    dropped = int(out.rsplit("[… ", 1)[1].split(" ")[0].replace(",", ""))
    assert dropped == len(paste) - 8000


def test_zero_cap_disables_the_paste_limit_but_not_attachment_stubbing():
    paste = "x" * 20_000
    assert tg.guard_user_turn(paste, turn_max_chars=0) == paste
    assert '"b": "x"' not in tg.guard_user_turn(_turn("hi", TEXT_BLOCK), turn_max_chars=0)


def test_turn_max_chars_is_configurable_and_defaults_sane():
    cfg = HonchoClientConfig()
    assert 0 < cfg.turn_max_chars <= cfg.message_max_chars
