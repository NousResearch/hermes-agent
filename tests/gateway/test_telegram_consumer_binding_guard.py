"""Regression guard: the gateway.platforms.telegram consumer's ParseMode.MARKDOWN_V2
must never resolve to the leaked plain string "MarkdownV2".

Cross-file pollution this guards (fixed in test_dm_topics.py): that module installs
a mock telegram whose ``constants.ParseMode.MARKDOWN_V2`` is the plain STRING
"MarkdownV2", then pops + reimports the consumer, rebinding its module-global
``ParseMode`` to that mock-constants object. Without the module-teardown restore,
every later gateway test in a single-process run sees ``MARKDOWN_V2 == "MarkdownV2"``
(a plain str), and assertions of the form ``"MARKDOWN_V2" in repr(parse_mode)`` fail
(``repr("MarkdownV2") == "'MarkdownV2'"`` — no ``MARKDOWN_V2`` token).

The invariant (matches what the real telegram tests rely on): the consumer's
``ParseMode.MARKDOWN_V2`` repr CONTAINS ``MARKDOWN_V2`` — true for the gateway test
env's normal ``MagicMock`` member (``<MagicMock name='mock.ParseMode.MARKDOWN_V2'>``)
AND for the real ``StringEnum`` member (``<ParseMode.MARKDOWN_V2>``), but FALSE for
the leaked plain string. We assert on ``repr`` (the member-name carrier), NOT ``==``:
``ParseMode`` is a ``StringEnum`` (str subclass), so ``MARKDOWN_V2 == "MarkdownV2"``
is True for BOTH the real enum AND the leaked string — a ``==`` check would pass
against the poison and could not RED-prove the fix.
"""
import gateway.platforms.telegram as _consumer


def test_consumer_markdown_v2_not_leaked_plain_string():
    pm = _consumer.ParseMode
    mv = getattr(pm, "MARKDOWN_V2", None)

    # The leaked-poison signature: a plain str whose repr loses the member name.
    assert not (isinstance(mv, str) and repr(mv) == "'MarkdownV2'"), (
        "consumer ParseMode.MARKDOWN_V2 is the leaked plain string 'MarkdownV2' "
        "(cross-file pollution from a telegram-mock module that didn't restore "
        f"the consumer binding): {mv!r}"
    )
    # The positive invariant the real telegram tests rely on: repr carries the name.
    assert "MARKDOWN_V2" in repr(mv), (
        f"consumer ParseMode.MARKDOWN_V2 repr lost the member name: {mv!r}"
    )
