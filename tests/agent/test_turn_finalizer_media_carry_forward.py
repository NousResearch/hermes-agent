"""Regression test: MEDIA: tags must survive a trailing tool call.

Only the FINAL assistant message of a turn is dispatched to the chat platform.
When the model emits ``MEDIA:/abs/path`` and then calls any tool (storing a
memory is the common case), the media-bearing message is demoted to an
intermediate turn and is never sent — while the new final message typically
claims the files went out.  ``_carry_forward_media_tags`` re-appends any tag
emitted earlier in the same turn that did not survive into the final text.
"""

from agent.turn_finalizer import _carry_forward_media_tags


def _turn(*assistant_texts, user="send me the report"):
    msgs = [{"role": "user", "content": user}]
    msgs += [{"role": "assistant", "content": t} for t in assistant_texts]
    return msgs


def test_tag_lost_to_trailing_tool_call_is_reattached():
    messages = _turn("Here it is.\nMEDIA:/tmp/report.pdf", "Sent 1 file just now.")
    out = _carry_forward_media_tags("Sent 1 file just now.", messages)
    assert "MEDIA:/tmp/report.pdf" in out
    assert out.startswith("Sent 1 file just now.")


def test_multiple_tags_keep_their_order():
    messages = _turn(
        "Both attached.\nMEDIA:/tmp/a.pdf\nMEDIA:/tmp/b.xlsx",
        "Sent 2 files.",
    )
    out = _carry_forward_media_tags("Sent 2 files.", messages)
    assert out.index("MEDIA:/tmp/a.pdf") < out.index("MEDIA:/tmp/b.xlsx")


def test_tag_already_final_is_not_duplicated():
    messages = _turn("Here it is.\nMEDIA:/tmp/report.pdf")
    final = "Here it is.\nMEDIA:/tmp/report.pdf"
    out = _carry_forward_media_tags(final, messages)
    assert out.count("MEDIA:/tmp/report.pdf") == 1


def test_tag_from_a_previous_turn_is_not_resent():
    messages = [
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "old one\nMEDIA:/tmp/old.pdf"},
        {"role": "user", "content": "second ask"},
        {"role": "assistant", "content": "nothing to attach this time"},
    ]
    out = _carry_forward_media_tags("nothing to attach this time", messages)
    assert "MEDIA:/tmp/old.pdf" not in out


def test_no_media_leaves_response_untouched():
    messages = _turn("just a plain answer")
    assert _carry_forward_media_tags("just a plain answer", messages) == "just a plain answer"


def test_empty_messages_are_safe():
    assert _carry_forward_media_tags("hello", []) == "hello"
    assert _carry_forward_media_tags("hello", None) == "hello"
