"""Raw <TOOLCALL> markup never reaches the user or re-enters the model via a reply (#61217)."""

from agent.agent_runtime_helpers import strip_think_blocks
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_final_response_drops_closed_and_cut_toolcall_blocks():
    """Visible text sent to a platform keeps the prose and loses every <TOOLCALL> block,
    including one cut mid-JSON with no closing tag."""
    closed = 'Checking.\n<TOOLCALL>[{"name": "terminal", "arguments": {}}]</TOOLCALL>\nDone.'
    cut = 'Looking now.\n<TOOLCALL>{"name": "browser_press", "argum'

    assert strip_think_blocks(None, closed).split() == ["Checking.", "Done."]
    assert strip_think_blocks(None, cut).strip() == "Looking now."


def test_reply_to_a_message_with_toolcall_markup_is_redacted_in_the_turn():
    """A user replying to an already-sent bot message that carries the markup gets a reply
    pointer with the markup replaced and the surrounding text intact."""
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm", user_id="42")
    quoted = 'Done!\n<TOOLCALL>{"name": "terminal", "arguments": {"command": "ls'
    event = MessageEvent(text="什么情况", source=source, reply_to_message_id="42", reply_to_text=quoted)

    row = GatewayRunner._prepend_inbound_reply_context(event, source, "什么情况")

    assert "TOOLCALL" not in row
    assert row.startswith('[Replying to: "Done!\n[processing...]"]')
    assert row.endswith("什么情况")


def test_reply_quoting_an_inline_toolcall_mention_keeps_its_trailing_prose():
    """Prose that merely mentions the tag is not a block: everything after it survives."""
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm", user_id="42")
    quoted = "The <TOOLCALL> tag appears in this log; please keep the next paragraph."
    event = MessageEvent(text="why?", source=source, reply_to_message_id="42", reply_to_text=quoted)

    row = GatewayRunner._prepend_inbound_reply_context(event, source, "why?")

    assert row.startswith('[Replying to: "The <TOOLCALL> tag appears in this log; please keep the next paragraph."]')
