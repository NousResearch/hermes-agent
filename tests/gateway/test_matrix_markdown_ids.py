"""Outbound Markdown → Matrix HTML: ``#123`` ids are not headings, a lone heading keeps its
paragraph break, and a list may follow a label line directly (CommonMark behaviour)."""

from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter, _list_interrupts_paragraph, _standalone_payloads


def _html(text: str) -> str:
    return MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
    ))._markdown_to_html(text)


def test_hash_id_at_line_start_is_not_a_heading():
    html = _html("I archived #43.\n\n#310–312 are dictation tests.")
    assert "<h1>" not in html
    assert "#310–312 are dictation tests." in html
    assert "<p>I archived #43.</p>" in html  # paragraphs keep their break


def test_single_line_starting_with_id_stays_plain():
    assert _html("#253 moved to tomorrow.") == "#253 moved to tomorrow."


def test_real_headings_still_render_and_keep_the_paragraph_break():
    html = _html("# Status\n\nAll good.")
    assert "<h1>Status</h1>" in html
    assert "<p>All good.</p>" in html


def test_single_paragraph_is_still_unwrapped():
    assert _html("**bold** text") == "<strong>bold</strong> text"


def test_list_directly_under_a_label_line_becomes_a_list():
    html = _html("**Today**\n- #252 Review BIM-200\n- #253 Cards (6pm)\n\nReply \"done 252\".")
    assert "<ul>" in html
    assert "<li>#252 Review BIM-200</li>" in html
    assert "<li>#253 Cards (6pm)</li>" in html
    assert "- #252" not in html


def test_list_interrupt_leaves_code_fences_and_existing_lists_alone():
    text = "Run:\n```\nlabel\n- not a list\n```\n- a\n- b\n\n1. one\n2. two"
    out = _list_interrupts_paragraph(text)
    assert "label\n- not a list" in out  # inside the fence: untouched
    assert "```\n- a\n- b" in out  # fence close is not paragraph text
    assert "- a\n- b" in out  # no blank lines inside a list (would make it loose)
    assert _list_interrupts_paragraph("Steps:\n1. one\n2. two") == "Steps:\n\n1. one\n2. two"
    assert _list_interrupts_paragraph("Total 2\n2. not a list start") == "Total 2\n2. not a list start"


def test_standalone_payloads_share_the_rules():
    (payload,) = _standalone_payloads("**Today**\n- #252 Review\n\n#310 is a test.")
    html = payload["formatted_body"]
    assert "<li>#252 Review</li>" in html
    assert "<strong>310" not in html and "#310 is a test." in html
