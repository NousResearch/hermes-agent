"""``/copy all`` / ``/save md --concise``: the visible dialogue of the WHOLE conversation.

Real ``SessionDB`` + temporary HERMES_HOME. What is pinned here is the contract, not the helpers:
every logical message once (a genuinely repeated one twice), compaction-archived turns and the
verified compression lineage included, undone/branched rows excluded, and nothing but visible
user/assistant text ever reaches the output.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from hermes_cli.session_export import (
    load_concise_transcript, parse_save_args, render_concise_markdown)
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _dialogue(db, sid, home):
    data = load_concise_transcript(db, sid, home)
    return [(m["role"], m["content"]) for m in data["messages"]]


def _compact(db, sid, tag, tail=2):
    live = db.get_messages_as_conversation(sid)
    db.archive_and_compact(
        sid, [{"role": "user", "content": f"{SUMMARY_PREFIX} summary {tag}"}] + live[-tail:], tail_count=tail)


def test_full_history_survives_compaction_and_carried_tail_appears_once(db, tmp_path):
    sid = "chat"
    db.create_session(sid, source="desktop")
    for epoch in range(3):
        for i in range(3):
            db.append_message(sid, "user", f"e{epoch} user {i}")
            db.append_message(sid, "assistant", f"e{epoch} assistant {i}")
        _compact(db, sid, epoch)

    dialogue = _dialogue(db, sid, tmp_path)

    assert dialogue == [
        (role, f"e{e} {role} {i}") for e in range(3) for i in range(3) for role in ("user", "assistant")]
    assert not any("summary" in text for _, text in dialogue)


def test_a_message_the_user_really_repeated_stays_twice(db, tmp_path):
    db.create_session("chat", source="desktop")
    for _ in range(2):
        db.append_message("chat", "user", "same words")
        db.append_message("chat", "assistant", "ok")
    _compact(db, "chat", 0)

    assert _dialogue(db, "chat", tmp_path).count(("user", "same words")) == 2


def test_rotation_lineage_is_joined_without_duplicating_the_moved_tail(db, tmp_path):
    """A real compression rotation (``publish_compression_child``): the protected tail is copied into the child."""
    parent, child = "parent", "child"
    db.create_session(parent, source="desktop")
    for i in range(3):
        db.append_message(parent, "user", f"P user {i}")
        db.append_message(parent, "assistant", f"P assistant {i}")
    tail = db.get_messages_as_conversation(parent)[-2:]
    assert db.try_acquire_compression_lock(parent, "t")
    try:
        db.publish_compression_child(
            parent_session_id=parent, child_session_id=child, source="desktop",
            messages=[{"role": "user", "content": f"{SUMMARY_PREFIX} handoff"}, *tail], compression_lock_holder="t")
    finally:
        db.release_compression_lock(parent, "t")
    db.append_message(child, "user", "C user 0")
    db.append_message(child, "assistant", "C assistant 0")

    texts = [t for _, t in _dialogue(db, child, tmp_path)]

    assert texts == [*(f"P {r} {i}" for i in range(3) for r in ("user", "assistant")), "C user 0", "C assistant 0"]
    assert texts.count("P user 2") == 1 and not any("handoff" in t for t in texts)


def test_rewound_rows_and_explicit_branches_and_other_sessions_are_excluded(db, tmp_path):
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "kept")
    db.append_message("chat", "assistant", "kept reply")
    db.append_message("chat", "user", "taken back")
    db.append_message("chat", "assistant", "taken back reply")
    target = next(m for m in reversed(db.get_messages("chat")) if m["role"] == "user")
    db.rewind_to_message("chat", target["id"])
    db.create_session("sibling", source="desktop", parent_session_id="chat")
    db.append_message("sibling", "user", "sibling only")
    db.create_session("other", source="desktop")
    db.append_message("other", "user", "unrelated chat")

    texts = [t for _, t in _dialogue(db, "chat", tmp_path)]

    assert texts == ["kept", "kept reply"]
    assert "sibling only" not in str(_dialogue(db, "chat", tmp_path))
    assert [t for _, t in _dialogue(db, "sibling", tmp_path)] == ["sibling only"]  # a branch stands alone


def test_only_visible_dialogue_leaves_tool_reasoning_and_hidden_records_stay_in(db, tmp_path):
    db.create_session("chat", source="desktop")
    db.append_message("chat", "system", "SYSTEM-PROMPT-SECRET")
    db.append_message("chat", "user", "hello Türkçe 🌍 [link](https://example.com)\n```py\nprint('x')\n```")
    db.append_message(
        "chat", "assistant", "calling", reasoning="REASONING-SECRET", reasoning_content="RC-SECRET",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "TOOLARG-SECRET"}'}}])
    db.append_message("chat", "tool", "TOOLRESULT-SECRET", tool_name="terminal", tool_call_id="c1")
    db.append_message("chat", "user", "scaffolding the user never typed", display_kind="hidden")
    db.append_message("chat", "user", "[System: model switched]")
    db.append_message("chat", "assistant", "final answer")

    data = load_concise_transcript(db, "chat", tmp_path)
    rendered = render_concise_markdown(data)

    for secret in ("SYSTEM-PROMPT", "REASONING-SECRET", "RC-SECRET", "TOOLARG", "TOOLRESULT", "scaffolding", "model switched"):
        assert secret not in rendered
    assert "hello Türkçe 🌍 [link](https://example.com)\n```py\nprint('x')\n```" in rendered
    # the assistant turn that carried a tool call keeps its visible words; the call and its result do not
    assert _dialogue(db, "chat", tmp_path) == [
        ("user", "hello Türkçe 🌍 [link](https://example.com)\n```py\nprint('x')\n```"),
        ("assistant", "calling"), ("assistant", "final answer")]


@pytest.mark.parametrize("kind", [
    "model_switch", "personality_switch", "internal_notification", "failed_turn", "auto_continue",
    "async_delegation_complete"])
def test_timeline_notices_are_not_dialogue(db, tmp_path, kind):
    """Rows the chat paints as timeline notices (not as something the user or assistant said) stay out."""
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "real question")
    db.append_message("chat", "assistant", f"NOTICE-{kind}", display_kind=kind)
    db.append_message("chat", "assistant", "real answer")

    assert _dialogue(db, "chat", tmp_path) == [("user", "real question"), ("assistant", "real answer")]


def test_a_steer_shows_the_users_own_words_and_a_skill_turn_its_invocation_not_the_expanded_body(db, tmp_path):
    from agent.prompt_builder import STEER_DISPLAY_KIND, format_steer_marker

    skill_body = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they want you to follow its instructions. '
        "The full skill content is loaded below.]\n\nSKILL-BODY-SECRET\n\n"
        "The user has provided the following instruction alongside the skill invocation: fix the leak")
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", skill_body)
    db.append_message("chat", "assistant", "on it")
    db.append_message("chat", "user", format_steer_marker("also check the tests").lstrip(), display_kind=STEER_DISPLAY_KIND)
    db.append_message("chat", "assistant", "done")

    dialogue = _dialogue(db, "chat", tmp_path)

    assert dialogue == [("user", "/work fix the leak"), ("assistant", "on it"),
                        ("user", "also check the tests"), ("assistant", "done")]
    assert "SKILL-BODY-SECRET" not in str(dialogue) and "OUT-OF-BAND" not in str(dialogue)


def test_attachments_never_leak_base64_or_data_urls(db, tmp_path):
    blob = "A" * 400
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{blob}"}}])
    db.append_message("chat", "assistant", f"see data:image/png;base64,{blob} done")

    rendered = render_concise_markdown(load_concise_transcript(db, "chat", tmp_path))

    assert blob not in rendered and "base64" not in rendered
    assert "what is this?" in rendered and "[image]" in rendered


def test_a_pure_handoff_is_dropped_but_real_text_inside_a_mixed_carrier_is_kept(db, tmp_path):
    from agent.context_compressor import _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER

    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "first ask")
    db.append_message("chat", "assistant", "first answer")
    db.append_message("chat", "user", f"{SUMMARY_PREFIX} a standalone handoff")
    mixed = (f"{_MERGED_PRIOR_CONTEXT_HEADER}\nREAL PRIOR ASK\n\n{_MERGED_SUMMARY_DELIMITER}\n\n"
             f"{SUMMARY_PREFIX} SYNTHETIC-SUMMARY-BODY")
    db.append_message("chat", "user", mixed)
    db.append_message("chat", "assistant", "reply")

    texts = [t for _, t in _dialogue(db, "chat", tmp_path)]

    assert texts == ["first ask", "first answer", "REAL PRIOR ASK", "reply"]


def test_reading_changes_nothing(db, tmp_path):
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "q")
    db.append_message("chat", "assistant", "a")
    before = (db.get_messages("chat", include_inactive=True), db.get_session("chat"))

    load_concise_transcript(db, "chat", tmp_path)
    load_concise_transcript(db, "chat", tmp_path)

    assert (db.get_messages("chat", include_inactive=True), db.get_session("chat")) == before


def test_unknown_session_and_empty_dialogue(db, tmp_path):
    assert load_concise_transcript(db, "nope", tmp_path) is None
    db.create_session("empty", source="desktop")
    assert load_concise_transcript(db, "empty", tmp_path)["messages"] == []


def test_markdown_shape_is_plain_headings_not_one_big_fence():
    md = render_concise_markdown({"id": "s", "title": "Example task", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "```py\nx=1\n```"}]})

    assert md == "# Conversation: Example task\n\n## User\n\nHi\n\n## Assistant\n\n```py\nx=1\n```\n"
    assert md.count("```") == 2


@pytest.mark.parametrize("words, expected", [
    (["md", "--concise"], (["md"], False, True)),
    (["md", "conversation.md", "--concise", "--redact"], (["md", "conversation.md"], True, True)),
    (["--concise", "md", "notes.md", "redact"], (["md", "notes.md"], True, True)),
    (["md", "notes.md"], (["md", "notes.md"], False, False)),  # the legacy call is untouched
    (["md", "redact"], (["md"], True, False)),
    (["md", "redact.md"], (["md", "redact.md"], False, False)),  # a file may be called redact.md
])
def test_save_argument_grammar_never_mistakes_a_flag_for_a_filename(words, expected):
    assert parse_save_args(words) == expected
