"""End-to-end: lean compaction emits exact ``m:<12hex>`` refs that ``session_search(ref=...)`` resolves.

Drives the real ``AIAgent`` -> ``compress_context()`` -> ``archive_and_compact()`` path with only the
auxiliary summary call mocked, then follows every pointer the compaction wrote exactly as the model
would: parse the emitted ``session_search(...)`` call out of the text and run it.

A demoted tail tool result is the case keyword recovery cannot reach at all: its full original is a
superseded carried-tail duplicate (``active=0, compacted=0``), outside every search and scroll shape.
"""

import ast
import json
import os
import re
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    _LEAN_RECOVERY_HEADING,
    _LEAN_SESSION_LOG_HEADING,
    _LEAN_USER_MESSAGES_HEADING,
)
from agent.message_metadata import MESSAGE_UID
from hermes_state import SessionDB

TAIL_NEEDLE = "TAILMARK_3141"
MID_NEEDLE = "MIDMARK_2718"
SESSION_ID = "s_exact_ref_e2e"
_REF_RE = re.compile(r"m:[0-9a-f]{12}\b")


def _llm_response(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def _emitted_calls(text: str) -> list:
    """Every ``session_search(...)`` call literally emitted in *text*, parsed to kwargs."""
    calls = []
    for match in re.finditer(r"session_search\(", text):
        start, depth = match.start(), 0
        for index, char in enumerate(text[start:], start):
            depth += char == "("
            depth -= char == ")"
            if char == ")" and depth == 0:
                node = ast.parse(text[start:index + 1], mode="eval").body
                calls.append({kw.arg: kw.value.value for kw in node.keywords})
                break
    return calls


def _search(db, **kwargs):
    """Run a call exactly as a model's tool call runs: through the inline executor, not the function."""
    from types import SimpleNamespace
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    agent = SimpleNamespace(_get_session_db_for_recall=lambda: db, session_id=SESSION_ID)
    return json.loads(INLINE_TOOL_EXECUTORS["session_search"](
        agent, kwargs, InlineToolContext(effective_task_id="task", tool_call_id="call")))


def _anchor(result):
    assert result.get("success") is True, result
    return next(m for m in result["messages"] if m.get("anchor"))


def _section(summary: str, heading: str) -> str:
    """One ``## `` section of a summary, without the handoff's closing END marker."""
    start = summary.index(heading)
    ends = [i for i in (summary.find("\n## ", start + len(heading)), summary.find("\n\n--- END", start)) if i != -1]
    return summary[start:min(ends)] if ends else summary[start:]


def _summary_text(compressed):
    return next(m["content"] for m in compressed
                if isinstance(m.get("content"), str) and _LEAN_RECOVERY_HEADING in m["content"])


def _seed(db, rounds=12, prefix="recent", start=0):
    """Tool rounds; round 5 of the first batch carries the tail needle."""
    ids = {}
    for index in range(start, start + rounds):
        call_id = f"call_{prefix}_{index:02d}"
        db.append_message(session_id=SESSION_ID, role="user", content=f"Run {prefix} check {index}")
        db.append_message(session_id=SESSION_ID, role="assistant", content="", tool_calls=[{
            "id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
        needle = TAIL_NEEDLE if (prefix == "recent" and index == 5) else f"FILLER_{prefix}_{index:02d}"
        ids[index] = (call_id, db.append_message(
            session_id=SESSION_ID, role="tool", content=(f"{prefix} build {index} output\n" * 170) + f"fatal: {needle}",
            tool_name="terminal", tool_call_id=call_id))
        db.append_message(session_id=SESSION_ID, role="assistant", content=f"{prefix} check {index} completed")
    return ids


def _build_session(db):
    db.create_session(SESSION_ID, source="cli", model="gpt-5")
    for index in range(12):
        db.append_message(session_id=SESSION_ID, role="user" if index % 2 == 0 else "assistant",
                          content=f"historical filler {index}")
    db.append_message(session_id=SESSION_ID, role="user", content="Run the historical positive-control check")
    db.append_message(session_id=SESSION_ID, role="assistant", content="", tool_calls=[{
        "id": "call_mid", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
    db.append_message(session_id=SESSION_ID, role="tool", content=("historical build output\n" * 180)
                      + f"fatal: {MID_NEEDLE}", tool_name="terminal", tool_call_id="call_mid")
    db.append_message(session_id=SESSION_ID, role="assistant", content="The historical check completed")
    # Twelve recent tool rounds: round 5 is inside the lean protected tail but older than the six rounds kept
    # verbatim, so its output is demoted to a stub.
    tail = _seed(db)
    db.append_message(session_id=SESSION_ID, role="user", content="What was the exact fatal line from check 5?")
    return tail


def _agent(db):
    from run_agent import AIAgent
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="gpt-5",
                        quiet_mode=True, session_db=db, session_id=SESSION_ID,
                        skip_context_files=True, skip_memory=True)
    agent.compression_in_place = True
    agent.context_compressor.tail_mode = "lean"
    return agent


def _compress(agent, messages):
    from agent.conversation_compression import compress_context
    summary = ("## Historical Task Snapshot\nRecover exact tool output after compaction.\n\n"
               f"{_LEAN_SESSION_LOG_HEADING}\n- Historical check recorded {MID_NEEDLE}.\n")
    with patch("agent.context_compressor.call_llm", return_value=_llm_response(summary)) as aux:
        compressed, _ = compress_context(agent, messages, approx_tokens=100_000, system_message="system prompt")
    assert aux.call_count == 1
    return compressed


def _rows(db):
    return [dict(r) for r in db._conn.execute(
        "SELECT id, role, content, message_uid, active, compacted FROM messages WHERE session_id = ? ORDER BY id",
        (SESSION_ID,)).fetchall()]


@pytest.fixture()
def db(tmp_path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def test_every_emitted_ref_resolves_exactly(db):
    tail = _build_session(db)
    messages = db.get_messages_as_conversation(SESSION_ID, include_row_ids=True)
    agent = _agent(db)
    compressed = _compress(agent, messages)
    live_uids = {m.get(MESSAGE_UID) for m in compressed}
    rows = _rows(db)

    # a. The demoted stub's emitted call is a ref call that returns the full demoted output.
    call_id, tail_row_id = tail[5]
    stub = next(m for m in compressed if m.get("tool_call_id") == call_id)["content"]
    assert "output demoted at compaction" in stub and "chars preserved" in stub, stub
    assert TAIL_NEEDLE not in json.dumps(compressed)
    original = next(r for r in rows if r["id"] == tail_row_id)
    assert (original["active"], original["compacted"]) == (0, 0)
    [stub_call] = _emitted_calls(stub)
    assert set(stub_call) == {"ref"} and _REF_RE.fullmatch(stub_call["ref"]), stub_call
    recovered = _anchor(_search(db, **stub_call))
    assert recovered["id"] == tail_row_id and recovered["content"] == original["content"]
    assert f"fatal: {TAIL_NEEDLE}" in recovered["content"]

    summary = _summary_text(compressed)

    # c. The footer's region-start ref opens the first compacted message; the end ref names the last.
    footer = _section(summary, _LEAN_RECOVERY_HEADING)
    first_ref, last_ref = _REF_RE.findall(footer)[:2]
    ref_calls = [c for c in _emitted_calls(footer) if "ref" in c]
    assert ref_calls and ref_calls[0]["ref"] == first_ref
    opened = _search(db, **ref_calls[0])
    first_row = next(r for r in rows if r["id"] == _anchor(opened)["id"])
    last_row = next(r for r in rows if r["id"] == db.resolve_message_ref(last_ref)["id"])
    ordered = rows
    first_pos, last_pos = ordered.index(first_row), ordered.index(last_row)
    assert first_pos < last_pos
    # Region boundaries: the region's rows left the live context; its neighbours are still live.
    assert first_row["message_uid"] not in live_uids and last_row["message_uid"] not in live_uids
    assert first_pos == 0 or ordered[first_pos - 1]["message_uid"] in live_uids
    assert ordered[last_pos + 1]["message_uid"] in live_uids
    # The keyword-recovery sentence is still there.
    assert "session_search(query='<keywords>'" in footer

    # b. Quoted user messages carry refs that resolve to their originals.
    quotes = _section(summary, _LEAN_USER_MESSAGES_HEADING)
    quoted = re.findall(r"^> \[(m:[0-9a-f]{12})\] (.*)$", quotes, flags=re.M)
    assert quoted, quotes
    for ref, text in quoted:
        anchor = _anchor(_search(db, ref=ref))
        assert anchor["role"] == "user" and anchor["content"].startswith(text.rstrip("…")), (ref, text, anchor)


def _two_compactions(db):
    _build_session(db)
    agent = _agent(db)
    first = _compress(agent, db.get_messages_as_conversation(SESSION_ID, include_row_ids=True))
    first_range = _REF_RE.findall(_section(_summary_text(first), _LEAN_RECOVERY_HEADING))[:2]
    _seed(db, rounds=12, prefix="later", start=0)
    db.append_message(session_id=SESSION_ID, role="user", content="And the later checks?")
    second = _compress(agent, db.get_messages_as_conversation(SESSION_ID, include_row_ids=True))
    second_footer = _section(_summary_text(second), _LEAN_RECOVERY_HEADING)
    second_range = _REF_RE.findall(second_footer)[:2]
    summary_one = next(r for r in _rows(db) if isinstance(r["content"], str)
                       and first_range[0] in r["content"] and _LEAN_RECOVERY_HEADING in r["content"])
    return first_range, second_range, summary_one, second_footer


def test_a_second_compaction_footer_range_spans_the_first_summary_row(db):
    first_range, (start_ref, end_ref), summary_one, _footer = _two_compactions(db)
    start, end = db.resolve_message_ref(start_ref), db.resolve_message_ref(end_ref)
    assert start["session_id"] == end["session_id"] == SESSION_ID
    assert start["id"] < summary_one["id"] < end["id"]
    # The first footer's own range still resolves: level two of the walk starts from it.
    level_two = _anchor(_search(db, ref=first_range[0]))
    assert level_two["id"] < summary_one["id"]


def test_the_second_footer_start_reaches_a_first_generation_summarized_original(db):
    # The merged handoff carrier keeps its tail message's uid (#126307), so the first summary has no ref of
    # its own. What the model CAN do: open the second footer's region start and reach the first generation's
    # summarized originals (compacted=1) from there, in the window or by scrolling from one of its rows.
    _first_range, _second_range, summary_one, footer = _two_compactions(db)
    rows = {r["id"]: r for r in _rows(db)}
    generation_one = {i for i, r in rows.items() if i < summary_one["id"] and (r["active"], r["compacted"]) == (0, 1)}
    [opened_call] = [c for c in _emitted_calls(footer) if "ref" in c]
    opened = _search(db, **opened_call)
    window_ids = [m["id"] for m in opened["messages"]]
    reached = generation_one.intersection(window_ids)
    if not reached:
        compacted_in_window = [i for i in window_ids if rows[i]["compacted"] == 1]
        assert compacted_in_window, window_ids
        scrolled = _search(db, session_id=SESSION_ID, around_message_id=compacted_in_window[0], window=20)
        reached = generation_one.intersection(m["id"] for m in scrolled["messages"])
    assert reached, (window_ids, sorted(generation_one)[:5])


def test_a_transcript_without_uids_emits_todays_exact_text(db):
    tail = _build_session(db)
    messages = db.get_messages_as_conversation(SESSION_ID, include_row_ids=True)
    for message in messages:
        message.pop(MESSAGE_UID, None)
    compressed = _compress(_agent(db), messages)

    call_id, tail_row_id = tail[5]
    original = next(r for r in _rows(db) if r["id"] == tail_row_id)
    stub = next(m for m in compressed if m.get("tool_call_id") == call_id)["content"]
    assert stub == (
        f"[terminal output demoted at compaction — {len(original['content']):,} chars preserved in session "
        f"history. Recover with session_search(query=..., session_id='{SESSION_ID}')]")

    summary = _summary_text(compressed)
    footer = _section(summary, _LEAN_RECOVERY_HEADING)
    region_len = int(re.search(r"The (\d+) compacted message\(s\)", footer).group(1))
    assert footer == (
        f"{_LEAN_RECOVERY_HEADING}\nThe {region_len} compacted message(s) remain fully preserved in "
        "session history. If you need any detail this summary does not carry (exact command output, file "
        "contents, error text, earlier reasoning), recover it with: "
        f"session_search(query='<keywords>', session_id='{SESSION_ID}') — do not guess at lost specifics "
        "when you can look them up.")
    quotes = _section(summary, _LEAN_USER_MESSAGES_HEADING)
    assert "[m:" not in quotes and re.search(r"^> Run recent check", quotes, flags=re.M)


# ── Deterministic sections, unit level ───────────────────────────────────────


def test_anchor_index_does_not_harvest_our_own_refs_as_commits():
    from agent.context_compressor import _build_anchor_index
    turns = [
        {"role": "tool", "content": "[terminal output demoted at compaction — 9,000 chars preserved in session "
                                    "history. Recover exactly with session_search(ref='m:3f2a9c1e0b7d')]"},
        {"role": "user", "content": "> [m:0123456789ab] earlier words, and a real commit deadbeef1234"},
    ]
    index = _build_anchor_index(turns)
    assert "deadbeef1234" in index
    assert "3f2a9c1e0b7d" not in index and "0123456789ab" not in index


def test_redaction_leaves_refs_intact():
    from agent.context_compressor import _redact_compaction_text
    text = ("> [m:3f2a9c1e0b7d] quoted words\nThe region spans m:0123456789ab … m:abcdefabcdef (40 messages); "
            "session_search(ref='m:0123456789ab', window=10)")
    assert _redact_compaction_text(text) == text


def test_helpers_fall_back_when_the_uid_is_absent_or_not_hex():
    from agent.context_compressor import (
        _build_recovery_footer, _build_verbatim_user_section, _lean_recovery_stub)
    uid = "3f2a9c1e0b7d" + "0" * 20
    assert _lean_recovery_stub("terminal", 2000, "sid", uid).endswith(
        "chars preserved in session history. Recover exactly with session_search(ref='m:3f2a9c1e0b7d')]")
    for bad in (None, "", "not-a-hex-uid", "E" * 32):
        assert _lean_recovery_stub("terminal", 2000, "sid", bad) == _lean_recovery_stub("terminal", 2000, "sid")
    turns = [{"role": "user", "content": "hello", MESSAGE_UID: uid}, {"role": "user", "content": "bye"}]
    assert "> [m:3f2a9c1e0b7d] hello" in _build_verbatim_user_section(turns)
    assert "> bye" in _build_verbatim_user_section(turns)
    footer = _build_recovery_footer("sid", turns)
    assert "m:3f2a9c1e0b7d" not in footer  # the last message carries no uid: no range sentence
    assert _build_recovery_footer("sid", turns[:1] * 2).count("session_search(ref='m:3f2a9c1e0b7d', window=10)") == 1
