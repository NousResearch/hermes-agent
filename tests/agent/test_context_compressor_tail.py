"""Lean stale-tail demotion: a demoted ``skill_view`` body keeps its reload marker (#32106)."""
import json

import pytest

import agent.context_compressor as cc
from agent.context_compressor_tail import _LEAN_TAIL_KEEP_TOOL_ROUNDS

SKILL = "synthetic-tail-skill"
BODY = "Skill instructions " + "s" * 2000


def _call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _round(call_id, name, args, content, tool_name=True):
    tool = {"role": "tool", "tool_call_id": call_id, "content": content}
    if tool_name:
        tool["tool_name"] = name
    return [{"role": "assistant", "tool_calls": [_call(call_id, name, args)]}, tool]


def _fresh_rounds(n=_LEAN_TAIL_KEEP_TOOL_ROUNDS):
    rows = []
    for i in range(n):
        rows += _round(f"fresh-{i}", "terminal", {"command": f"echo {i}"}, f"Step {i} done.")
    return rows


def _stub_for(rows, call_id):
    return next(r["content"] for r in rows if r.get("role") == "tool" and r.get("tool_call_id") == call_id)


@pytest.fixture
def compressor(monkeypatch):
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    monkeypatch.setattr(cc, "call_llm", lambda **kw: {"choices": [{"message": {
        "content": "## Goal\nSynthetic work is recorded."}, "finish_reason": "stop"}]})
    comp = cc.ContextCompressor(model="test/model", config_context_length=100_000, protect_first_n=0,
                                protect_last_n=3, quiet_mode=True, tail_mode="lean")
    comp.tail_token_budget = 2000
    return comp


def test_reload_marker_survives_the_next_compaction(compressor):
    # With a session id the marked stub passes the Phase-1 prune floor (200 chars); it must not be
    # re-summarised into a plain "[skill_view] ..." line that drops the marker.
    compressor.bind_session_state(session_id="0123abcd-4567-89ef-0123-456789abcdef")
    rows = [
        {"role": "system", "content": "Synthetic test conversation."},
        {"role": "user", "content": "Summarize the completed earlier work."},
        {"role": "assistant", "content": "Old notes " + "x" * 40_000},
        *_round("older-terminal", "terminal", {"command": "old-job"}, "Earlier output " + "o" * 8000),
        {"role": "user", "content": "Continue the current work."},
        *_round("skill-1", "skill_view", {"name": SKILL}, BODY),
        *_fresh_rounds(),
        {"role": "assistant", "content": "All rounds completed."},
    ]
    first = compressor.compress(rows + [{"role": "user", "content": "Report the current work."}], force=True)
    stub = _stub_for(first, "skill-1")
    assert len(stub) > cc._PRUNE_MIN_CHARS and cc._extract_pruned_skill_names(stub) == [SKILL]

    later = [{"role": "assistant", "content": "Reported."}, {"role": "user", "content": "Next task."}]
    for i in range(8):
        # Enough newer output that the stub falls outside Phase-1 tail protection (tail_token_budget).
        later += _round(f"later-{i}", "terminal", {"command": f"echo later {i}"}, f"Later step {i} " + "l" * 1400)
    second = compressor.compress(first + later + [{"role": "assistant", "content": "Done."}], force=True)

    assert compressor.compression_count == 2 and not compressor._last_summary_fallback_used
    text = "\n".join(str(r.get("content") or "") for r in second)
    assert SKILL in cc._extract_pruned_skill_names(text), "reload marker lost at the next compaction"


def _args_for(rows, call_id):
    for row in rows:
        if row.get("role") != "assistant":
            continue
        for call in row.get("tool_calls") or []:
            if call.get("id") == call_id:
                return call["function"]["arguments"]
    return None


def _big_args(extra="c" * 3000):
    return {"path": "~/playground/x.py", "content": extra}


def _stale_rows(call_id="stale-write", name="write_file", args=None, content="Wrote x.py."):
    """A conversation whose stale round sits inside the lean tail (shaped like the reload-marker test):
    without the older bulk the round is summarised away instead of demoted."""
    return [
        {"role": "system", "content": "Synthetic test conversation."},
        {"role": "user", "content": "Summarize the completed earlier work."},
        {"role": "assistant", "content": "Old notes " + "x" * 40_000},
        *_round(call_id, name, args if args is not None else _big_args(), content),
        {"role": "user", "content": "Continue the current work."},
        *_fresh_rounds(),
        {"role": "assistant", "content": "All rounds completed."},
    ]


def _compress(compressor, rows):
    compressor.bind_session_state(session_id="0123abcd-4567-89ef-0123-456789abcdef")
    return compressor.compress(rows + [{"role": "user", "content": "Now what?"}], force=True)


def test_stale_call_arguments_are_demoted_with_their_round(compressor):
    # The argument half of a stale round is the larger half (whole write_file bodies, patch sides, inline
    # scripts) and stays duplicated on disk once the write landed: it demotes with the round.
    out = _compress(compressor, _stale_rows())

    stale = json.loads(_args_for(out, "stale-write"))
    assert set(stale) == {"_demoted"} and "demoted at compaction" in stale["_demoted"]
    # …while the kept round (result still in the tail) keeps its arguments verbatim.
    assert json.loads(_args_for(out, "fresh-0")) == {"command": "echo 0"}


def test_demoted_arguments_keep_the_call_identity(compressor):
    # The id and name must survive: the API rejects a tool result whose call is gone, and the marker
    # protocol looks a skill_view call up by id.
    out = _compress(compressor, _stale_rows())

    call = next(c for r in out if r.get("role") == "assistant" for c in r.get("tool_calls") or []
                if c.get("id") == "stale-write")
    assert call["function"]["name"] == "write_file" and call["type"] == "function"


def test_demoted_arguments_name_the_archived_original(compressor, monkeypatch):
    # Where the folder is readable the stub names the file, so the exact text is one read_file away.
    from hermes_constants import get_hermes_home
    from tools import tool_result_storage as trs

    monkeypatch.setattr(trs, "_archive_backend", lambda: "local")
    body = _big_args()["content"]
    out = _compress(compressor, _stale_rows())

    stub = json.loads(_args_for(out, "stale-write"))["_demoted"]
    assert "cache/pruned/" in stub, "stub does not name the archive"
    archived = [p for p in (get_hermes_home() / "cache" / "pruned").iterdir() if "_write_file_args_" in p.name]
    assert len(archived) == 1 and body in archived[0].read_text(encoding="utf-8")


def test_demoted_arguments_stand_plain_where_no_archive_is_readable(compressor, monkeypatch):
    # A plugin or hosted backend (no readable folder) still reclaims: the plain stub stands, as the result half does.
    from tools import tool_result_storage as trs

    monkeypatch.setattr(trs, "_archive_backend", lambda: None)
    out = _compress(compressor, _stale_rows())

    stub = json.loads(_args_for(out, "stale-write"))["_demoted"]
    assert "demoted at compaction" in stub and "cache/pruned/" not in stub


def test_failed_archive_write_keeps_the_arguments_whole(compressor, monkeypatch):
    # The row goes back to its original when the write fails: never a stub without the file behind it.
    from tools import tool_result_storage as trs

    original = _big_args()
    monkeypatch.setattr(trs, "_archive_backend", lambda: "local")
    monkeypatch.setattr(trs, "_write_to_spillover", lambda *a, **k: None)
    out = _compress(compressor, _stale_rows(args=original))

    assert json.loads(_args_for(out, "stale-write")) == original


def test_skill_handles_and_short_arguments_are_left_alone(compressor):
    # A skill_view call carries the reload handle itself (#32106); short arguments cost nothing to keep.
    # Called directly: an argument-only row needs no result row, and this pins the demoter's own guards.
    from agent.context_compressor_tail import demote_stale_tail_tools

    big_skill = {"name": SKILL, "note": "n" * 3000}
    rows = _stale_rows("stale-skill", "skill_view", big_skill, BODY)
    out, demoted = demote_stale_tail_tools(rows, 0, "0123abcd-4567-89ef-0123-456789abcdef")

    assert json.loads(_args_for(out, "stale-skill")) == big_skill
    out2, demoted2 = demote_stale_tail_tools(out, 0, "0123abcd-4567-89ef-0123-456789abcdef")
    assert demoted2 == 0, "a second pass must not re-demote (or double the pointer)"
