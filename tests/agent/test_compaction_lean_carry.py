"""Lean sections are rebuilt each compaction: quotes and anchors carry, pruned tool bodies feed the index."""

import json

import pytest

import agent.context_compressor as cc

SUMMARY = "## Goal\nSynthetic work is recorded."


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    state = {"requests": [], "summary": SUMMARY, "fail": False}

    def summarize(**kwargs):
        state["requests"].append(kwargs)
        if state["fail"]:
            raise RuntimeError("Synthetic summarizer unavailable.")
        return {"choices": [{"message": {"content": state["summary"]}, "finish_reason": "stop"}]}

    monkeypatch.setattr(cc, "call_llm", summarize)
    state["new"] = _compressor
    state["compressor"] = _compressor()
    return state


def _compressor(tail_tokens=50):
    comp = cc.ContextCompressor(
        model="test/model", config_context_length=100_000, protect_first_n=0,
        protect_last_n=3, quiet_mode=True, tail_mode="lean",
    )
    comp.tail_token_budget = tail_tokens
    return comp


def _progress(tag):
    return [row for i in range(8) for row in (
        {"role": "user", "content": f"Record {tag} progress {i}."},
        {"role": "assistant", "content": f"Progress {i} recorded."},
    )]


def _section(text, heading):
    start = text.index(heading)
    end = text.find("\n\n## ", start + len(heading))
    return text[start:end if end >= 0 else len(text)]


def _summary_row(rows):
    return next(row["content"] for row in rows if cc._LEAN_USER_MESSAGES_HEADING in (row.get("content") or ""))


def _tool_round(call_id, body):
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "terminal", "arguments": json.dumps({"command": "make build"})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def _long_output(marker):
    lines = "\n".join(f"line {i}: ordinary build output" for i in range(200))
    return lines[:3000] + f"\n{marker}\n" + lines[3000:]


def _g4a_rows():
    body = _long_output("artifact deadbeefcafe1234 published for #987654")
    rows = [{"role": "system", "content": "Synthetic test conversation."}, {"role": "user", "content": "Run the build."}]
    rows += [*_tool_round("call_g4a", body), *_progress("middle"), {"role": "user", "content": "Run the steps."}]
    for i in range(8):  # more rounds than the lean tail keeps, so the oldest come back as stubs
        rows += _tool_round(f"call_tail_{i}", _long_output(f"step {i} done"))
    return rows + [{"role": "assistant", "content": "Steps done."}, {"role": "user", "content": "Thanks."}]


def test_pruned_tool_only_identifier_reaches_anchor_index(harness):
    out = harness["compressor"].compress(_g4a_rows(), force=True)
    prompt = json.dumps(harness["requests"][-1]["messages"])
    assert "deadbeefcafe1234" not in prompt, "The tool body must still be pruned in the summary prompt."
    anchors = _section(_summary_row(out), cc._LEAN_ANCHOR_HEADING)
    assert "deadbeefcafe1234" in anchors and "#987654" in anchors


def _without_anchor_section(text):
    if cc._LEAN_ANCHOR_HEADING not in text:
        return text
    return text.replace("\n\n" + _section(text, cc._LEAN_ANCHOR_HEADING), "")


def test_pre_prune_bodies_feed_only_the_anchor_index(harness, monkeypatch):
    rows = _g4a_rows()
    with_bodies = harness["new"](2_000).compress(rows, force=True)
    prompt = harness["requests"][-1]["messages"]
    monkeypatch.setattr("agent.compaction_lean_carry.with_pre_prune_tool_bodies", lambda turns, source: turns)
    without = harness["new"](2_000).compress(rows, force=True)
    assert harness["requests"][-1]["messages"] == prompt, "The summary prompt must be byte-identical."
    assert len(with_bodies) == len(without)
    tools = [(a, b) for a, b in zip(with_bodies, without) if a.get("role") == "tool"]
    assert tools and all(a == b for a, b in tools), "Returned tool rows (stubs included) must be identical."
    originals = {row["tool_call_id"]: row["content"] for row in rows if row.get("role") == "tool"}
    assert any(a["content"] != originals[a["tool_call_id"]] for a, _ in tools), "The tail must hold a stub."
    summary, plain = _summary_row(with_bodies), _summary_row(without)
    assert _without_anchor_section(summary) == _without_anchor_section(plain)
    assert "deadbeefcafe1234" in summary and "deadbeefcafe1234" not in plain


def _two_compactions(harness, first_region, second_region, echo=False, fail=False):
    comp = harness["compressor"]
    head = [{"role": "system", "content": "Synthetic test conversation."}]
    first = comp.compress(head + first_region + _progress("first"), force=True)
    harness["summary"] = comp._previous_summary if echo else SUMMARY
    harness["fail"] = fail
    return _summary_row(comp.compress(first + second_region + _progress("second"), force=True))


def test_previous_anchors_carry_after_new_ones(harness):
    old = {"role": "user", "content": "Review #111222 at commit abcdef1234567."}
    new = {"role": "user", "content": "Then review #333444."}
    summary = _two_compactions(harness, [old], [new])
    anchors = _section(summary, cc._LEAN_ANCHOR_HEADING)
    assert summary.count(cc._LEAN_ANCHOR_HEADING) == 1
    assert "abcdef1234567" in anchors
    assert anchors.index("#333444") < anchors.index("#111222"), "New anchors come before carried ones."


def test_model_copied_sections_are_replaced_once(harness):
    old = {"role": "user", "content": "Please keep the first ask about #111222."}
    new = {"role": "user", "content": "Please keep the second ask about #333444."}
    summary = _two_compactions(harness, [old], [new], echo=True)
    for heading in (cc._LEAN_ANCHOR_HEADING, cc._LEAN_USER_MESSAGES_HEADING):
        assert summary.count(heading) == 1, heading
    quotes = _section(summary, cc._LEAN_USER_MESSAGES_HEADING)
    assert quotes.count("> " + old["content"]) == 1
    assert quotes.index("> " + new["content"]) < quotes.index("> " + old["content"])
    assert "#333444" in _section(summary, cc._LEAN_ANCHOR_HEADING)


def test_carried_quotes_respect_the_budget(harness):
    big = cc._LEAN_USER_MESSAGE_MAX_CHARS - 100
    old = [{"role": "user", "content": f"OLD-{i} " + "o" * big} for i in range(5)]
    new = [{"role": "user", "content": f"NEW-{i} " + "n" * big} for i in range(3)]
    summary = _two_compactions(harness, old, new)
    body = _section(summary, cc._LEAN_USER_MESSAGES_HEADING)
    quoted = sum(len(line) - 2 for line in body.splitlines() if line.startswith("> "))
    assert quoted <= cc._LEAN_USER_MESSAGES_BUDGET_CHARS
    order = [body.find(f"> {tag}") for tag in ("NEW-2", "NEW-1", "NEW-0", "OLD-4", "OLD-3", "OLD-2")]
    assert -1 not in order and order == sorted(order), "New quotes first, then carried quotes newest first."
    assert "> OLD-0" not in body, "The budget drops the oldest carried quote."


def test_fallback_summary_rebuilds_sections_with_carried_quotes(harness):
    old = {"role": "user", "content": "Please keep the first ask."}
    new = {"role": "user", "content": "Please keep the second ask."}
    summary = _two_compactions(harness, [old], [new], fail=True)
    assert harness["compressor"]._last_summary_fallback_used
    assert summary.count(cc._LEAN_USER_MESSAGES_HEADING) == 1, "The previous snapshot's copy is replaced."
    quotes = _section(summary, cc._LEAN_USER_MESSAGES_HEADING)
    assert quotes.index("> " + new["content"]) < quotes.index("> " + old["content"])


def _lines(text, line):
    return text.split("\n").count(line)


def test_a_longer_heading_that_shares_a_lean_prefix_survives(harness):
    plan = "## Context Recovery Plan\n- restore the db from the nightly copy"
    harness["summary"] = f"## Goal\nx\n\n{plan}\n\n## Next\nok"
    comp = harness["compressor"]
    comp._session_id = "synthetic-session"
    rows = [{"role": "system", "content": "Synthetic test conversation."}, *_progress("only")]
    summary = next(row["content"] for row in comp.compress(rows, force=True) if "## Goal" in (row.get("content") or ""))
    assert plan in summary and "## Next\nok" in summary, "Only the exact heading line marks a lean section."
    assert _lines(summary, cc._LEAN_RECOVERY_HEADING) == 1, "The fresh recovery footer is still appended."


def test_pre_prune_bodies_skip_ambiguous_tool_call_ids(harness):
    rows = _g4a_rows()[:-2]  # the unique pruned body (deadbeefcafe1234) must still reach the index
    rows[2:2] = _tool_round("call_dup", "OLD body for #111111")
    rows += [*_tool_round("call_dup", "TAIL body for #999999"), {"role": "assistant", "content": "Done."},
             {"role": "user", "content": "Thanks."}]
    out = harness["compressor"].compress(rows, force=True)
    tail = [row["content"] for row in out if row.get("tool_call_id") == "call_dup"]
    assert tail == ["TAIL body for #999999"], "The reused id's second row stays in the tail."
    anchors = _section(_summary_row(out), cc._LEAN_ANCHOR_HEADING)
    assert "deadbeefcafe1234" in anchors
    assert "#111111" in anchors and "#999999" not in anchors, "A reused tool_call_id keeps the row's own body."


def test_only_hermes_format_previous_quotes_are_carried(harness):
    first = harness["compressor"].compress(
        [{"role": "system", "content": "Synthetic test conversation."}, *_progress("first")], force=True)
    summary = _summary_row(first)
    malformed = (cc._LEAN_USER_MESSAGES_HEADING + "\n>no space quote\n\n> Mixed entry\nunprefixed line\n\n"
                 "> Valid carried ask.\n> Its second line.")
    start = summary.index(cc._LEAN_USER_MESSAGES_HEADING)
    end = summary.find("\n\n## ", start)
    resumed = [dict(row, content=summary[:start] + malformed + summary[end:])
               if row.get("content") == summary else row for row in first]
    out = harness["new"]().compress(resumed + _progress("second"), force=True)  # resume: no in-memory summary
    quotes = _section(_summary_row(out), cc._LEAN_USER_MESSAGES_HEADING)
    assert "> Valid carried ask.\n> Its second line." in quotes
    assert "space quote" not in quotes and "Mixed entry" not in quotes, "Non-format entries are skipped whole."


def test_a_long_repeated_quote_is_emitted_once_across_four_compactions(harness):
    comp = harness["compressor"]
    long_ask = {"role": "user", "content": "HEAD " + "a" * 5000 + " TAIL"}
    rows = [{"role": "system", "content": "Synthetic test conversation."},
            {"role": "user", "content": "First ask, only in pass one."}]
    for npass in range(4):
        rows = comp.compress(rows + [long_ask] + _progress(f"pass{npass}"), force=True)
        quotes = _section(_summary_row(rows), cc._LEAN_USER_MESSAGES_HEADING)
        assert quotes.count("> HEAD a") == 1, f"pass {npass + 1}"
        assert "> First ask, only in pass one." in quotes, f"pass {npass + 1}"
        assert all(len(line) - 2 <= cc._LEAN_USER_MESSAGE_MAX_CHARS for line in quotes.splitlines())
