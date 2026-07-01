# FIX BRIEF — Preserve-Prompt build, code-review BLOCK (4 findings)

The initial build (`docs/BUILD-SPEC-preserve-prompt-restart.md`, 4 commits already on this branch) passed
12 in-memory unit tests but an Opus diff-review found 4 real bugs. Fix ALL FOUR with TDD (RED first).
Commit per fix, author `Apollo <apollo@ang.ventures>`. Do NOT push/PR/restart. Hermetic test command:
```
env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c 'cd ~/.hermes/worktrees/gw-preserve-prompt; PYTHONPATH=$PWD ~/.hermes/hermes-agent/venv/bin/python -m pytest <files> -p no:cacheprovider -o addopts="" -q'
```

---

## FIX 1 (CRITICAL — the feature currently no-ops in prod): flag must survive the SQLite round-trip

**Root cause (ground-truthed):** the resume site reads the RELOADED history via
`self._session_db.get_messages(session_id)` (`gateway/run.py:13453`), which reconstructs each message from
FIXED columns of the `messages` table (`hermes_state.py:703` schema; `:3025` `append_message` takes only
named params; `:3297` `get_messages`). The `_interrupt_close` key is NOT a column → it is **dropped on
persist and never returned on reload**. So `_is_interrupt_close_tail` (which keys on `_interrupt_close`)
is ALWAYS False in production → the surface-and-ask branch never fires → the whole fix silently no-ops.
The 12 unit tests pass only because they operate on in-memory dicts where the extra key trivially survives.

**Fix — use an EXISTING round-tripping column as the carrier: `finish_reason`.** `finish_reason` is a real
`messages` column that round-trips (`get_messages` returns it; `get_messages_as_conversation:70` restores
it on assistant msgs). Set `finish_reason="interrupt_close"` on the synthetic close turn — no schema
migration, collision-proof (no normal turn uses that value).

Changes:
- `agent/message_sanitization.py` `close_interrupted_tool_sequence`: on the appended synthetic close turn
  AND on the in-place flagged assistant-text tail, ALSO set `"finish_reason": "interrupt_close"` (keep
  `_interrupt_close: True` too — belt-and-suspenders for the in-memory path; the persisted discriminator
  is `finish_reason`). Add a module constant `_INTERRUPT_CLOSE_FINISH_REASON = "interrupt_close"`.
- `gateway/run.py` `_is_interrupt_close_tail`: return True if `t.get("_interrupt_close") is True` **OR**
  `t.get("finish_reason") == "interrupt_close"` (so it works BOTH pre-persist in-memory AND post-reload
  from SQLite).
- **VERIFY `append_message` actually persists `finish_reason` from the message dict.** Trace how
  `_flush_messages_to_session_db` (agent side) maps a message dict → `append_message(finish_reason=...)`.
  If the flush does NOT pass `finish_reason` through from the dict, wire it so the close turn's
  `finish_reason` reaches the DB. This is the load-bearing check — do not assume it flows.

**RED test (`tests/gateway/test_resume_flag_roundtrip.py`) — THE missing acceptance gate (AC-1/AC-5iii):**
persist a REAL interrupted turn to a temp `HERMES_HOME` via the actual SessionDB (`append_message` for a
user turn, an assistant(tool_calls), a tool result, then `close_interrupted_tool_sequence` + the real
flush/persist path), then RELOAD via `get_messages(session_id)` and assert:
- the reloaded tail is the synthetic close turn AND `_is_interrupt_close_tail(reloaded_history)` is True
  (proves the discriminator survives SQLite);
- a control: a normal assistant turn with content "Operation interrupted." but `finish_reason=None`
  reloads with `_is_interrupt_close_tail` False (collision-proof).
Confirm RED on the current code (flag lost → False), GREEN after the finish_reason carrier.

---

## FIX 2 (SAFETY): INV-D7 interlock must be TURN-scoped, not round-scoped

**Root cause:** `_block_resume_summary_only_tools` (`agent/tool_executor.py`) sets
`agent._resume_summary_only = False` on the FIRST blocked tool-round. But a resume turn is a full agentic
loop: the block-results are fed back to the model, which is re-invoked and can emit NEW tool calls — now
with the gate already consumed → they EXECUTE. The interlock must hold for the WHOLE resume turn, until
the next genuine HUMAN inbound message.

**Fix:** do NOT clear `_resume_summary_only` inside `_block_resume_summary_only_tools`. Instead:
- keep blocking every tool round while `_resume_summary_only` is True;
- clear it ONLY at the start of the next genuine human-inbound turn. The gateway resume site already sets
  `agent._resume_summary_only = True` (surface-and-ask) / `False` (other branches). Ensure that on ANY
  normal inbound user turn (message present, not the internal resume turn), `_resume_summary_only` is
  reset to False exactly once at turn start. Find where an inbound user message begins handling and clear
  it there (or confirm the existing `else: agent._resume_summary_only = False` at the resume site runs on
  every subsequent inbound turn — see FIX 4).

**RED test (`tests/agent/test_resume_interlock.py`, extend):** with `_resume_summary_only=True`, simulate
TWO tool-call rounds within the SAME turn (block round 1 → model re-emits a tool call round 2) → assert
round 2 is ALSO blocked (0 real tool invocations across both). This is the test that currently ENSHRINES
the hole — rewrite it to assert the correct turn-scoped behavior. Then a separate genuine next-human-turn
executes normally.

---

## FIX 3: allow pure reads on the resume turn; don't false-alert

**Root cause:** the interlock blocks EVERY tool indiscriminately. AC-3/INV-D7 require pure reads of
already-loaded history to be allowed, and blocking a benign read fires a false
`RESUME_AUTOCONTINUE_VIOLATION → #alerts` (a surface Ace keeps quiet-unless-degraded).

**Fix (minimal + safe):** the simplest correct rule the spec endorses is "the resume summary turn should
emit TEXT only." So: keep blocking all tool calls, BUT only emit `RESUME_AUTOCONTINUE_VIOLATION` (the
#alerts-routed warning) when the blocked call is a FORWARD/MUTATING tool, not a known read-only one. Keep
a small read-only allowlist constant (e.g. read_file, search_files, ls-like) — a blocked read is logged
at debug (not #alerts) and still returns the summarize-only synthetic result. If classification is
uncertain, treat as forward (fail-safe, still blocked, still alerted). Do NOT let a read consume/relax
the turn-scoped gate (that was FIX 2).

**RED test:** a blocked read-only tool → no `RESUME_AUTOCONTINUE_VIOLATION` warning emitted (assert the
logger/callback not called with the violation signal); a blocked mutating tool → violation emitted.

---

## FIX 4: prove the `_resume_summary_only` reset actually runs on ordinary turns

**Root cause:** the happy path is a text-only summary (no tool calls) → the block fn never runs →
`_resume_summary_only` stays True → the flag must be cleared by the NEXT inbound turn's gateway path. If
that reset is gated behind "resume-entry exists," it won't run on an ordinary "go" turn → the go turn's
first real tool call is wrongly blocked + false-alerted.

**Fix + RED test (`tests/gateway/test_resume_summary_only_reset.py`):** simulate: (1) resume turn sets
`_resume_summary_only=True` and emits text only (no tools, flag stays True); (2) next inbound human turn
("go") → assert `_resume_summary_only` is False BEFORE its tools run, and a tool call executes normally.
If the current reset only runs when `_is_resume_pending`, move/add an unconditional reset at the start of
normal inbound handling.

---

## AC-4 (cache/alternation — the reviewer also flagged it was unproven)
Add `tests/gateway/test_resume_note_cache_safe.py`: assert the resume note is PREPENDED to the new
synthetic turn (placement unchanged vs the original code), and that the assembled history for the resume
turn has no two-same-role-in-a-row (alternation validator). Keep it a focused unit.

## FINAL
Run the 3 focused files + the two new ones + `tests/agent tests/gateway` (report counts; a NEW failure
that is an intended-behavior change gets the OLD-shape test updated w/ a comment, not the feature
reverted). Report per-fix: file, test, RED→GREEN, commit SHA. Report the FIX-1 `finish_reason`-flows-to-DB
verification result explicitly (it's load-bearing).

## REFERENCE
- `hermes_state.py:703` messages schema (no metadata col); `:3025` append_message (named cols only);
  `:3297` get_messages; `:3619` get_messages_as_conversation (finish_reason restore at ~:3688).
- `gateway/run.py:13453` resume-load reads `get_messages`; `:679` `_is_interrupt_close_tail`;
  `:17931` resume site (sets `_resume_summary_only`).
- `agent/tool_executor.py` `_block_resume_summary_only_tools` (+`execute_tool_calls_concurrent/sequential`).
- `agent/message_sanitization.py:362` `close_interrupted_tool_sequence`.
- Find `_flush_messages_to_session_db` (agent side) → confirm the dict→append_message field mapping for
  `finish_reason`.
