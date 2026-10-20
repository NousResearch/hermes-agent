# Task: implement the fix for upstream hermes-agent#141562

You are implementing a contained gateway fix in this repository (a fresh clone of
NousResearch/hermes-agent, branch `fix/watch-notification-admission-conflict`, cut from
`upstream/main`).

**Full specification:** read `/home/kv-thinkpad-t420-ubuntu/Documents/equational-wiki/records/handoffs/2026-10-19-handoff-upstream-pr-141562-watch-notification-drop-fix.md` FIRST and follow it.

## Hard constraints

1. Work ONLY in this clone. NEVER read-write-modify `~/.hermes/hermes-agent` (it is the LIVE
   gateway's running code). Reading its `venv` python for test execution is allowed:
   `~/.hermes/hermes-agent/venv/bin/python -m pytest ...`
2. TDD, strictly: write test 1 (reset-pinned completion → DROP) against current main first,
   run it, confirm it fails the right way (event requeued, no drop), THEN implement the fix.
3. Re-locate all symbols on CURRENT main by name (`_owner`, `_drain_watch_notifications`,
   `move_route_wakes`, `_hm_stop_owner_turn`, `get_compression_tip`) — the handoff's line
   numbers are from a rev 1014+ commits old and WILL have shifted.
4. Read commit `19c96fb8c3` ("fix(gateway): bound failed completion notification retries")
   before designing the requeue-cap/backoff so your bounded-requeue policy matches it instead
   of diverging.
5. Check upstream movement before coding: issues #141562, #140899, #141153 (all OPEN as of
   2026-10-19 ~14:00 UTC) and any existing PRs touching `run_notifications.py`. If you find a
   merged fix, STOP and report instead of building a competing one.
6. Commits: `fix(gateway): <one-line>` style. Tests per the handoff's Tests section (all 4).
   Verification bar: `pytest tests/gateway/ -k "automation or notification or watch"` green.

## STOP boundary — do not cross

When the implementation is committed and tests are green, STOP. Do NOT `git push`, do NOT
open a PR, do NOT comment on upstream issues. A review ladder and human approval happen
before anything leaves this machine.

## Final report (last message in this session)

Print a summary with: files changed, commit hashes, test results (exact pytest command +
pass/fail counts), the RED-phase evidence for test 1, any deviations from the handoff design
and why, and open questions for the reviewer.
