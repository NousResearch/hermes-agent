# Post-branch drift + private-overlay-phase-h residual patches

**Added:** 2026-06-21, in response to a Council finding that the line-partition's
"0 unaccounted" was achieved by file-level padding rather than proven coverage.

When the honest (non-padding) `partition.py` was run, **225 added lines** were truly
unaccounted by the strict "proven-in-a-deferred-patch-or-PR-diff" test. This directory
makes every one of them **pullable** so nothing is lost, and the partition reaches a
TRUE 0 (each line is now in either an owner-PR diff or one of these proof patches).

## `post-branch-drift/` — William-authored, contributable, but drifted

These 6 files have William-authored content in the overlay HEAD (`378b32ef7`) that
**postdates** when the owner PR branch was cut, so the owner PR's diff is an
as-of-branch-cut snapshot missing the later lines. Each `.patch` is the FULL
`git diff v0.16.0 -- <file>` = the **current overlay state** for that file. It SUPERSEDES
the owner PR's version of that one file (do not stack both; this is the newer full version).

| Patch | Owner PR | Genuine drift it adds over the PR |
|-------|----------|-----------------------------------|
| `cli.py.patch` | #49917 | autopilot re-apply-on-agent-rebuild block (`autopilot_mode = True` / `_autopilot_goal` survive route changes) — **confirmed absent** from the pushed #49917 |
| `tools_mcp_tool.py.patch` | #48069 | comment drift + `_inflight_tasks.discard` wrap |
| `agent_system_prompt_prelude.py.patch` | #48101 | docstring + config-example lines (tier2 humanizer pass touched overlay after branch cut) |
| `agent_transports_chat_completions.py.patch` | #49644 | `effort in {"high","xhigh","max"}` variant line |
| `tests_hermes_cli_test_model_switch_copilot_api_mode.py.patch` | #50064 | added assertion + doc lines |
| `gateway_run.py.patch` | (orphan) | generic image/media-type detection block — contributable but introduced by the phase-h overlay merge, never assigned to a PR |

**Verified:** all 6 apply cleanly (`git apply --3way`) onto v0.17.0 (`2bd1977d`); 0 private
tokens (agy/cmx/auto-router/review-path scan = 0).

## `private-overlay-phaseh/` — NOT contributable

These 6 files' overlay changes were introduced by the private v2026.6.5 update-merge
machinery (`Hermes: phase-h: apply 60 MODIFY decisions from EMPIRICAL_MERGE_MATRIX.md`).
They are copilot-context test-file modifications + inventory/skills_tool edits that are
part of the private update overlay, not William's PR-candidate work. Kept here only as a
pullable reference; **not** intended for any public PR.

## How to pull

```
git fetch <fork> deferred/residual-lines-on-v0.17.0 && git checkout FETCH_HEAD
git apply --3way deferred/post-branch-drift/<file>.patch   # supersedes owner PR's file
```

## Net effect on the partition

With these patches present, the honest partition is: every `git diff v0.16.0..HEAD` added
line is now in EITHER (a) an owner PR's diff, OR (b) one of these proof patches (post-branch
drift = William/contributable; private-overlay-phaseh + the agy/cmx/auto-router lines in the
sibling category dirs = deferred). **0 lines remain unproven.** The earlier headline "0"
was padded; this is the real, demonstrable 0.
