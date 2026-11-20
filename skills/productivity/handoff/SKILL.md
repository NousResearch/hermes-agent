---
name: handoff
description: "Capture session state so the next agent resumes fast."
version: 1.0.0
author: baoyu0, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Handoff, Session, Continuity, Context, Productivity]
    related_skills: [weekly-review-planning, document-to-action-items]
---

# Handoff Skill

Capture the working state of a session in a `HANDOFF.md` at the project root, so the next session — another model, another agent, or the same one days later — resumes without the user re-explaining the job. It writes one file and reads it back; it does not plan the work, summarise chat logs, or stand in for commits, PRs, and issue threads, which stay authoritative.

## When to Use

- A session ends with work in flight and the next one picks it up.
- Context was compacted or truncated and the thread of work has to survive it.
- Work spans sessions, agents (Hermes, Codex, Claude Code), or machines.
- "Write a handoff", "save where we are", "leave notes for the next session".

Don't use for: a task that finishes in one session (the final reply is enough), or project documentation that should outlive the work (README, ADR, design doc).

## Prerequisites

- A project root to hold `HANDOFF.md`: the repo, the workspace, or the directory the work lives in. Never a temp directory — the next session has to find it.
- `scripts/check_handoff.py` for the mechanical check. Stdlib only, no network, runs anywhere.

## How to Run

- **Write:** ask for a handoff, or run `/skill handoff`. The skill writes `HANDOFF.md` with `write_file`.
- **Read:** at the start of a session, `read_file` on `HANDOFF.md` before anything else. If it is absent, say so and ask for a one-line state — do not reconstruct from guesses.
- **Validate:** `terminal` → `python <skill-dir>/scripts/check_handoff.py HANDOFF.md`.

The bare `/handoff` slug belongs to a built-in command, so the skill alias is not registered; `/skill handoff` always works.

## Quick Reference

| Section | Answers | Failure it prevents |
|---|---|---|
| Current Task | what are we building or fixing, in one line | next session re-derives the goal |
| Done | finished work, each with a path, commit, or PR | repeating landed work |
| Blockers | what is stuck, with the error text and where it lives | re-hitting a wall with no context |
| Next Steps | ordered actions, each with an acceptance condition | drifting into a different job |
| Landmines | trap → root cause → fix | the same mistake twice |
| Suggested Skills | skill name and why it applies | next session hunting for the right tool |

## Procedure

### 1. Read the existing file

If `HANDOFF.md` exists, `read_file` it first. It is a log, not a document: keep earlier entries and add this session's, so the project arc stays traceable. Done when you know whether this is the first entry or an append.

### 2. Fill the six sections from evidence

Take each item from something you can point at: a commit, a diff, a `read_file` line, a command output, a PR URL. A claim with no pointer is a guess — drop it or mark it open. Done when every Done and Blockers item carries a path, a commit, or a URL.

### 3. Write Next Steps as testable actions

Each step names the action, the thing that proves it is finished, and the dependency if any. "Look into the flaky test" is not a step; "make `tests/web/test_chat_page.py::test_follow_scroll` pass on Windows, prove with the suite green on a fresh checkout" is. Done when a stranger could tell whether step 1 is complete without asking.

### 4. Write Landmines with the root cause

The value is the mechanism, not the symptom. Record trap, why it happens, and the move that avoids it — for example "`term.write()` applies asynchronously, so scrolling right after it uses the stale viewport; scroll inside the write callback". Done when each landmine explains why, not just what.

### 5. Bound and redact

No secrets, tokens, cookies, internal hostnames, or customer data — reference where the credential lives instead of copying it. Link to PRDs, issues, and specs rather than restating them; they change and the handoff should not fork them. Done when the file can be committed to a repo without a second thought.

### 6. Validate and close

Run the checker, fix what it reports, and end the file with `▶ Next session: read HANDOFF.md first`. Done when the checker exits 0 and the closing line is present.

## Template

```markdown
# HANDOFF

## Current Task
One line: what we are building or fixing.

## Done
- [x] What landed — `path/to/file.ts:120`, commit `abc1234`, PR #42

## Blockers
- What is stuck — exact error text, where the log lives

## Next Steps
1. Action — acceptance: how we know it is done

## Landmines
- Trap: what happened. Root cause: why. Fix: what to do instead.

## Suggested Skills
- `skill-name` — why it applies here

▶ Next session: read HANDOFF.md first
```

## Pitfalls

- **Rewriting the file.** Replacing last week's entry destroys the arc; append.
- **Landmines without a mechanism.** "Hit error X" sends the next session into the same wall.
- **Next steps with no acceptance condition.** The next session cannot tell when to stop.
- **Copying secrets or internal addresses.** A handoff is meant to be committed.
- **Writing to a temp directory.** The next session will not find it.
- **Duplicating the tracker.** Restating a PR or issue instead of linking it makes two sources of truth.
- **Handing off everything.** If the work is done, the handoff is one line saying so, not a report.

## Verification

- [ ] `python scripts/check_handoff.py HANDOFF.md` exits 0.
- [ ] Every Done and Blockers item names a path, commit, or URL.
- [ ] Each Next Steps entry has an acceptance condition.
- [ ] Each Landmine states the root cause, not only the symptom.
- [ ] No secrets, tokens, or internal hostnames; no machine-local absolute paths.
- [ ] File sits at the project root, earlier entries kept, closing line present.
