# Claude Code Task Packet — Phase 2 Structured Pending Input Queue

You are preparing to implement **one narrow Hermes phase** in an isolated git worktree.

> IMPORTANT: This is a task packet. Do not broaden the scope. Do not implement task registry, background worker lanes, or follow-up classifier beyond the minimal structured pending queue foundation.

## Worktree

```text
/tmp/hermes-orchestrator-phase-2
```

## Baseline

Phase 1 was committed on `main` as:

```text
afd8b7f72 feat(cli): add integrated busy input mode
```

Phase 1 added:
- `/busy integrated` in CLI command handling and command registry.
- Integrated CLI busy payload tagging via identity-only sentinel.
- Gateway recognition of `display.busy_input_mode: integrated` with queue-like fallback.
- Tests proving queue/steer/interrupt semantics remain intact.

## Source plans

Read these files for context:

```text
docs/plans/2026-05-12-hermes-orchestrator-first-update-plan.md
docs/plans/2026-05-12-integrated-busy-queue.md
docs/plans/2026-05-12-orchestrator-implementation-handoff.md
docs/plans/2026-05-12-claude-phase1-task-packet.md
docs/plans/2026-05-12-claude-phase2-task-packet.md
```

## Current phase

Phase 2 — Structured Pending Input Queue.

## Product intent

Hermes currently has several pending-input paths that rely on raw strings, image tuples, per-session pending slots, and ad-hoc merge behavior. That is enough for simple queueing, but not for Telegram-style fragmented input where a follow-up may be text, command, media, correction, status query, or a hard boundary.

Phase 2 should introduce a **small, explicit, structured pending input representation** that preserves boundaries and order without changing user-facing semantics.

The goal is a foundation, not the full Manus-like task fabric.

### Deeper product purpose

This is not a generic queue refactor. The long-term product direction is to make Hermes a **focused orchestrator agent for Woo**:

- Hermes remains the accountable main orchestrator: understanding Woo's intent, preserving context, deciding what belongs in the current task, and deciding what should become a separate focused worker task.
- Claude Code, delegate_task subagents, and future workers are implementation/review lanes, not replacements for Hermes' judgment.
- The user experience target is: Woo can send fragmented Telegram/CLI/TUI follow-ups naturally, and Hermes can preserve their order, boundaries, and purpose instead of flattening them into ambiguous text.
- Phase 2 exists because Hermes cannot safely create focused agents or route work later unless the pending inputs are first represented explicitly.

### Ralph concept / focused-agent framing

Treat **Ralph** as the conceptual name for the future focused worker/task unit that Hermes may eventually create and manage.

For Phase 2, do **not** implement Ralph, task registry, worker lanes, or background agents. Instead, design the pending-turn structure so it can later support Ralph-like focused agents by preserving:

- what the user actually sent;
- whether the input is a continuation, correction, command, media/caption, or hard boundary;
- source/session metadata;
- ordering and coalescing boundaries;
- enough metadata for Hermes to later decide: "append to current focus", "steer current worker", "create a new Ralph/focused task", or "ask user for clarification".

In other words: Phase 2 is the **input substrate** for future Ralph/focused-agent orchestration, not the Ralph implementation itself.

## Phase 2 goal

Create a structured pending-turn queue abstraction that can represent:

```text
text fragments
slash commands / control messages
media / image / document attachments
caption-like boundaries
source/session metadata
created_at ordering
```

Then apply it minimally where it reduces current fragility, while keeping existing behavior green.

## Scope boundaries

### Do

1. Inspect current pending-input paths before editing:
   - `cli.py`
   - `gateway/platforms/base.py`
   - `gateway/run.py`
   - `tui_gateway/server.py`
   - `ui-tui/src/app/useSubmission.ts`
   - relevant tests under `tests/cli`, `tests/gateway`, `tests/tui*` if present.

2. Propose/implement a minimal structured type, likely in a new file such as:

```text
agent/pending_turn_queue.py
```

Suggested dataclass shape:

```python
@dataclass
class PendingTurnItem:
    id: str
    session_key: str | None
    source: str                 # telegram|cli|tui|api|unknown
    kind: str                   # text|command|media|attachment|control
    text: str | None
    media_refs: list[str]
    media_types: list[str]
    created_at: float
    reply_to: str | None
    task_hint: str | None
    boundary: str               # coalesce|hard|caption|command
    raw: Any | None = None
```

A helper/container may also be appropriate:

```python
class PendingTurnQueue:
    append(item)
    drain_coalescible_text_until_boundary()
    peek()
    pop()
    __len__()
```

3. Add conversion helpers so existing legacy payloads can round-trip safely:

```python
from_legacy_cli_payload(payload) -> PendingTurnItem
maybe_to_legacy_cli_payload(item) -> payload
from_gateway_event(event, session_key) -> PendingTurnItem
```

4. Preserve Phase 1 behavior:
   - `/busy integrated` still works.
   - Integrated CLI tagged payload does not collide with image tuples.
   - Gateway `integrated` remains queue-like fallback until full gateway structured routing exists.

5. Preserve command/media boundaries:
   - Slash commands must not be swallowed into text coalescing.
   - Image/document/media payloads must not be merged into plain text.
   - Telegram album/media merge behavior must remain intact.

6. Add focused unit tests for the new structured queue abstraction.

7. Add/adjust integration tests only where the new abstraction is wired into existing CLI/gateway paths.

### Do not

- Do not implement Phase 3 task registry.
- Do not implement Phase 4 detached/background worker lanes.
- Do not add `delegate_task(background=True)` yet.
- Do not implement a model-based follow-up classifier.
- Do not change default busy mode.
- Do not redefine `/busy interrupt|queue|steer|integrated` semantics.
- Do not do broad TUI/gateway refactors unless required for a narrow queue boundary fix.
- Do not commit changes.
- Do not touch credentials, `.env`, auth files, or unrelated local config.

## Preferred implementation strategy

Use a **compatibility-first, purpose-fit** approach:

```text
Step 1: Create structured data model + tests.
Step 2: Add legacy conversion helpers + tests.
Step 3: Wire the CLI coalescing path to use helpers if low-risk.
Step 4: Wire gateway pending merge only if it can be done without broad behavior changes.
Step 5: Leave TUI changes as documented follow-up if direct wiring is too broad for Phase 2.
```

If a full gateway replacement is too risky, implement the abstraction and tests first, then adapt only the safest path and document remaining bridge points. Phase 2 should be mergeable and safe, not maximal.

### Deliberation and time-boxing guidance

Before editing code, spend real reasoning time on the smallest useful design. This is an Opus-level design/implementation task, not a mechanical refactor.

Use this decision rule:

```text
Implement until Phase 2 has a coherent, testable, reviewable foundation.
Stop before the work becomes architectural expansion.
```

Do not optimize for "more code". Optimize for **purpose-fit leverage**:

- If a small dataclass + queue + conversion helpers + two safe integration points solves the Phase 2 purpose, stop there.
- If replacing every pending-message path would require broad gateway/TUI surgery, do not do it in Phase 2; document bridge points instead.
- If you find yourself inventing task lifecycle, worker state, model classification, routing policy, or Ralph behavior, stop and report it as Phase 3+ design material.
- Prefer boring, explicit, serializable shapes over clever abstractions.
- Prefer compatibility wrappers over call-site churn.
- Prefer tests that protect semantics over abstractions that merely look elegant.

### Model and effort expectation

Use the strongest available Claude Code reasoning model for this worker run: **Claude Opus 4.7 / Opus-class model**. Use high/max effort reasoning. Do not silently downgrade to a cheaper model unless the controller explicitly approves it.

## Likely files allowed

```text
agent/pending_turn_queue.py
cli.py
gateway/platforms/base.py
gateway/run.py
tests/agent/test_pending_turn_queue.py
tests/cli/test_busy_queue_coalescing.py
tests/gateway/test_*pending*.py
tests/gateway/test_session_race_guard.py
```

Potentially inspect-only unless truly needed:

```text
tui_gateway/server.py
ui-tui/src/app/useSubmission.ts
```

If you modify TUI files, explain why and add appropriate tests or a clear manual verification note.

## Required tests

Run the narrowest relevant tests first:

```bash
python -m pytest tests/agent/test_pending_turn_queue.py -q
python -m pytest tests/cli/test_busy_queue_coalescing.py tests/cli/test_busy_input_mode_command.py -q
python -m pytest tests/gateway/test_restart_drain.py tests/gateway/test_session_race_guard.py -q
```

Then run any additional tests covering files you changed.

If possible, also run:

```bash
python -m pytest tests/gateway -q
```

If broader gateway/CLI tests fail for unrelated environment reasons, report exact failures and keep targeted tests green.

## Acceptance criteria

Phase 2 can be accepted only if:

- A structured pending item representation exists with tests.
- Text fragments preserve order.
- Slash commands are represented as command/control or hard boundaries, not swallowed into text.
- Media/attachment payloads keep explicit boundaries and metadata.
- Existing CLI busy queue coalescing still works.
- Phase 1 `/busy integrated` behavior remains green.
- Gateway `integrated` remains queue-like fallback and does not interrupt active sessions.
- No broad task registry / worker lane / classifier code is introduced.
- Targeted tests pass.
- `git diff --check` passes.
- Runtime Python files compile.

## Review risks to watch

- Dataclass or sentinel shape that cannot be safely serialized if it later crosses process boundaries.
- Accidentally changing legacy pending payload semantics before all call sites are migrated.
- Treating slash commands as model text.
- Treating image tuples as integrated text payloads.
- Breaking Telegram media group / album merge behavior.
- Adding a large abstraction but not using it anywhere meaningful.
- Overbuilding Phase 2 into task registry or worker lane behavior.

## Return format

At the end, report:

```text
SUMMARY:
PURPOSE-FIT DESIGN RATIONALE:
WHAT YOU INTENTIONALLY DID NOT BUILD:
RALPH/FUTURE FOCUSED-AGENT NOTES:
CHANGED FILES:
TESTS RUN:
RESULTS:
KNOWN RISKS:
OPEN QUESTIONS:
SUGGESTED NEXT REVIEW COMMANDS:
```

## Controller note

Hermes main orchestrator will review your diff, run tests, request independent spec/quality reviews, and only then import accepted changes back to main.

The controller expects a focused, purpose-fit foundation for future Ralph/focused-agent orchestration — not an impressive-looking broad rewrite.
