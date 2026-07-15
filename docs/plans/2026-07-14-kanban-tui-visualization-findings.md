# Kanban TUI Visualization Findings

Date: 2026-07-14

## Scope decision

This pass builds and verifies the read-only activity projection, local TUI RPC, pure rail model, and fixture-rendered Ink surface. It deliberately does **not** mount the dock, start polling from a live TUI session, change keyboard routing, or write anything into transcript/history state.

## Official Nous documentation consulted

- [Kanban reference](https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban)
- [Kanban tutorial](https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban-tutorial)
- [TUI guide](https://hermes-agent.nousresearch.com/docs/user-guide/tui)
- [Subagent delegation](https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation)
- [Slash-command reference](https://hermes-agent.nousresearch.com/docs/reference/slash-commands)

## Boundaries confirmed by the docs

1. Kanban is a durable, multi-profile, multi-board queue; `delegate_task` is a current-session fork/join mechanism. The Kanban dock must not be folded into `/agents` or imply that board workers belong to the current transcript.
2. Human, model-tool, CLI, and dashboard surfaces share the `kanban_db` layer. The TUI should consume a shared read projection through its local JSON-RPC gateway, not call the dashboard HTTP API.
3. Boards are separate databases and hard isolation boundaries. Multi-board grouping must be explicit and deterministic.
4. The TUI owns screen chrome while Python owns state and agent/runtime behavior. The activity spine belongs outside message history.
5. Current TUI ambient activity is hidden by default. A permanently noisy activity wall would fight the product's existing defaults.
6. Worker visibility endpoints can expose operational details such as PID and log path, but those are unnecessary and too sensitive/noisy for the composer-adjacent dock.
7. `/kanban` can mutate or inspect the board mid-run, but this V1 dock should remain read-only. Existing slash commands and the dashboard remain the control surfaces.

## Strong visual model

### Resting line: answer one question

> What durable work needs my attention right now?

Use one calm line, hidden when irrelevant. Prioritize blocked/failed/stale work over ordinary running work. Progressively collapse title and rail before allowing wrap.

### Focused spine: separate three signals

- **Topology:** indentation and `├──` / `└──` show dependency structure.
- **Lifecycle:** `●`, `◉`, `○`, `!`, `×` show task/run state.
- **Health:** restrained semantic color and a stale label show heartbeat condition.

Keeping these signals separate prevents the rail from pretending that every graph is a linear checklist.

### Information hierarchy

1. attention state and task title
2. dependency position
3. assignee/profile
4. elapsed or heartbeat freshness
5. board label only when multiple boards are relevant

Raw IDs, prompts, bodies, comments, commands, environment, full logs, and worker context do not belong in this surface.

## Improvements beyond a basic status line

### V1-worthy

- Show why a task is waiting: dependency-gated versus merely ready.
- Preserve node positions when heartbeat timestamps change; movement should mean topology/status changed.
- Compress large sibling sets into a truthful `+N more` row rather than a scrolling log.
- Treat stale heartbeat as degraded health, not failure.
- Keep blocked reason bounded and presentation-safe.
- Make single-board mode visually lighter; only group/label boards when there is more than one.
- Keep all understanding available in static mode; no animation required.

### High-value follow-up after visual approval

- Add a safe `attempt_count` and latest terminal outcome so retries/crash recovery are visible without exposing run logs. Retry history is a first-class Kanban concept in the official tutorial.
- Add structured heartbeat fields (`phase`, `current`, `total`) only when produced by the worker contract. Never infer percentages from prose.
- Consider an event/change-sequence transport after bounded polling proves useful. Do not start with cross-process notification subscriptions or transcript delivery.
- Add a full-screen read-only graph/inspector only if users need more than the composer-adjacent spine; do not turn the resting dock into a tiny dashboard.

## Visual review decisions (2026-07-15, accepted)

The Fable design review was accepted in full. These are now binding visual decisions:

1. **Segmented row paint.** The rail/connector prefix is always muted, the title renders in
   the default foreground, `— owner` is dim, and only the glyph and the `· state` word carry
   the lifecycle tone. No row is ever painted edge-to-edge in one color.
2. **One light rail.** The heavy `┃` rail is retired; topology uses `│` exclusively and never
   encodes lifecycle. Ancestor continuation segments (`│   ` vs `    `) keep branch lines
   attached to their parent at depth 2+, and the last visible child keeps `├──` when an
   omitted-siblings summary row follows.
3. **State survives truncation.** Labels drop the blocked-reason detail first, then the
   owner, then title characters; the state word is removed only when fewer than 8 cells
   remain for the title. `— unassigned` is never rendered.
4. **Accent means live.** Only fresh-heartbeat running work (and the active dock lamp) uses
   accent. `ready` / `review queued` are neutral (default foreground); `scheduled`,
   `triage`, and `archived` stay muted.
5. **Board headers are identity, not activity.** Bare board name in bold default foreground
   (no `Kanban ·` repetition); an unavailable board appends `· unavailable` in the warn tone.
   Name and suffix share one width budget so the row can never wrap, and at widths too
   narrow for both, the error signal survives and the name is dropped.
6. **Dock ladder.** Wide: `Kanban · N need attention · N active · <headline> <state>`, with
   the headline ranked failed > blocked > stale > running > recent-terminal; if fewer than
   12 cells remain for the headline the dock falls back to counts. Narrow (<28 cols) speaks
   glyph shorthand with no separate lamp prefix, painted as segments — `K` neutral, `!N`
   warn, `◉N` accent, `●N` success; tiny (<14 cols) drops the `K`. Zero-count badges are
   never shown: completed-only recent activity reads `K ●N`, never `◉0`.
7. **Title identity is inviolable.** Task titles render in the default foreground with no
   dim and no tint on every row, including muted-tone rows; only the glyph and state word
   carry lifecycle color.

## Activation gate

Live mounting should wait until all of these are demonstrated against isolated fixtures:

- backend projection is bounded, deterministic, and read-only
- malformed links cannot recurse indefinitely
- multi-board failures degrade per board
- no session lookup, message event, notification queue, or subscription side effect
- wide, narrow, blocked, stale, failed, branching, and empty render states are visually approved
- polling cleanup and unchanged-snapshot behavior are tested
- integration proves transcript/history length is unchanged

Until then, the implementation remains dormant and test-only by design.

## Current implementation pass

Delivered in the isolated worktree:

- shared read-only activity projection with recent-completion visibility, graph bounding, cycle handling, forced secret redaction, and presentation-field allowlisting
- DB-pin isolation that rejects cross-board relabeling instead of returning the pinned board's data under another slug
- independently bounded seed, recursive-scan, display, and per-child parent-edge budgets; every overflow sets `truncated` and renders `+ more activity not shown`
- preserved multi-parent truthfulness, including references to parents outside a truncated presentation slice
- dashboard `/workers/active` reuse of a shared database helper without changing its response contract
- local `kanban.activity` JSON-RPC aggregation with canonical multi-board deduplication, explicit overflow diagnostics, and no session/delivery side effects
- typed payloads, pure normalization/presentation helpers, and static Ink dock/spine components
- focused wide, narrow, branching, blocked, failed, stale, multi-board, empty, and dormant-integration tests

Deliberately deferred:

- live TUI mounting or composer integration
- polling/store lifecycle
- keyboard focus and task selection
- animation
- live-session screenshots

Those deferred items are the activation phase, not missing invisible wiring. The standalone fixture render for this pass produced:

```text
│ ◉ Adaptive cycle core — implementer · running
├── ● Models — implementer · completed
├── ◉ Persistence — implementer · running
└── ○ Independent review — implementer · ready
│ ! Visual approval — implementer · blocked: Needs human visual choice

K !1 ◉2
```

(Fixture updated 2026-07-15 for the accepted narrow-dock grammar; the spine rows read the
same in plain text — the accepted changes redistribute color across segments rather than
changing the row text.)
