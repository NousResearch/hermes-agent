# ADR 0020 — Kanban chain-break + note-stall detection

**Status:** Proposed (sw-architect, card t_9106d1e1, 2026-11-21).
**Binding for:** `hermes_cli/kanban_db.py` (G-55 successor guard + wake, completion close), `hermes_cli/kanban_db_notify.py` (subscription + event-kind allowlist), `hermes_cli/kanban_db_dispatch.py` (TTL/heartbeat reclaim), `gateway/kanban_watchers_notifier.py` (5 s notifier poll, `TERMINAL_KINDS`/`_WAKE_KINDS`/`diagnostic_event`), `gateway/kanban_watchers_dispatcher.py` (dispatch/orphan tick), `tools/kanban_tools.py` (`kanban_comment` / `kanban_heartbeat` / `kanban_schedule` + `fleet-wake-driver`), `hermes_cli/config_defaults.py` (`kanban.successor_guard` / `note_stall` / `board_liveness`), and the estate cron family `~/.hermes/scripts/stall-sentinel.py` + `board-followthrough.py` (design `hermes-estate/docs/BOARD-FOLLOWTHROUGH-DESIGN-2026-11-03.md`) — which this ADR extends but does not duplicate.
**Design companion (build DAG):** `docs/adr/0020-impl.md` companion not yet written — build cards t_d5708f00 (A+D-chain) + t_68d5b5a4 (B+C+D-stall) + replay t_546e126a depend on this doc; handoff comments pin file:line seams and test contracts.
**Falsification:** revisit when any §7 trigger fires.

## 1  Context

Operator report 2026-11-21, board gargantua01 — two incidents with one root symptom: the board "goes dead-quiet while work is still open, and only operator prompts discover it."

1. **Chain-break on completion — the Q8 bisect card.** Body and `summary` at `kanban_complete` both said "preparing handoff successor." No `SUCCESSOR: required` line, no `metadata.successor_spec`/`downstream_action`, no `created_cards`, no `t_<hex>` child link. `complete_task` (L3040 `~kanban_db.py`) discharged the card, emitted `completed`, ran `_gate_successor` → pass (gate only sees the explicit declaration or `successor_spec`/`downstream_action`), ran `_maybe_wake_successor` → `disabled` or `no_pattern` with `wake:false` (see L1932–1945 `config_defaults.py` default wake disabled; wake patterns `r"\bnext …\b" | r"\bsuccessor\b" | …` never matched this prose), wrote no `successor_wake_skipped` that mattered. Net effect: open parent chain (real goal unfinished) + zero `running`/`ready` — pipeline silently dead.

2. **Silent stall that "looks alive" — two ~45–80 min running cards, zero human-readable notes.** Workers called the internal liveness tool `kanban_heartbeat` every ~60 s (claim-TTL pump into `tasks.last_heartbeat_at` + `task_runs.claim_expires`; the gateway's `GatewayKanbanWatchersMixin._kanban_notifier_watcher` 5 s poll in `gateway/kanban_watchers.py` sees only `task_events. kind IN TERMINAL_KINDS`, so those heartbeats produced no notification). No `kanban_comment` posted, hence no `task_comments` row, hence no `commented` event. The board therefore rendered "working." The existing estate watchers — `stall-sentinel.py` (NOHB = no heartbeat >3 h, QUICK/FED/etc. only for *parked* cards), `board-followthrough.py` (followed-root landings + blocked-on-you lines only), `recurring`/`lcm` health — have no running-but-silent path.

3. **Detection was human-powered.** `TERMINAL_KINDS = ("completed","blocked","gave_up","crashed","timed_out","status","archived","unblocked","block_loop_detected","review_requested","changes_requested")` (L~24 `kanban_watchers_notifier.py`) plus subscription cursors `claim_unseen_events_for_sub` (L~591 `kanban_db_notify.py`) only fire on terminals/blocks. Nothing fires for "running but has not said anything useful in a while" or "open work exists but nothing is runnable."

Reference specimen folded mid-run per operator note (SPECIMEN-14, 2026-11-21): this card (t_9106d1e1) itself ran ~75 min with heartbeats every ~60 s, zero milestone notes, first human-visible signal only at ~49 min of silence from the progress watcher. Had `stalled_suspect` at 25–30 min existed, a ping would have fired ~20 min earlier. Cheap `STILL-WORKING:`/`NOTE:` notes are the behavioral mitigation the notifier contract should encourage (see §3.6).

Cost: GPU-reserved runners idle-burn ~$4–10/h during the silence; human babysitting compounds across boards.

### 1.1  What "actual kernel" this ADR pins

*This ADR is a delta on the shipped kernel, not a greenfield proposal.* Every insertion seam names a real file and a commit-stable anchor (grep-string / symbol / line vicinity as of `~/.hermes/hermes-agent` `feature/hermes_board_layers` @ `dc42521` + estate hotfix spine; `durable-workers/v4` shares the same dispatcher/watcher split at `gateway/kanban_watchers_*`):

- **G-55 successor gate + wake** — `hermes_cli/kanban_db.py` L2685–3170 (`_gate_successor` at ~L2846, `_maybe_wake_successor`+`_maybe_wake_unguarded` at ~L2954, `_successor_matched_pattern` at ~L2881, `_successor_wake_event`/`successor_wake_skipped|fired|unroutable` events in `complete_task`'s `_maybe_wake_successor` tail at L3118). Config `hermes_cli/config_defaults.py` L1926–1932 (`kanban.successor_guard: {gate, wake, wake_patterns, fresh_window_s, routers}` + fallback orchestrator). Schema `hermes_cli/kanban_db.py` L960–1040 (`task_links`, `task_comments`, `task_events`, `task_runs`). The **spurious-wake precedent** is the same machinery: `wake_patterns` contains `r"\bsuccessor\b"` (L1938), which also matches literal strings like `wall_mid_turn_successor` (word-boundary `\b` still fires at the `_` boundary) — cited explicitly as the false-positive case this fix must *not* repeat.
- **Heartbeat/claim liveness** — `hermes_cli/kanban_db.py` L284–360 (`DEFAULT_CLAIM_TTL_SECONDS=15*60`, `DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS=60*60`, `_touch_activity` via `agent/run_agent.py:bridge_activity`, `heartbeat_claim`, `release_stale_claims` backstop at L2424 "heartbeat older than max-stale → reclaim even if PID alive").
- **Notifier delivery + subscription lifecycle** — `hermes_cli/kanban_db_notify.py` L42–480 (allowlist, `add_sub`/`remove_sub`/`claim_unseen_events_for_sub`), `gateway/kanban_watchers_notifier.py` (5 s polling loop `GatewayKanbanWatchersMixin._kanban_notifier_watcher`, `TERMINAL_KINDS`, `_WAKE_KINDS`, `diagnostic_event`, `MAX_SEND_FAILURES=12`, cursor-advancement dedup — "subscription removed only at `archived`, cursor handles dedup" comment L~34 `kanban_db_notify.py`).
- **Dispatcher / orphan / stale handling** — `hermes_cli/kanban_db_dispatch.py` + `gateway/kanban_watchers_dispatcher.py` (`_DispatcherSettings`/`_KanbanDispatcher`, stale-timeout, `reconcile_orphans`, failure counter/block-gate).
- **Progress watchers to extend not duplicate** — estate cron `~/.hermes/scripts/stall-sentinel.py` (fleet SILENT-STALL watchdog t_d681fa72; classes FED/UPSTREAM/TRIAGE/REPLANNED/QUIET/STRANDED/NOHB; `NOHB_AFTER=3*3600`; alert-only, consolidated KANBAN-OPS delivery, per-card `STALL-SENTINEL:` comment, cadence `CADENCE=(0,4h,24h)`, defer `STALL-DEFER`, `IGNORED_COMMENT_RE` skips sentinel/marker prose), `~/.hermes/scripts/board-followthrough.py` (follow-through of followed roots/chains), `tools/kanban_tools_schemas.py` + `agent/kanban_stop.py` (`kanban_schedule` → status `scheduled` + marker `SCHEDULED_UNTIL=ISO-8601`, external `fleet-wake-driver` generalizing `swco-wake-driver.py`).
- **Container proof venue** — `docs/CI-CONTAINERS-GUIDANCE.md §3`: upstream-bound work runs in `hermes-upstream:fresh` via `ci/hermes-upstream/run-branch.sh <worktree>` (throwaway `HERMES_HOME`, stub LLM, `--network none` for tests, nightly image tag+sha as receipt); estate-shaped work in `hermes-estate-twin` (`twin-test`/`twin-clone-test` with `=== TWIN-RECEIPT v1 ===`). See §5 for this ADR's replay contract.

Non-goal of the design doc itself: rolling out to live boards is a separate GO after the container proof is green (operator ruling in card body).


## 2  Decisions (ranked per operator build order)

### 2.1  A — Extend the successor-guard at completion (kernel, completion path)

**Rule:** a completion must not go quiet when it *promises* a successor but creates none. Today `G-55` only gates on the explicit declaration line `^SUCCESSOR:\\s*required\\b` (L2751 `_BODY_DECLARATION_RE`) or `metadata.{successor_spec,downstream_action}` (L2753 `_DECLARATION_KEYS`) *and nothing else* (L2828–2870 `_gate_successor`). This ADR extends the detector to three mutually reinforcing signals; any one suffices to trigger the **chain-break** path:

1. **Explicit declaration (existing).** Unchanged: `_BODY_DECLARATION_RE` / `_DECLARATION_KEYS` → `MissingSuccessorError` unless `_successor_discharged` (`verified_cards` or live child link) or valid waiver `metadata.no_successor` ≥12 chars (L2792 `_WAIVER_MIN_CHARS`).
2. **Successor-intent prose (new).** `summary`+`result`+metadata flattened text (L2854 `_successor_completion_text`) contains a *successor-intent phrase* (operator ranking: the Q8 shape "preparing handoff successor / successor next / follow-on card will…"). Matched via a **tightened** pattern list (see §3.1), distinct from `wake_patterns` so the tightening does not regress the existing `[SUCCESSOR-WAKE]` fallback that must remain governed by §3.1's deny-list.
3. **Structural chain-deadness heuristic (new, conservative).** At `complete_task` commit time: the board has open work that *would have been promotable* from this card's lineage (open parent-chain ancestor or sibling chain with unsatisfied `todo`/`ready` roots) **and** the completing card has zero linked children / created_cards. This predicate alone never fires in isolation unless the detector is configured to — operator ranked intent-prose first; the heuristic is an *additional* arm that only fires when intent prose also matched **or** when `kanban.successor_guard.structural_break_check` is explicitly on (default off for phase 1). Rationale: the structural check has the widest blast radius; shipping it as a warning-only signal first prevents alert fatigue while the intent detector hardens.

**Reaction — the board must not go quiet:**

- **Event:** `chain_broke` (new kernel event kind, emitted inside the completion txn next to `completed`; see §4 for payload). Always written even when the completion is allowed to land, so replay and metrics are deterministic.
- **Triage card auto-filed:** `triage` card titled `CHAIN-BROKE: <parent title> — promised successor, created none` with body citing the offending `summary`/`result` excerpt, the matched phrase, the board + completing `task_id`, and the `chain_broke` event id. Idempotency: `idempotency_key = f"chain-broke:{completing_task_id}"`; a retried completion does not duplicate the triage card. The triage card is *not* auto-parented beyond information — downstream decomposition is the normal triage path.
- **Gate vs warn:** phase-1 `warn` (completion lands, triage card + alert carry) for arms (2)/(3); phase-2 (operator-reviewed) may promote (2) to `reject` (raise `MissingSuccessorError` exactly like G-55) once false-positive rate is measured against the replay harness (§5). The ADR requires the **reject toggle** to be a single boolean `kanban.successor_guard.intent_gate` in `config_defaults.py` so the promotion is a config line, not a code branch.

Insertion seams (all in `hermes_cli/kanban_db.py`):
- new `INTENT_PATTERNS` + `STRUCTURAL_BREAK_PREDICATE` + `INTENT_DENYLIST` adjacent to L2751 `_BODY_DECLARATION_RE`;
- new helpers `_successor_intent_match()`, `_structural_chain_break()`, `_emit_chain_broke()` next to L2881 `_successor_matched_pattern`;
- branch inside `_gate_successor` (L2828) *or* a new post-gate checker `_check_chain_break()` called from `complete_task` just before `_maybe_wake_successor` at L3118 so explicit-G-55 rejections remain first;
- `_successor_guard_config()` extension for the three new knobs; `config_defaults.py` defaults added;
- `tools/kanban_tools.py` needs no change — `kanban_complete` already threads `summary`/`metadata`/`created_cards` to `complete_task`.

### 2.2  B — Note-stall signal for running cards (kernel/watcher hybrid)

**Problem restated with today's machinery:** `tasks.last_heartbeat_at` pumps from `kanban_heartbeat`/`_touch_activity` and feeds `release_stale_claims` reclaim at `DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS=3600` (1 h) — a *hard* failure path (reclaim → retry-budget accounting → `blocked`/`triage`). A worker that is busy-looping in application logic but still heartbeating will *never* hit that backstop. The only "human" signal is `task_comments` rows (→ `task_events kind=commented`), which no existing watcher turns into a running-card alert: `stall-sentinel.py:NOHB` explicitly checks `last_heartbeat_at is None OR stale` (>3 h), not `commented`-silence.

**Signal:** `stalled_suspect` — a kernel event (not a heartbeat claim state, not a `blocked` transition) that fires when a `running` card has had heartbeats but **no milestone note** for a configurable window. A *milestone note* is defined as any `task_comments` row whose body starts with the operator-ratified note prefix `RECHECK:` / `NOTE:` / `STILL-WORKING:` / `HANDOFF:` (see §3.6 contract) — *not* arbitrary chatter, and *not* heartbeat-only `commented` events. The implementation reads `task_comments WHERE task_id=? AND created_at >= running_since ORDER BY id DESC LIMIT 1` plus a filter on prefixes; cards with zero matching notes are treated as `age = now - started_at`.

**Window + overrides:** default 25–30 min (operator spec; ADR pins **30 min** as the shipped default, matching SPECIMEN-14's 25–30 guidance — implementers should set `kanban.note_stall.window_seconds = 1800`). Per-card override `task.meta.note_stall_window_seconds` when present; per-board override `kanban.note_stall.by_board.<slug>` for noisy boards.

**Cadence + dedup:** first `stalled_suspect` at window expiry, then once per window thereafter (so a 75 min run like SPECIMEN-14 produces pings at ~30, ~60). At most one live `stalled_suspect` per running episode; closing the run (→ `completed`/`blocked`/`reassigned`) clears the episode — a respawned run starts a new silence window. The dispatcher is not the right owner (it already owns reclaim/orphan); the new **note-stall watcher** runs alongside it.

**Where it lives:** new module `hermes_cli/kanban_note_stall.py` (pure DB helpers, testable without gateway) + a periodic check called from the gateway's dispatcher tick (`gateway/kanban_watchers_dispatcher.py:_KanbanDispatcher.tick_once`, right after stale/orphan accounting, before `auto_decompose_tick`) *and* as a fallback 30 s loop inside `GatewayKanbanWatchersMixin._kanban_dispatcher_watcher` when gateway is alive. In `hermes-upstream:fresh` there is no gateway — replay tests instead invoke the helper directly against a temp `kanban.db` (see §5). This satisfies the card's "kernel/watcher" framing: the helpers are kernel-adjacent and unit-tested on a DB; the watcher is just the scheduler.

No `kanban_comment` schema change: the prefix convention reuses `body` text search (`WHERE body LIKE 'NOTE:%' OR body LIKE 'RECHECK:%' …`) plus a small in-Python prefix match, exactly like `stall-sentinel.py`'s `IGNORED_COMMENT_RE`/`SUMMARY_RE` splits. When the project adopts structured milestones later, this predicate upgrades to a column without changing the event contract (§4).

### 2.3  C — Board-liveness invariant (daemon-level, per board)

**Invariant:** if a board has *open goals* and *nothing is running or promotable*, the board is dead-quiet and an operator/butler must be told — "the why-did-nobody-notice fix."

**Definition (DB-predicate precise):**

- `open_goals = EXISTS tasks WHERE status IN ('todo','ready','running','blocked','triage','scheduled')` (i.e., not `done`/`archived` on that `board`'s DB). In practice the interesting case is `todo`/`ready` roots with unfinished parent chains; `scheduled` is counted as open but **excluded** from dead-quiet firing (its `SCHEDULED_UNTIL` future is legitimate quiet — fleet-wake-driver owns it).
- `running_any = EXISTS status='running'`
- `promotable_any = EXISTS status IN ('ready') OR (status='todo' AND every parent is done/archived AND assignee is live)`
  - "promotable" reuses `kanban_db_dispatch._promotable`/`has_spawnable_ready` semantics already in `_DispatcherSettings` — nothing new to invent.
- Dead-quiet fires iff `open_goals AND NOT running_any AND NOT promotable_any`.

This predicate distinguishes the two failure shapes in the card body:

- **DEAD** — `open_goals` are `todo`/`ready` but nothing is runnable (e.g., every open card is `blocked`/`triage` with no operator path, or `todo` whose parents are not done because the parent chain broke). This is today on gargantua01: Q8 completed, children absent, sibling `todo` parents gating on a now-dead branch → `promotable_any=false`.
- **QUIET-BY-DESIGN** — all open goals are `scheduled` (legit future wake), or `blocked` specifically `block_kind='needs_input'` (operator decision owns the clock), or `todo` gating on a legit upstream `running`/`blocked` card on another board. The board *should* be quiet. These are **excluded**: `scheduled` alone never fires; `blocked/needs_input`-only boards emit at most one `quiet_by_design` digest (see below), never `DEAD`.

**Labels + cadence:**

- `board_dead_quiet` (DEAD) — immediate on detection, then daily while still dead (mirrors `stall-sentinel.CADENCE` daily tail).
- `board_quiet_by_design` (informational, same liveness daemon) — fires once per open-goal episode, labeled distinctly so the notifier can render it quieter (e.g., "Board idle — all open cards wait on you/schedule, no action needed"). Re-arms only when the set of open `todo`/`ready`/`blocked` ids changes.

**Where it lives:** new helper `hermes_cli/kanban_board_liveness.py:check_board_liveness(conn, board, now)` returning `(state, detail)` enum `LIVENESS_DEAD|QUIET_BY_DESIGN|HEALTHY`, plus an event writer `_emit_board_liveness(conn, board, state)`. Scheduled as a **periodic per-board check** from the dispatcher liveness beat — every `kanban.board_liveness.interval_seconds` (default 5 min; test knob 10 s via config). Notifier is not the place: notifier is event-driven per-subscription; liveness is a board-level predicate with no subscribing task.

### 2.4  D — Notification coverage (parity with completions)

Every new signal must route like completions do today, otherwise the fix is invisible:

- **Subscription + cursor parity:** `stalled_suspect`, `chain_broke`, `board_dead_quiet`/`board_quiet_by_design` are all `task_events` rows on a real `task_id` (`tasks` row always exists; board-level events anchor to a synthetic per-board sentinel task `board:<slug>:liveness` that the migration creates once — see §4). `hermes_cli/kanban_db_notify.py:claim_unseen_events_for_sub` therefore sees them with zero protocol change: any subscription whose allowlist includes the new kinds (and every existing "all kinds" subscription already will) receives them. `hermes_cli/kanban_db_notify.py:TERMINAL_KINDS`-adjacent allowlist grows to include them in `gateway/kanban_watchers_notifier.py:TERMINAL_KINDS` and `_WAKE_KINDS` (for board-dead wake) — one-line additions, not a new delivery path.
- **Delivery:** existing `GatewayKanbanWatchersMixin._kanban_notifier_watcher` 5 s poll renders them; `MAX_SEND_FAILURES=12` dead-chat rule unchanged; cursor-advance dedup unchanged ("removed only at archived").
- **Routing:** `diagnostic_event` predicate in `gateway/kanban_watchers_notifier.py` returns False for the new kinds (they are *primary* attention, not diagnostic), so they are never suppressed by diagnostic filtering.

The ADR therefore delivers the card's "notification coverage: like completions do today" without inventing a notifier — it delivers by extending `task_events.kind` and `TERMINAL_KINDS`, and the existing gateway loop does the rest. Estate cron `board-followthrough.py` is explicitly **not** the delivery for these signals: follow-through reports *landings + blocked-on-you*; conflating running-stall / dead-quiet into the same feed would bury them.

## 3  Semantics + false-positive guardrails

### 3.1  "Successor intent" — tight matching (the wall_mid_turn_successor lesson)

The shipped `wake_patterns` entry `r"\\bsuccessor\\b"` (L1938) matches inside `wall_mid_turn_successor` because `_` is a word-boundary in `re` — `\\b` fires between `n` and `_`. That is the precedent the operator cites; the successor-intent list must *not* repeat it.

Ship these rules (unit-tested against the corpus at `tests/hermes_cli/test_kanban_successor_guard.py` + a small added fixture with `wall_*_successor` + sample-data strings):

1. **Phrase-level, not token-level.** Valid triggers are *multi-token phrases*, not single words:
   - `preparing handoff`
   - `handoff successor` / `successor handoff`
   - `follow[- ]?on card (will|to|needs?)`
   - `next step (is|will|needs?)`
   - `filed as t_[0-9a-f]{8}` with successor-proximal language in the same `summary`/`result` block (detects "continuing in t_…" prose).
2. **Case-insensitive, punctuation-normalized, bounded span.** Each phrase regex is `(?i)` with `[^\\w]*` punctuation bridges limited to ≤20 non-letters before the second token, and requires a verb (`will|to|needs|should|preparing`) *in order*. This rejects standalone noun mentions ("the successor field") and sample-data strings ("dataset successor_labels.csv") where `successor` appears without a handoff verb in the same clause.
3. **Deny-list for known data-word substrings** (cheap, no model needed): any match whose full matched span is contained in a `\\w*[_/](?:wall|sample|dataset|artifact|snapshot|fixture)[_\\w]*successor` token is dropped. Additions to the list are a code change with a regression fixture, never config.
4. **Scope:** match against `summary` + `result` + `metadata` flattened text only — never against `title`/`body`/`task_links` — because chain-break is a *completion-time* promise: what the worker claimed would happen next. Title/body matches stay in the existing `[SUCCESSOR-WAKE]` path (L1932 wake) where they belong.
5. **One intent pattern is enough** for the event; the payload records `matched_phrase` (first hit), not a count — keeps the triage card body readable and the dedup key stable (see §4).

Why keyword matching at all (false-positive risk)? Operator ranking answers it: every incident converts unattended hours into babysitting, and the alternative — "only explicit `SUCCESSOR: required`" — already missed Q8. The mitigation is not to avoid matching but to **tighten the patterns + deny-list real specimen strings**, log every `chain_broke_skipped_no_phrase` / `chain_broke_denied_by_denylist` diagnostic event so the daily feed shows how much signal is filtered, and keep phase-1 as `warn` so false trips produce a triage *card* (tolerable) not a rejected completion (intolerable until measured).

### 3.2  Per-card / per-board overrides (config knobs that matter)

Config lives in `config.yaml` (`kanban.successor_guard`, `kanban.note_stall`, `kanban.board_liveness`) — the existing singleton pattern in `config_defaults.py` plus the shared-root read in `_successor_guard_config()` (L2762 `read_config_doc(kanban_home()/config.yaml)`). New defaults:

```yaml
kanban:
  successor_guard:
    gate: true
    wake: false            # unchanged — wake stays opt-in; chain-break has its own triage-card path
    intent_gate: false     # phase-1 warn, phase-2 reject (see §2.1)
    structural_break_check: false
    intent_patterns: [...] # §3.1 phrase list
    intent_denylist: [...] # §3.1 deny tokens
    fresh_window_s: 21600  # reused for wake
    routers: {}
  note_stall:
    enabled: true
    window_seconds: 1800   # 30 min shipped default; operator guidance 25–30
    by_board: {}           # {slug: seconds} overrides
    # per-task override is not config — it is task.metadata.note_stall_window_seconds (see §3.2)
  board_liveness:
    enabled: true
    interval_seconds: 300  # 5 min tick; test knob 10 s in harness (§5)
```

Per-card override path: `task.metadata.note_stall_window_seconds` (integer seconds, >0) is read if present; longer than default windows are honored for explicitly long-running tasks. This is the "configurable window, not a real 80 min" lever the replay contract relies on — the harness sets it to 5–15 s.

### 3.3  `stalled_suspect` vs `stall-sentinel` — no double-alert

These are disjoint by construction:

- `stall-sentinel:NOHB` fires at 3 h with no heartbeat. `stalled_suspect` fires at 30 min with heartbeats-but-no-note. A single running episode can therefore emit at most one of them first (timescale separation).
- When `stalled_suspect` has fired, `stall-sentinel`'s STRANDED/FED/etc. classes are mutually exclusive: they only scan `status IN ('blocked','triage','todo','ready')` per `load_board()` — a `running` card is ineligible. The only overlap would be NOHB after 30-min note-silence + 3 h heartbeat-silence on the same card; the dedup rule is: **`stalled_suspect` receipt comment prefix is `NOTE-STALL:` (new), `stall-sentinel` prefix is `STALL-SENTINEL:` — the gateway's `IGNORED_COMMENT_RE` (which already skips `STALL-SENTINEL|STALL-DEFER|…`) is extended to skip `NOTE-STALL:` for sentinel's "new-input" scan, and `note_stall.py` skips cards that already have a `STALL-SENTINEL:` comment in the current running episode.** In plain terms: whichever watchdog fires first wins the episode; the other's per-episode comment is suppressed. Both still log, just not double-ping the operator.

The estate cron file gets one non-semantic edit: `IGNORED_COMMENT_RE` at `~/.hermes/scripts/stall-sentinel.py:L~35` extends by `|NOTE-STALL`.

### 3.4  Board-liveness must not fire on legitimately idle boards

"Open work but nothing runnable" has two meanings; the alert must tell them apart:

- **"Nothing runnable exists" (DEAD)** — `todo` cards exist but none are promotable (every parent is itself unfinished and not running; or there are no children where the design expected them; or `ready` set empty and no `todo` can become `ready`). Example: gargantua01 today.
- **"Everything waits on a human" (QUIET-BY-DESIGN)** — every open card is `blocked/needs_input`, `blocked/capability` with a `DECISION-SUMMARY`, `scheduled` until a future `SCHEDULED_UNTIL`, or `triage` awaiting a person. The board *should* be silent. Alerting daily on such boards is alert fatigue.

The predicate in §2.3 encodes this; the rendering contract in §4 sends the two states with distinct titles ("Board dead-quiet" vs "Board idle — waiting on decisions/schedule") and the board-liveness event payload carries `state=DEAD|QUIET_BY_DESIGN` so `diagnostic_event` and board-specific muting can key on it without parsing prose.

### 3.5  Schedule / fleet-wake integration

`kanban_schedule` tasks (`status='scheduled'` + `task_events` marker `SCHEDULED_UNTIL=<ISO-8601>`, driven by `~/.hermes/scripts/fleet-wake-driver.py` 60 s external loop + `GatewayKanbanWatchersMixin` in-gateway mirror) are intentionally *outside* dead-quiet: the liveness predicate treats `scheduled` as open-but-not-counted. The wake driver's job (`kanban_schedule` does not create a timer; the external watcher does — `tools/kanban_tools_schemas.py: KANBAN_SCHEDULE_SCHEMA` note) is unaffected. Likewise `board-followthrough.py` never sees these events — it only follows blocked/landing.

### 3.6  Cheap still-working notes — the behavioral mitigation

The new signals make silence expensive (an alert *will* fire). To make that sustainable, the notifier contract encourages a cheap mitigation: any running worker can suppress its own next `stalled_suspect` window by posting a two-line progress note:

```
NOTE: still working — <what I just did>
NEXT: <what I am about to do>
```

or equivalently `STILL-WORKING:` / `NOTE:` / `HANDOFF:` / `RECHECK:` prefixes. The running card `kanban_comment` tool already exists (`tools/kanban_tools.py:317 kanban_comment`, `tools/kanban_tools_schemas.py: KANBAN_COMMENT_SCHEMA`); this ADR adds **no new tool** — it just standardizes the prefix so the `note_stall` predicate is deterministic and SPECIMEN-14's mitigation ("cheap still-working notes") is literally what the dispatcher tells the operator ("Post a NOTE: to silence the next window").

Long-card session discipline (G-31) is strengthened accordingly: implementer cards t_d5708f00/t_68d5b5a4 must require at least one such note per ~20 min of wall-clock (the review gates in t_b35e2e73/t_d58fb0d3 enforce this by checking comment timestamps in the running window).


## 4  Event + notification contract (B/D parity)

All new signals are `task_events` rows — the existing subscription/cursor machinery carries them with zero gateway change beyond allowlist bumps.

### 4.1  Event kinds, anchoring, idempotency

| Kind (payload `kind`) | Anchored to | Emitted by | Idempotency | Dedup window |
|---|---|---|---|---|
| `chain_broke` | the **completing** `task_id` (the card whose `completed` just landed) | `complete_task` txn, adjacent to `completed` (never in a separate txn — atomic with the completion so replay can re-derive) | `idempotency_key = chain-broke:{task_id}` on the triage card; the event itself is unique per completion (one completion → at most one `chain_broke`) | none needed — completion is once |
| `stalled_suspect` | the **running** `task_id` | `kanban_note_stall.check_running_notes` periodic (see §4.2) | per-run window: `dedup_key = stalled:{task_id}:{run_id}:{window_epoch}` where `window_epoch = floor((now - running_since)/window_s)` — one event per window per run | `note_stall.window_seconds` (default 1800) |
| `board_dead_quiet` | synthetic per-board task `board:<slug>:liveness` (migration `kanban_db: ensure_sentinel_tasks()` creates it `status=todo, assignee=null` once; never dispatched) | `kanban_board_liveness.check_board_liveness` periodic | per-board: `dedup_key = board-dead:{slug}:{day_epoch}` (once per UTC day while DEAD — mirrors `stall-sentinel.CADENCE` daily tail) | `board_liveness.interval_seconds` |
| `board_quiet_by_design` | same sentinel task | same | `dedup_key = board-quiet:{slug}:{open_goal_fingerprint}` (once per fingerprint while still quiet) | open-set fingerprint change re-arms |

Payloads (JSON in `task_events.payload`, all `created_at = now` unix seconds):

```jsonc
// chain_broke
{"matched_phrase": "preparing handoff successor", "summary_excerpt": "preparing handoff successor — …", "completing_task": "t_9106d1e1", "board": "gargantua01", "structural": false, "triage_task": "t_<new>"}
// stalled_suspect
{"running_since": 1732130000, "window_s": 1800, "window_epoch": 1, "last_note_at": null, "specimen": "SPECIMEN-14 shape — 75m no milestone notes, first ping at ~30m"}
// board dead / quiet
{"state": "DEAD", "open_goals": 7, "running": 0, "ready": 0, "promotable_todo": 0, "blocked": 6, "triage": 1, "reason": "open goals exist, nothing runnable — chain likely broke at <parent>"}
```

Sentinel-task migration: a single `CREATE TABLE IF NOT EXISTS` guard + `INSERT OR IGNORE INTO tasks(id,title,status,board) VALUES('board:'||board||':liveness', …)` per-board at startup. No dashboard change — sentinel tasks are filtered from list views (`WHERE id NOT LIKE 'board:%:liveness'` in `kanban_db.list_tasks` when `include_system=false` — default list stays user-visible only).

### 4.2  Delivery (like completions today)

Two allowlist bumps ship with this ADR — the only notifier-touching code change:

1. `gateway/kanban_watchers_notifier.py: TERMINAL_KINDS += ("chain_broke","stalled_suspect","board_dead_quiet","board_quiet_by_design")`.
2. `gateway/kanban_watchers_notifier.py: _WAKE_KINDS += ("board_dead_quiet",)` (dead-quiet is actionable for wake routing; `board_quiet_by_design` is not).

Then, out of the box:

- Any subscription (`hermes_cli/kanban_db_notify.py: add_notify_sub(board=…, task_id, platform, chat_id)`) whose `kinds` allowlist is `None`/empty (the default "all terminal kinds" today) receives the new events. No per-kind adoption step.
- Per-subscription cursor `claim_unseen_events_for_sub(conn, sub_id, kinds_filter) → [Event]` already advances atomically on claim (L380 `kanban_db_notify.py`); dedup is therefore per-subscription, not global — the same semantics that already protect `completed` from double-delivery across two gateway restarts.
- Diagnostic suppression (L48 `diagnostic_event`) explicitly returns False for `kind in ("stalled_suspect","chain_broke","board_dead_quiet")` — these are primary, not diagnostic, so they are never filtered on the subscriber side.
- Rendering lives in `gateway/kanban_watchers_notifier.py: _render_event(event)` next to `completed`/`blocked` — a short human line ("Board dead-quiet on gargantua01 — 7 open cards, none runnable (since …)" / "Chain broke here: t_… promised a successor and created none — triaged as t_…") with a deep-link to the triage card. No new Telegram/estate surface.

Gateway-disabled environments (like `hermes-upstream:fresh`) still produce the `task_events` rows — they just do not push Telegram. The replay harness in §5 reads `task_events` directly and is therefore gateway-independent, which is exactly what the card requires ("standing container proof only").


## 5  Acceptance — replay contract (deterministic, in hermes-upstream:fresh)

The card mandates: "today's two gargantua01 incidents replay deterministically in a software-co standing container on a synthetic board." The proof is a pytest suite run inside `hermes-upstream:fresh` via `ci/hermes-upstream/run-branch.sh <worktree>`; the worktree is the build branch's own checkout (`git worktree add` per existing pattern). No real 80 min wait — the configurable window, not wall-clock, is the lever.

### 5.1  Harness shape

Test DB: `tempfile.mkdtemp()` + `HERMES_KANBAN_DB=<tmp>/kanban.db` (the hermes test isolation pattern in `tests/conftest.py: tmp_path_factory + monkeypatch.setenv`). Isolation is per-test, no fleet DB touched. Shared-root reads are stubbed via `monkeypatch.setattr(kanban_db, "_successor_guard_config", lambda: {...})` per `tests/hermes_cli/test_kanban_successor_guard.py: _wake_cfg` pattern — no real `config.yaml` needed.

Two synthetic boards via `kanban_db.create_board` (or simply two distinct `HERMES_KANBAN_DB` files): `board-a` for chain-break, `board-b` for note-stall + liveness. Tests use real kernel functions only: `create_task`, `claim_task`, `complete_task`, `add_comment`, `list_events`; no direct SQL beyond `SELECT` assertions (so schema shifts do not silently pass).

Image tag+sha pin: `ci/hermes-upstream/status.json` at test time (the standard `CI-CONTAINERS-GUIDANCE.md` receipt) is logged in test stdout by the helper `tests/helpers/container_receipt.py` (existing helper; if absent, the test prints `open("ci/hermes-upstream/status.json").read()` verbatim).

### 5.2  Scenario 1 — chain-break (BUILD-A)

```
given  parent t_parent = create_task(board-a, title="Q8 bisect", body="…", assignee="dev")
       run = claim_task(t_parent)             # -> running
when   complete_task(t_parent, summary="preparing handoff successor — filing t_next tomorrow",
                     result="…", metadata={}, created_cards=[])
then   row = one_event(t_parent, kind="chain_broke")   # exists, payload.matched_phrase == "preparing handoff successor"
       triage = get_task(row.payload.triage_task)       # status=triage, title startswith "CHAIN-BROKE:"
       assert "preparing handoff successor" in triage.body
       assert one_event(t_parent, kind="completed") still exists (completion not rejected in phase-1)
```

Negative arm (same harness, same board): `summary="fixed wall sample wall_mid_turn_successor label"` + `created_cards=[]` → **no** `chain_broke`, no triage card (proves the deny-list/tight phrasing fix for the spurious precedent).

Regression arm: explicit `SUCCESSOR: required` in body with `created_cards=[]` still raises `MissingSuccessorError` and emits `completion_blocked_missing_successor` (existing guard unchanged).

### 5.3  Scenario 2 — silent stall + liveness (BUILD-BC)

```
given  t = create_task(board-b, title="long job", assignee="dev")
       claim_task(t)  # running_since = now
       # no comments posted
when   advance clock by note_stall.window_seconds (harness override 8 s via task.metadata.note_stall_window_seconds=8)
       call kanban_note_stall.check_running_notes(conn, board-b, now)  # helper under test
then   one_event(t, kind="stalled_suspect") exists, window_epoch==1
when   post NOTE: "NOTE: still working — step 2 of 3" via add_comment(t, "NOTE: …")
       advance one more window
       call check_running_notes again
then   zero new stalled_suspect events (notes reset the window)

# board liveness on same DB, after stalling card is completed without successors:
when   complete_task(t, summary="done", result="done")   # t -> done, no open running/ready
       # but create sibling open goals with broken parents:
       p = create_task(board-b, title="sibling root", assignee="dev")  # todo
       link parent chain so p is blocked on a now-done-but-childless parent
       call kanban_board_liveness.check_board_liveness(conn, board-b, now)
then   one_event("board:board-b:liveness", kind="board_dead_quiet") exists, state=DEAD

# quiet-by-design arm:
when   clear all todo/ready; only (blocked/needs_input + scheduled + future-SCHEDULED_UNTIL) open
       call check_board_liveness
then   kind == "board_quiet_by_design" (or none if suppressed by dedup), never "board_dead_quiet"
```

The "80-min note-less heartbeat run → stalled_suspect by ~minute 30" requirement is met by the 30-min default window (1800 s) — the harness compresses it to 8 s via the per-card override so CI never sleeps 80 min; the test asserts stalled_suspect fires within `window ± 10%` (wall-clock jitter budget) and that normal-notes runs produce exactly zero events.

### 5.4  Evidence the completion metadata must carry

Per G-02 board discipline, each build card's `kanban_complete(metadata=…)` pastes:

- branch + commit `git rev-parse HEAD` on that worktree
- `ci/hermes-upstream/status.json` tag+sha pin
- pytest tail (`-q` with pass/fail counts) for its own test file + the replay file
- per-scenario one-line verdict (PASS/FAIL + event id + triage id) — so the reviewer can `kanban_show` the proof without re-running.


## 6  Implementation DAG — seams for the two build lanes

This ADR ships as one design but lands in two parallel build cards (operator-ranked build order; both depend on this ADR). Non-goals common to both: no live-estate rollout, no Telegram surface, no `stall-sentinel.py`/`board-followthrough.py` rewrites beyond the one-line `IGNORED_COMMENT_RE` edit in §3.3.

### 6.1  BUILD-A — successor-guard extension (card t_d5708f00 → sw-dev-a)

**Seams (all `hermes_cli/kanban_db.py` unless noted):**

- `hermes_cli/kanban_db.py: _BODY_DECLARATION_RE` (~L2751, adjacent): add `INTENT_PATTERNS: list[re.Pattern]` + `INTENT_DENYLIST: list[re.Pattern]` + `_successor_intent_match(text)->Optional[str]`.
- `hermes_cli/kanban_db.py: _successor_matched_pattern()` (~L2881): leave intact (wake path); chain-break uses its own `_successor_intent_match()` so wake's false-positive tuning is independent.
- `hermes_cli/kanban_db.py: _gate_successor()` (~L2828) or new `_check_chain_break(conn, task_id, summary, result, metadata, verified_cards)->Optional[str]`: called from `complete_task()` (L~3070–3120, right before `if verified_cards or _has_live_child` / `_maybe_wake_successor`) — reads intent prose + structural predicate, emits `chain_broke` + triage card via `create_task(..., status="triage", idempotency_key=f"chain-broke:{task_id}", parents=[task_id])`. Must not raise in phase-1 (see §2.1 `intent_gate`).
- `hermes_cli/config_defaults.py: kanban.successor_guard` (~L1926): add `intent_gate: False`, `structural_break_check: False` (phase-1 warn), plus `intent_patterns`/`intent_denylist` defaults; document in `DEFAULT_CONFIG` dict.
- `gateway/kanban_watchers_notifier.py: TERMINAL_KINDS / _WAKE_KINDS` (~L24) + `gateway/kanban_watchers_notifier.py:_render_event` adjacent to `completed`/`blocked` branches: chain-break rendering.
- `hermes_cli/kanban_db.py: ensure_sentinel_tasks()` (new, called at startup via `hermes_cli/kanban_db_connect.py: connect()` migration hook) — creates `board:<slug>:liveness` sentinel rows; gated by `CREATE TABLE IF NOT EXISTS` so existing DBs auto-migrate.
- Tests: extend `tests/hermes_cli/test_kanban_successor_guard.py` (positive Q8 prose → `chain_broke` + triage, wall_mid_turn_successor → none, explicit `SUCCESSOR: required` regression), plus replay case in `tests/hermes_cli/test_kanban_replay_chainbreak.py` (new file, §5.2).

**Out:** BUILD-A must not touch `kanban_note_stall` / `kanban_board_liveness`.

### 6.2  BUILD-BC — note-stall + board liveness (card t_68d5b5a4 → sw-dev-b)

**Seams:**

- New `hermes_cli/kanban_note_stall.py`: `NOTE_PREFIXES = ("NOTE:","STILL-WORKING:","HANDOFF:","RECHECK:")`, `check_running_notes(conn, board, now, *, window_s, get_config=_note_stall_config)` — scans `tasks WHERE status='running'` + latest `task_comments` per task, emits `stalled_suspect` with window-epoch dedup, returns emitted ids. Pure-DB, no gateway import.
- New `hermes_cli/kanban_board_liveness.py`: `check_board_liveness(conn, board, now)->LivenessResult`, `_emit_board_liveness(...)`, helpers `open_goals_exist` / `promotable_any` reusing `kanban_db_dispatch.has_spawnable_ready` / `_promotable` logic via injection (pass in predicate to avoid circular import). Sentinel tasks via shared `ensure_sentinel_tasks()`.
- `hermes_cli/config_defaults.py: kanban.note_stall / kanban.board_liveness` (~L1926 sibling blocks): defaults per §3.2 + §2.3.
- `gateway/kanban_watchers_dispatcher.py: _KanbanDispatcher.tick_once()` (~L130 `GatewayKanbanWatchersMixin` equivalent) + `gateway/kanban_watchers.py: GatewayKanbanWatchersMixin._kanban_dispatcher_watcher` (the 60 s dispatcher liveness beat): call `check_running_notes` and `check_board_liveness` once per tick, guarded by `config.kanban.{note_stall,board_liveness}.enabled`. In `hermes-upstream:fresh` no gateway runs — the periodic caller is simply never started; tests drive the helpers directly.
- `~/.hermes/scripts/stall-sentinel.py: IGNORED_COMMENT_RE` (~L35): add `|NOTE-STALL` (one-line cron-side dedup; see §3.3).
- `gateway/kanban_watchers_notifier.py: TERMINAL_KINDS += ("stalled_suspect","board_dead_quiet","board_quiet_by_design")`, `_WAKE_KINDS += ("board_dead_quiet",)` + rendering branches for `stalled_suspect` ("Running 30 min with no progress note — post a NOTE: …") and `board_dead_quiet` ("Board dead-quiet on … — N open cards, none runnable (chain likely broke at …)") — same `_render_event` switch as completions.
- Tests: `tests/hermes_cli/test_kanban_note_stall.py` + `tests/hermes_cli/test_kanban_board_liveness.py` (unit on temp DBs), plus combined replay in `tests/hermes_cli/test_kanban_replay_liveness.py` (§5.3). All tests use the per-card `metadata.note_stall_window_seconds` short window; none sleep 80 min.

**Out:** BUILD-BC must not widen `successor_guard.wake_patterns` or duplicate `stall-sentinel.py`'s parked-card taxonomy.

### 6.3  Ordering + shared contract

Both lanes branch from the same `durable-workers/v4` tip (current branch HEAD) after this ADR lands there; either can land first. The shared contracts that prevent merge pain are frozen here:

- Event kinds / payload keys / idempotency shapes in §4.1.
- `task_events.kind` is append-only: new rows carry new kinds, never mutate old rows.
- Notifier allowlist edits are additive (`+= tuple(new_kinds)` — never rewrite `TERMINAL_KINDS` wholesale).
- Sentinel tasks creation is idempotent (`INSERT OR IGNORE`), so both lanes can call `ensure_sentinel_tasks()`.

## 7  Alternatives rejected (and why)

1. **Only tighten wake_patterns and turn wake:true on.** Rejected: wake already missed Q8 (gated on wake_require_idle_queue + fresh_window_s 21600), and a wake routing card does not fix operator discovery. Operator asked for a guard with a triage card.
2. **Per-card last_note_at column + write-time trigger.** Rejected: schema-migration + write-path coupling. Prefix scanning on task_comments gives the same semantics with pure reads and matches sentinel's proven scan pattern.
3. **Use only the dispatcher sweep for stall detection.** Rejected: dispatcher tick is per-board and gateway-optional; fresh-container replay needs a helper invokable without a dispatcher tick. Kernel helpers (kanban_note_stall.py) with an optional periodic caller keep both venues simple.
4. **Broad token regex for successor intent.** Rejected: it is the wall_mid_turn_successor false-positive source. Phrase-level patterns + deny-list + summary/result scoping required.
5. **Board-liveness as external cron.** Rejected: liveness needs promotable_any (dispatcher predicates); a cron would reimplement them and diverge. Keep it in the dispatcher liveness beat.

## 8  Consequences

- Boards no longer go dead-quiet without an event: chain-break to chain_broke + triage card, running silence to stalled_suspect at ~30 min (SPECIMEN-14 class), open-but-not-runnable boards to board_dead_quiet / board_quiet_by_design.
- Operator burden shifts from asking what is waiting to reading the alert and clicking the triage card or posting a NOTE. GPU idle burn window shrinks from 49-80 min to 25-30 min (configurable faster per board via by_board).
- False-positive budget is spent on chain-break triage cards (warn-only in phase 1, tolerable) while note-stall is bounded by one ping per window per run.
- stall-sentinel.py stays authority for parked-card stalls; new watchers own running + board predicates. One-line sentinel edit in 3.3 is the only cron overlap.
- Every new signal is an auditable task_events row with per-subscription cursor delivery. Replay at any timestamp is SELECT plus re-check.

Non-goal carried forward: live-board rollout is a separate operator-GO after the container proof (this doc ships on durable-workers/v4, proven in hermes-upstream:fresh per section 5).

## 9  Falsification — when to revisit

- False-positive rate on chain-break exceeds tolerance (more than about 1 spurious CHAIN-BROKE card per board per week). Action: narrow INTENT_PATTERNS, grow deny-list with new specimen, or keep intent_gate warn-only.
- stalled_suspect cadence annoys (noisy boards with legit 45-min CPU phases trip every window). Action: raise kanban.note_stall.window_seconds per board via by_board, or adopt structured milestones.
- Board-liveness misclassifies scheduled or decision-blocked boards as DEAD despite 3.4 predicates. Action: fix the predicate, not the cadence.
- Notifier parity breaks (new kinds filtered as diagnostic or lost across restarts). Action: fix diagnostic_event and claim_unseen_events_for_sub handling.
- Container proof shows heartbeat vs claim coupling drift (dispatcher reclaim window changes invalidate the 3h vs 30 min separation in 3.3). Action: revisit thresholds together.

---
Appendix — seam index (grep-stable anchors): hermes_cli/kanban_db.py: _gate_successor, _maybe_wake_successor, _successor_matched_pattern, _BODY_DECLARATION_RE, complete_task, SuccessorGuardConfig; hermes_cli/config_defaults.py: successor_guard; hermes_cli/kanban_db_notify.py: TERMINAL_KINDS, claim_unseen_events_for_sub; gateway/kanban_watchers_notifier.py: TERMINAL_KINDS, _WAKE_KINDS, diagnostic_event, _kanban_notifier_watcher; gateway/kanban_watchers_dispatcher.py: _KanbanDispatcher, tick_once, _DispatcherSettings; tools/kanban_tools.py: kanban_comment, kanban_heartbeat, kanban_schedule; stall-sentinel.py: NOHB_AFTER, IGNORED_COMMENT_RE, FED, UPSTREAM, TRIAGE, STRANDED, NOHB; board-followthrough.py: BOARD-FOLLOWTHROUGH-DESIGN.

