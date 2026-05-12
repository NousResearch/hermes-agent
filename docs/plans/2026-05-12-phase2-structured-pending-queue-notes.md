# Phase 2 — Structured Pending Input Queue: implementation notes

> Companion to `docs/plans/2026-05-12-claude-phase2-task-packet.md`. Written by the
> Phase 2 implementation worker for the Hermes controller's review. No commits made.

## What shipped

- **`agent/pending_turn_queue.py`** — a leaf module (stdlib-only imports, no
  global state) with:
  - `PendingTurnItem` — a small dataclass for one pending input: `source`,
    `kind` (`text|command|media|attachment|control`), `text`, `media_refs`,
    `media_types`, `boundary` (`coalesce|hard|caption|command`), `session_key`,
    `reply_to`, `task_hint`, `origin_busy`, `created_at`, `id`, `raw`. Every
    field except `raw` is JSON-serializable; `to_dict()` drops `raw` and
    `from_dict()` rebuilds. `is_coalescible_text()` is the single boundary
    predicate.
  - `PendingTurnQueue` — an ordered `deque`-backed container: `append`,
    `appendleft`, `extend`, `clear`, `peek`, `snapshot`, `pop`, `__len__`,
    `__bool__`, `__iter__` (non-consuming), and
    `drain_coalescible_text_until_boundary(*, origin_busy=None)`.
  - Conversion helpers: `from_legacy_cli_payload` / `maybe_to_legacy_cli_payload`
    (lossless round-trip of `str`, `(caption, [paths])`, and the
    `(INTEGRATED_BUSY_PAYLOAD, text)` tag), `from_gateway_event` (duck-typed
    one-way bridge from a gateway `MessageEvent`), `looks_like_slash_command`
    (faithful mirror of `cli._looks_like_slash_command`), `legacy_cli_payload_is_coalescible_text`,
    `coalesced_text`, and the `INTEGRATED_BUSY_PAYLOAD` sentinel + its
    `make_/is_/unwrap_` helpers.
- **`cli.py`** — now imports the `INTEGRATED_BUSY_PAYLOAD` sentinel from the new
  module (single source of truth) and re-implements `_coalesce_pending_busy_queue`
  and `_coalesce_pending_integrated_busy_queue` as thin shims that build a
  `PendingTurnQueue` from the drained legacy payloads, call
  `drain_coalescible_text_until_boundary(...)`, and re-queue the tail in order.
  External behavior is unchanged — only the `queue.Queue` plumbing stays in
  `cli.py`; the "what is a mergeable text fragment / where is a boundary" rule
  now lives in the structured module. `_prepare_pending_input_for_turn` and the
  `_make_/_is_/_unwrap_integrated_busy_payload` / `_busy_payload_for_mode` /
  `_format_integrated_busy_input` helpers are untouched.
- **`tests/agent/test_pending_turn_queue.py`** — focused unit tests for the new
  module (item shape + boundary rules, queue ops + drain, legacy CLI round-trip,
  gateway event conversion, JSON serializability, sentinel helpers). It includes
  regression coverage for uncopyable `raw` payloads and raw-string gateway duck
  typing.

## Why this shape (purpose-fit, not a rewrite)

Phase 2 is the *input substrate* for future focused-agent orchestration ("Ralph"
is only the conceptual name for that future focused worker/task unit — not built
here). The structure deliberately preserves, per pending unit:

- *what the user sent* (`text` / `media_refs` / `media_types` / `raw`);
- *what kind of input it is* (`kind`) and *how it relates to its neighbours*
  (`boundary`) — so a command, a media attachment, a caption, or a hard "new
  topic" wall is never silently flattened into a coalesced text run;
- *provenance* (`source`, `session_key`, `created_at`, `reply_to`,
  `origin_busy`) — enough for a later orchestrator to decide "append to current
  focus", "steer the active worker", "start a new focused task", or "ask the
  user", without that policy being baked in here;
- *order* — `PendingTurnQueue` is an ordered queue, not a single replaceable
  slot, and `drain_coalescible_text_until_boundary` only consumes the leading
  run of mergeable text.

Boring, explicit, serializable shapes were preferred over clever abstractions;
compatibility wrappers (`from_/maybe_to_legacy_cli_payload`) over call-site
churn. The `raw` passthrough keeps every legacy payload reconstructable, so the
CLI rewrite is provably behavior-preserving (the existing
`tests/cli/test_busy_queue_coalescing.py` / `test_busy_input_mode_command.py`
suite is the contract).

## What was intentionally NOT built (out of scope for Phase 2)

- No task registry (Phase 3), no detached/background worker lanes (Phase 4), no
  `delegate_task(background=True)`, no model-based follow-up classifier (Phase 5),
  no "Ralph" focused-agent runtime.
- No change to `/busy interrupt|queue|steer|integrated` semantics, no change to
  the default busy mode.
- No change to the gateway's pending-message model. `gateway/platforms/base.py`
  `merge_pending_message_event` and the `adapter._pending_messages: Dict[str,
  MessageEvent]` single-slot model are untouched: that function is directly
  unit-tested (`tests/gateway/test_session_race_guard.py`) and feeds ~6 call
  sites and the restart-drain / race-guard paths — swapping it to a
  `PendingTurnQueue` is exactly the "broad gateway surgery" the packet says to
  defer. `from_gateway_event` is provided and tested as the one-way bridge for
  whoever does that migration.
- No TUI changes. `tui_gateway/server.py` has its own `_load_busy_input_mode`
  and the queue is currently frontend-owned (`ui-tui/src/app/useSubmission.ts`
  `queueRef` / `composerActions.enqueue`). Wiring it to the structured queue is
  a follow-up (see bridge points).

## Bridge points for later phases

1. **Gateway adoption.** Replace `adapter._pending_messages: Dict[str,
   MessageEvent]` with `Dict[str, PendingTurnQueue]` (or keep the dict and store
   a `PendingTurnQueue` per session). On ingress, `from_gateway_event(event,
   session_key)` → `queue.append(item)`. At drain (`_dequeue_pending_event` /
   `_drain_pending_after_session_command` / the post-run pickup in
   `gateway/run.py`), `queue.drain_coalescible_text_until_boundary()` for the
   text run, then pop the next non-text item separately. `merge_pending_message_event`'s
   photo-burst / album / `merge_text` logic becomes a small `PendingTurnQueue.append`
   policy (coalesce adjacent media with the same group; absorb captions into the
   adjacent media item via `boundary=caption`). Keep the restart-drain semantics:
   a multi-item queue should redeliver in order on resume.
2. **CLI `_pending_input`.** It could eventually *be* a `PendingTurnQueue` instead
   of a `queue.Queue` of mixed legacy payloads, but it is also written from the
   prompt_toolkit UI thread and read from `process_loop`, so that swap needs a
   thread-safety wrapper (a lock, or keep `queue.Queue` and store
   `PendingTurnItem`s in it). Not needed for Phase 2 — the current shims already
   route the *logic* through the structured module.
3. **TUI.** Port `useSubmission.ts`'s queue behavior to mirror `PendingTurnItem`
   boundaries (commands and media are hard boundaries; adjacent plain text
   coalesces), and have `tui_gateway/server.py` recognize `display.busy_input_mode:
   integrated` like the gateway already does. Longer term, move queue ownership to
   the backend `PendingTurnQueue`.
4. **Follow-up classification (Phase 5).** A classifier would consume
   `PendingTurnItem`s (and the `origin_busy` / `boundary` / `task_hint` hints)
   and decide append vs steer vs new-task vs clarify. Nothing in this module
   makes that decision; `task_hint` is reserved for it.

## Known risks / open questions

- The CLI shims drain the *entire* `_pending_input` queue, then re-queue the
  tail. `process_loop` is the sole runtime consumer of `_pending_input.get`, so
  this is equivalent to the prior "pull until boundary, then drain the rest into
  `restore`" behavior. There is still a theoretical producer/consumer race if a
  prompt-toolkit producer enqueues exactly between drain-empty and tail restore;
  that race already existed in the Phase 1 coalescer and is not widened here.
  Closing it properly requires changing `_pending_input` ownership/locking (for
  example a `PendingTurnQueue` plus lock, or an atomic front-restore wrapper), so
  it is intentionally documented as a Phase 3/4 queue-ownership hardening item
  rather than hidden inside this compatibility shim.
- `agent/pending_turn_queue.py` keeps a faithful copy of
  `cli._looks_like_slash_command` to stay a leaf module; if the CLI's version
  ever changes, the copy must be kept in sync (cross-reference comments are in
  both files).
- Open: should the gateway's album/caption merge become a `PendingTurnQueue`
  `append` policy (option 1 above) or stay a separate pre-queue step? Either
  works; the structured shape supports both.
