---
artifact: tasks
slug: so-answerable-needs-input
planning_mode: intent
generated_at: 2026-07-01T00:00:00Z
---

# TASKS — so-answerable-needs-input

Retarget the so-MCP answerable design onto the single-writer
`session_orchestration/` watcher. Reference source for reuse:
`~/dev/z-harness/scripts/hermes/mcp_hermes_orchestrator.py`
(`_extract_menu_options` L509-538, `_extract_needs_input_context` L477-506,
`_is_rule` L472-474, `_MENU_FOOTER_HINTS` L468-470) and its test module.

Legend: `Advances:` cites the INTENT.md acceptance checklist item(s).

---

## T001 — Port the menu-parse pure module + tests `[ ]`
**Files:** `session_orchestration/menu_parse.py` (new),
`session_orchestration/tests/test_menu_parse.py` (new)
**Depends:** none
**Advances:** acceptance #1
Port `_is_rule`, `_MENU_FOOTER_HINTS`, `_extract_menu_options`, and
`_extract_needs_input_context` from `mcp_hermes_orchestrator.py` into a pure,
dependency-free `menu_parse.py` exposing `extract(pane_text) -> (question:
str, options: list[str], is_menu: bool)`. `is_menu` is true iff ≥1 box-row
option label was extracted. Port the so-MCP unit tests verbatim where possible
(ordered labels, indented-description skip, bare-prompt→empty, menu cleanup,
unknown-shape raw-tail fallback). No tmux, no I/O.
**Complexity:** medium

## T002 — Registry migration: `last_options` + `last_input_kind` `[ ]`
**Files:** `session_orchestration/registry.py`
**Depends:** none
**Advances:** acceptance #2
Add nullable columns `last_options TEXT` (JSON array) and `last_input_kind TEXT`
(`menu`|`prompt`|null) mirroring the existing `last_question` migration
(registry.py:~331). Additive/idempotent migration; extend the row read/write
mappers. No backfill.
**Complexity:** low

## T003 — Watcher: extract + persist question/options on WAITING_USER `[ ]`
**Files:** `session_orchestration/watcher.py`,
`session_orchestration/adapters/omp.py`,
`session_orchestration/tests/test_watcher_needs_input.py` (new/extend)
**Depends:** T001, T002
**Advances:** acceptance #2
In `_process_row`, when a row transitions to (or is observed in) `WAITING_USER`
via the pane path, call `menu_parse.extract(pane_text)` on the already-captured
pane text (watcher.py:~1451, lock held) and persist `last_question`,
`last_options`, `last_input_kind` on the row (single-writer). For omp,
`is_menu` distinguishes a selection menu from a bare `❯` prompt. Do NOT regress
the marker-driven `last_question` path for Claude Code. Test with fixture panes.
**Complexity:** medium

## T004 — Thread presentation: numbered options in-thread `[ ]`
**Files:** `session_orchestration/feed.py`,
`session_orchestration/tests/test_feed_turn_change.py` (new/extend)
**Depends:** T003
**Advances:** acceptance #3
Extend `push_turn_change` (feed.py:460-537) so that when `last_input_kind ==
menu` the thread message renders the question then a numbered option list
(`1. <label>` …) plus a one-line "reply with the option number" hint. Free-form
`prompt` rows keep current behavior. Preserve the existing `@`-mention + icon.
**Complexity:** medium

## T005 — Notify debounce keyed on state + question digest `[ ]`
**Files:** `session_orchestration/feed.py`,
`session_orchestration/tests/test_feed_turn_change.py`
**Depends:** T004
**Advances:** acceptance #6
Change the `_last_notified` debounce (feed.py:~534) to key on
`(new_state, sha1(last_question or ""))` so a *new* question within the same
`WAITING_USER` state re-notifies. Keep `clear_last_notified` on leaving
attention. Test: menu→different-menu without an observed RUNNING tick re-emits.
**Complexity:** medium

## T006 — Answer route: Escape + natural-language selection `[ ]`
**Files:** `gateway/run.py` (`_handle_managed_thread_reply` ~10954),
`session_orchestration/adapters/omp.py`,
`session_orchestration/relay.py`,
`session_orchestration/tests/test_menu_answer.py` (new)
**Depends:** T003
**Advances:** acceptance #4, #5
When the matched row has `last_input_kind == menu` and the reply is a valid
option number `n ∈ [1..len(options)]`: (a) send a single `Escape` to the pane to
cancel the menu (new adapter helper, e.g. `cancel_menu()` / `drive(..., pre_keys=["Escape"])`,
using `tmux send-keys` which omp.py already uses at L700/L810); (b) compose and
paste a natural-language instruction naming the chosen label — e.g. `"[Hermès]
You presented a selection menu. The user chose option {n}: «{label}». Please
proceed with that choice."` — via the existing lock-holding `drive()` paste
path. Non-numeric / out-of-range replies fall through to the current paste path
unchanged. Route all of this through the relay so the row lock is respected;
the watcher re-detects the next state naturally (no manual re-arm).
**Complexity:** high

## T007 — #feed digest becomes a concise @-mention router `[ ]`
**Files:** `session_orchestration/feed.py` (`render_attention_digest`
291-362), `session_orchestration/tests/test_feed_digest.py` (new/extend)
**Depends:** T003
**Advances:** acceptance #7
Reshape each digest line to a one-line router:
`<@user> — <agent>/<repo>: <question truncated ~80 chars> → <#thread_id>`.
The full question/options stay in the thread (T004), not the digest. Keep the
single-message upsert/content-hash reconcile. Handle a missing
`discord_thread_id` (raw task-id fallback, as today).
**Complexity:** medium

## T008 — Reap on session end (verify + close gaps) `[ ]`
**Files:** `session_orchestration/watcher.py`,
`session_orchestration/feed.py`,
`session_orchestration/tests/test_reap.py` (new/extend)
**Depends:** T007
**Advances:** acceptance #9
Verify and, where missing, ensure that on terminal state (DONE/ERROR/dead-tmux
reap at watcher.py:1315-1330, 1404-1447) the attention item is resolved, the
feed router line drops out of the digest, and `@`-pings stop
(`clear_last_notified`). Add a test asserting the full teardown for a session
that ends while `WAITING_USER`.
**Complexity:** medium

## T009 — Thread archive/close listener → terminate intent `[ ]`
**Files:** `plugins/platforms/discord/adapter.py`, `gateway/run.py`
**Depends:** none
**Advances:** acceptance #8
Add a Discord listener for thread archive/close/delete
(`on_thread_update` w/ `archived`, `on_thread_delete`). On event: look up the
row whose `discord_thread_id` matches; if found, `enqueue_intent("terminate",
task_id, reason="thread_archived")`. Gateway only enqueues — never writes the
registry directly (single-writer).
**Complexity:** medium

## T010 — Watcher: handle terminate intent → kill + reap `[ ]`
**Files:** `session_orchestration/watcher.py`,
`session_orchestration/registry.py`,
`session_orchestration/tests/test_terminate_intent.py` (new)
**Depends:** T008, T009
**Advances:** acceptance #8
Drain the `terminate` intent: kill the tmux session, mark the row terminal
(ERROR/DONE per convention), resolve the attention item, and drop the feed
router line — reusing the T008 teardown path. Idempotent if the session is
already gone. Test the archive→terminate→reap chain end-to-end.
**Complexity:** medium

## T011 — Remove dead old-path reaction wiring `[ ]`
**Files:** `plugins/platforms/discord/adapter.py`
**Depends:** none
**Advances:** acceptance #10
Revert the disabled so-MCP reaction code (`_render_so_needs_input`,
`_so_reaction_index`, `_navigate_so_session`, `on_raw_reaction_add`,
`_so_inflight`) flagged in the handoff. Confirm nothing in the live
session_orchestration path references it before deleting.
**Complexity:** low

## T012 — Live smoke + suite green `[ ]`
**Files:** (test/docs only)
**Depends:** T001-T011
**Advances:** acceptance #4, #11
Run `python -m pytest` for touched `session_orchestration` modules. Live smoke:
`@Hermès so omp z-harness "<task that triggers a menu>"` → thread shows question
+ numbered options; reply "2" → menu Escapes, agent proceeds; archive thread →
session terminates + feed line drops. Confirm the `Escape`-cancel happy path
(the one flagged risk). Record results in the plan archive.
**Complexity:** medium
