---
artifact: intent
slug: so-answerable-needs-input
level: deep
generated_at: 2026-07-01T00:00:00Z
frozen_at: pending
planning_mode: intent
---

## Intent

When a tool-spawned omp session in the `session_orchestration/` subsystem hits
`WAITING_USER` on an interactive selection menu, the user currently sees only a
state icon — no question, no options, and no way to answer (a pasted "2" selects
nothing on an arrow-key menu). This effort makes those needs-input moments
**answerable in the session thread**: the watcher extracts the menu's question
and option labels from the tmux pane text, presents them as a numbered list in
the thread, and lets the user answer with a plain option number. A numeric reply
is resolved by cancelling the menu with `Escape` and pasting a natural-language
selection into omp via the existing relay — no keystroke-navigation arithmetic.
The `#feed` digest becomes a concise `@`-mention router that links to the thread
rather than dumping the full question, and attention/feed items are reaped (and
the session terminated) when the thread is closed/archived or the session ends.

It retargets the proven so-MCP answerable design
(`z-harness/scripts/hermes/mcp_hermes_orchestrator.py`: `_extract_menu_options`,
`_extract_needs_input_context`, `_is_rule`) onto the single-writer watcher, and
builds on committed `92c9661` (thread adoption + `discord_user_id` + `@`-ping).

## Not doing

- **No Down×n+Enter keystroke navigation.** The so-MCP `navigate_so_session`
  cursor-arithmetic path is intentionally NOT ported. Menu answers use
  `Escape` + natural-language selection through the existing paste `drive()`
  route. No `navigate()`, no index→keystroke mapping, no re-capture index race.
- **No reactions.** No emoji-reaction listener, no `1️⃣–4️⃣` mapping. Per the
  handoff user preference, the answer channel is a numeric text reply in-thread.
- **No re-solving the spawn submit-race / TUI-ready gaps.** Already fixed in the
  current omp adapter (`_await_tui_ready`, literal-send, settle, separate Enter).
- **No change to free-form `❯` prompt answering.** Text replies to a bare prompt
  already paste correctly via `_handle_managed_thread_reply`; left untouched.
- **No new omp marker emission.** omp stays marker-less; extraction is
  pane-text only. Claude Code marker-driven `last_question` is not regressed.
- **No general Discord thread lifecycle framework.** Only the archive/close →
  terminate hook this feature needs.

## Consider for this

- **Single-writer invariant is load-bearing.** Only `watcher.py` mutates the
  registry. Menu extraction runs inside `_process_row` while the row lock is
  held (pane text is already captured there); the gateway reply-handler and the
  archive listener MUST route mutations through `enqueue_intent(...)`, never
  direct writes.
- **Extraction must be a pure, unit-tested function** (`menu_parse.py`) so both
  the omp and claude pane paths can call it and so the box-drawing heuristics
  (`│ Label │` rows, indented-description skip, rule/footer filtering,
  tail-keep + raw fallback) are testable without a live tmux.
- **Multi-turn re-notify.** omp can show menu→menu faster than the 1-minute
  watcher tick. The current notify debounce keys on `state` alone, so a *new*
  question within the same `WAITING_USER` state can be suppressed. Debounce must
  key on `state + question-digest`.
- **`Escape`-cancel semantics are the one real risk.** Validate early that
  `Escape` exits an omp selection menu into a paste-accepting composer; if a
  given menu doesn't cancel cleanly, the pasted natural-language instruction is
  still interpretable by the agent, but the acceptance check must confirm the
  happy path on a live session.
- **Reap must be idempotent and complete.** On session-end OR thread
  archive/close: resolve the attention item, drop the `#feed` router line (falls
  out of `reconcile_attention_digest` once resolved), stop `@`-pinging
  (`clear_last_notified`), and — for archive/close — terminate the tmux session
  and mark the row terminal.
- **Invalid replies.** A non-numeric or out-of-range reply on a menu row should
  fall through to the normal paste path (agent can interpret prose), not error.
- Dead old-path reaction wiring in `plugins/platforms/discord/adapter.py`
  (`_render_so_needs_input`, `_so_reaction_index`, `_navigate_so_session`,
  `on_raw_reaction_add`, `_so_inflight`) should be reverted — it targets the
  disabled path and contradicts the no-reactions decision.

## Acceptance checklist

- [ ] A pure `menu_parse.extract(pane_text)` returns `(question, options,
      is_menu)`; unit tests cover: box-row option extraction, indented-
      description skip, rule/footer filtering, tail-keep of the question, and
      the raw-tail fallback for unknown TUI shapes (ported from the so-MCP tests).
- [ ] On an omp `WAITING_USER` menu, the watcher persists `last_question`,
      `last_options`, and `last_input_kind` (`menu`|`prompt`) to the registry row
      while holding the row lock (single-writer preserved); a registry migration
      adds the `last_options` and `last_input_kind` columns.
- [ ] The session thread shows the extracted question followed by a numbered
      option list (`1. …`, `2. …`) when `last_input_kind == menu`.
- [ ] Replying with a valid option number in the thread cancels the omp menu
      with `Escape` and pastes a natural-language selection naming the chosen
      option label; the session advances (observed leaving `WAITING_USER`) on a
      live smoke test.
- [ ] Replying with a non-numeric or out-of-range value on a menu row falls
      through to the existing paste path without raising.
- [ ] A second, different question within `WAITING_USER` re-notifies (notify
      debounce keys on `state + question-digest`, verified by test).
- [ ] The `#feed` digest renders each unresolved item as a concise one-line
      `@`-mention router linking to the thread (`<#thread_id>`), with the
      question truncated — full question/options live in the thread, not the feed.
- [ ] Closing or archiving the session thread enqueues a terminate intent; the
      watcher then kills the tmux session, marks the row terminal, resolves the
      attention item, and drops the feed router line.
- [ ] Session end (DONE/ERROR/dead-tmux) resolves the attention item, drops the
      feed router line, and stops further `@`-pings.
- [ ] The dead old-path reaction wiring in the Discord adapter is removed.
- [ ] `python -m pytest` for the touched session_orchestration test modules is
      green.
