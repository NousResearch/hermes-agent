# T012 — test results + live smoke checklist

## Automated tests (2026-07-01)

Fully-owned modules — **all green**:
`menu_parse, registry, feed, watcher, relay` → **143 passed**.

Full session_orchestration sweep (explicit paths): **550 passed, 13 failed**.
The 13 failures are **pre-existing prior-session breakage**, NOT from this work:
- `test_session_orchestration_claude_code.py::TestLaunch` (4)
- `test_session_orchestration_omp.py::TestDrive` + `TestLaunchMarkerInjection` (9)

Proof they are pre-existing: each **passes on committed HEAD** and fails only
against the prior session's uncommitted `omp.py` / `claude_code.py` rewrites
(the `_wait_for_ready` box-detection + marker-injection changes described in
`HANDOFF-session-orchestration-feed.md`). They are the prior session's tests to
refresh when it commits its keeper changes. None touch this plan's areas.

## Coverage boundary

- `resolve_menu_answer` (pure), the `pre_keys` relay threading, watcher
  extraction/persist, the feed router + debounce, session-end reap, and the
  terminate-intent reap all have direct unit tests.
- The Discord `on_thread_update`/`on_thread_delete` listener (T009) is thin glue
  (registry lookup by `discord_thread_id` → `enqueue_terminate`), following the
  same live-smoke-only convention as `_handle_managed_thread_reply`. Its reap
  *semantics* are covered by the watcher terminate-reap test.

## Live run 2026-07-01 — extraction bug found & fixed

Restarted the gateway on `feat/so-answerable-needs-input`; a live omp session
(`b3269bdc…`) was sitting at `WAITING_USER` on an `ask` menu. First pass proved
the plumbing (columns populated, `last_input_kind=menu`) but the **ported
so-MCP heuristics mis-parsed omp v16.2's actual `ask` TUI**:
- grabbed a transient/top box instead of the live menu (multiple `│` boxes in
  the pane);
- leaked Nerd-Font glyphs (U+F192/U+F10C) into labels;
- dropped the 2-space-indented "Other (type your own)" as a description;
- captured omp's reasoning stream as the question.

**Fix (menu_parse.py):** footer-anchored parse — locate the
`enter select / esc cancel` footer, take the `│`-block directly above it, strip
leading private-use glyphs, classify label-vs-description by indent, and pull the
question from the line above the block. Legacy whole-pane scan kept as fallback
for Claude-style menus. Locked in as `test_extract_real_omp_v16_ask_menu`.

**Verified live** — after the fix the watcher re-parsed the running session and
the registry row now reads:
`last_input_kind=menu`,
`last_question="What task should I work on after this clarification?"`,
`last_options=["Code change (Recommended)","Investigation","Plan/review","Other (type your own)"]`.

Still needs a human action to validate end-to-end: **reply `1` in that
session's thread** and confirm the omp menu is Escaped and the agent proceeds
(T006 answer route + the Escape-cancel risk).

## Live run 2 — omp free-form wait detection (added, greenlit)

Live test surfaced a second gap: omp answered a "clarifying question" as
free-form prose and idled at its composer — no `❯`, no menu footer, no marker —
so `parse_pane_lifecycle` returned RUNNING, the session never became
WAITING_USER, the question was never forwarded, and it tripped the hang ladder
instead. (Also: `push_hang_notification` doesn't @-mention or hit the feed.)

**Fix (greenlit "stable+idle ⇒ WAITING_USER"):**
- `adapters/omp.py::pane_is_idle_waiting` — TUI composer box present AND not busy
  AND not already a menu/prompt/handoff → candidate free-form wait; exposed via
  `OmpAdapter.idle_waiting`.
- `watcher.py` — promote detect()=RUNNING → WAITING_USER only when the omp pane
  is STABLE (prior pane hash == this tick's) and `idle_waiting()` is true; gated
  on `getattr(adapter,"idle_waiting")` so only omp is affected. Stability gate
  avoids flapping on a momentary mid-turn idle.
- `menu_parse.py::_extract_free_form_question` — isolate the real question from
  omp's reasoning stream / update banner / injected nudge / status chrome
  (prefer the last `?`-terminated line).

Tests: `TestPaneIsIdleWaiting` (6), `test_extract_free_form_omp_question_isolated`,
`test_omp_stable_idle_promotes_running_to_waiting_user`. Owned suite: 152 green.

**Verified live:** session `936f60bb` was promoted RUNNING→WAITING_USER
(kind=`prompt`) by the new detection after a gateway restart.

Known tradeoff (accepted): a finished-turn idle (e.g. agent says "Complete.")
also surfaces as WAITING_USER. The state+question-digest debounce keeps it to one
ping per distinct message. Tunable later (e.g. suppress completion-phrase idles).

## Live smoke checklist (run on the gateway host)

Restart gateway to pick up code: `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`

1. `@Hermès so omp z-harness "<task that triggers an omp Accept/Defer menu>"`
   → thread created; when omp shows the menu, the thread message shows the
   **question + numbered options** + "Reply with the option number".
2. Reply `2` in the thread → **RISK CHECK**: confirm the omp menu is cancelled
   by `Escape` and the pasted "user chose option 2: «…»" advances the agent
   (session leaves WAITING_USER). This is the one design risk (Escape-cancel).
3. Trigger a second, different menu → confirm it re-notifies (debounce keyed on
   state + question digest).
4. Check `#feed`: the digest line is a concise `<#thread>` + `<@user>` router
   with a truncated question, not a full box dump.
5. Archive/close the thread → session terminates (tmux killed), feed line drops,
   attention resolved.
   Registry check:
   `sqlite3 ~/.hermes/state.db "select task_id,state,last_input_kind,last_options,discord_thread_id from session_orchestration order by rowid desc limit 3;"`

## Caveats / follow-ups

- **Discord thread auto-archive footgun.** Discord auto-archives inactive
  threads (guild setting: 1h/24h/3d/7d). That fires the same
  `on_thread_update(archived=True)`, so a quiet `WAITING_USER` session could be
  auto-terminated if the user is slow to answer. Options if this bites:
  (a) reap only on `on_thread_delete` (deliberate close), not archive; or
  (b) skip terminate when `state == WAITING_USER`; or (c) raise the thread's
  `auto_archive_duration`. Left as the user chose (archive⇒terminate) — flag.
- Prior-session `omp.py`/`claude_code.py` unit tests need refreshing when those
  keeper changes are committed (13 failing, unrelated to this plan).
