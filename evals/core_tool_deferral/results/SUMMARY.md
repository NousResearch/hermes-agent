# PR #97979 A/B verdict — core-tool deferral (288 live runs)

Date: 2026-08-29 · Harness: /tmp/ab97979/harness · Method: METHOD.md

## Arms
base = origin/main 3f36c87e1ebd (27 direct tools in the eval assembly, 47.4KB schema chars)
pr   = main + #97979 e16ad33a9d24 (12 direct: 9 working set + 3 bridge; 19 deferred; 21.0KB schema chars, −56%)

## Headline (mean of task means, 14 tasks × 3 reps; contested cells re-run to n=6)

| model | arm | accuracy | turns | tokens(k) | wall(s) |
|---|---|---|---|---|---|
| gpt-5.6-terra (large) | base | 0.938 | 6.0 | 80.9 | 27.6 |
| gpt-5.6-terra | pr | 0.879 | 6.6 | **62.5 (−23%)** | 27.1 |
| glm-5.3-flash (medium) | base | 0.915 | 6.0 | 101.0 | 56.5 |
| glm-5.3-flash | pr | **0.963 (+0.05)** | 8.8 | **89.6 (−11%)** | 59.5 |
| qwen3.8-27b (small) | base | 0.915 | 7.1 | 127.4 | 53.5 |
| qwen3.8-27b | pr | 0.907 | 9.4 | **118.4 (−7%)** | 79.2 |

Grand accuracy: base 0.923 vs pr 0.916 — flat within rep noise once the two
contested tasks were extended to n=6. Tokens down on every model. Turns up
~1–2 (bridge discovery round-trips), wall flat on terra/glm, +48% on qwen
(27B pays real latency for extra bridge turns).

## Deferred-tool discovery (PR arm, tasks requiring the tool, all models)
Perfect (9/9 or 18/18): session_search, todo_list, image_generate,
desktop_project, desktop_preview, drive_preview, annotate_preview,
apply_layout, focus_pane, read_terminal, read_window_below.
Near-perfect: cronjob_manage 16/18, gui_tour 8/9, process_manage 8/9.
Weak: computer_use 6/9, show_tip 6/9, clarify 7/18, setup_mcp 4/9*,
close_terminal 4/9*.
(*base-arm usage on the same tasks: setup_mcp 3/9, close_terminal 0/9 —
these two are NOT deferral regressions; models skip them even when visible.)

## The one real regression: clarify
base: clarify used 18/18, score 1.00 on the ambiguous-delete trap, all models.
pr: clarify used 7/18 → terra 0/6 (0.50), glm 3/6 (0.80), qwen 4/6 (0.87).
Models still ask — but as plain text, ending the turn (extra user round-trip,
no structured choices). The harness credits scripted replies; without that
continuation the task scores 0. Exactly trade-off #1 flagged in the PR body.
Safety note: in 0 of 288 runs was the WRONG file deleted — the failure mode
is degraded UX, never destructive action.

## screenshot_ambiguous (n=6): split, not directional
terra base 1.00 → pr 0.67 (2 reps answered from read_window_below instead of
discovering computer_use — catalog-stub misrouting to a cheaper adjacent tool);
but glm 0.67→1.00 and qwen 0.50→0.83 IMPROVED under deferral (the focused
catalog line beats 27 competing schemas for weaker models). Model-split, nets
to ~flat across the tier ladder.

## Controls
eager_refactor_control (eager-only tools): pr arm −49% tokens at held 1.00 —
pure schema-shrink win, no behavior change.
config_grep_distractor: 1.00 both arms, 0 false bridge calls on terra/glm —
no discovery-overhead tax on tasks that don't need deferred tools.

## Anomalies audited
- glm pr layout rep2 (41 turns, 514k tok): after completing the GUI task via
  bridge it burned 30 terminal calls "verifying"; score 1.0. Model paranoia,
  not a bridge failure.
- qwen pr screenshot rep3: hard wall timeout, scored 0, kept in denominator.
- 1 errored run / 288 total; raw-XML provider noise: 0.

## Verdict: SHIP, with one follow-up — un-defer (or pin) `clarify`.
The deferral mechanism works: discovery is essentially perfect for 14/19
tools, accuracy is flat overall (large model −0.06, medium +0.05, small
−0.01), token cost drops on every model, and the eager-surface control shows
the −49%-token schema win with zero accuracy cost. The single consistent
regression is clarify: structured ask-the-user collapses to plain-text
questions when the schema is invisible (7/18 vs 18/18). PR #91125
(always-visible deferred-tool pins) is the natural mechanism — pin clarify
eager by default, or drop it from _DEFAULT_DEFERRED_TOOLS (~250 tok cost).
computer_use on frontier models is worth watching but is model-split, not
directional. todo_list discipline concern from the PR body did NOT
materialize (18/18 discovery, multi-step scores held at 1.00).
