# Build brief: kanban task-tree renderer

**One-sentence goal**: Let any subtask's lineage be traced from the board by adding a
`hermes kanban tree [root]` command (Jira-style ASCII hierarchy + `--json` + `--mermaid`
gantt/flow output), so every child's original parent task is visible at a glance in CLI,
TUI, and the desktop/dashboard (which render the same output and already render mermaid).

**Acceptance criteria**:
- [ ] `hermes kanban tree` (no arg) renders the full board as a forest of roots; each
      subtree is indented under its root, with every non-root row showing its parent chain.
- [ ] `hermes kanban tree <root_id>` renders only that root's subtree (descendants), with the
      root at top.
- [ ] `--json` emits a nested tree structure {id, title, status, assignee, children:[...]} —
      machine-consumable, no orphaned children (each node nested under its parent).
- [ ] `--mermaid` emits a `flowchart TD` graph from the same traversal; node ids equal task ids
      so the desktop's existing mermaid embed renders it directly.
- [ ] Both a parent/child task pair and a decomposed graph render correctly (uses the same
      `task_links` edges that `show` reports).
- [ ] Cycles (shouldn't exist by DB guard, but defensively) terminate without infinite loop.
- [ ] All existing kanban tests still pass; new tests cover tree rendering, json, mermaid, and
      subtree scoping.

**Out of scope**:
- No dashboard plugin / new web panel in this change — the desktop/dashboard surface is the
  embedded TUI and mermaid embed, both of which consume this command's output. A standalone
  dashboard kanban panel would be a separate, larger change (the "matching dashboard panel"
  from the earlier proposal).
- No changes to the board's data model; `task_links` already stores lineage.
- No click-handling / interactive navigation.

## Design notes
- Approach: new `hermes_cli/kanban_tree.py` sibling module (mirrors `kanban_decompose` /
  `kanban_specify` pattern) exposing a pure builder + formatters. Add DB helpers in
  `kanban_db.py` / graph module to bulk-load the adjacency (extend the existing
  `task_graph_contexts` style). Register the `tree` verb in `kanban_parser.py` `_SPECS` and
  dispatch in `kanban.py` `_HANDLERS`.
- Dependencies added: none (stdlib + existing sqlite).
- Test strategy: unit tests on the pure tree builder + formatters; a couple of CLI-level tests
  via `run_slash`/`kanban_command` in `tests/hermes_cli/test_kanban_cli.py` style, with the
  `kanban_home` fixture.

## Verification (local)
- Tests pass: `tests/hermes_cli/test_kanban_tree.py` — 15 passed. Adjacent kanban suites green:
  `test_kanban_cli (4)`, `test_kanban_cli_dispatch_passthrough (2)`, `test_kanban_graph_identity (2)`,
  `test_kanban_swarm (5)`, plus `test_kanban_boards`, `test_kanban_db`, `test_kanban_core_functionality`,
  `test_kanban_decompose`, `test_kanban_decompose_db` (windows_only marked files skipped on linux as intended).
- Build succeeds: python-bytecode precompile in run_tests.sh clean; imports resolve.
- Smoke test (real board, `python -m hermes_cli.main kanban tree`):
  - Whole board forest: two independent roots, subtasks nested under parent with `├─`/`└─`.
  - `tree <root>` scoped subtree; 2-level nesting (`└─` then `   └─`).
  - `--json`: nested `{id, title, status, assignee, children}` — subtask under its parent.
  - `--mermaid`: `flowchart TD` with `parent --> child` edges; desktop mermaid embed ready.

## Security surfaces (Rocio's focus)
- `--json` / `--mermaid` serialize task titles verbatim into output. Mermaid labels are escaped
  (`\`, `"`, `[`, `]` and newlines) in `kanban_tree._mermaid_label`, so a hostile title cannot
  break out of a node label into diagram text — the desktop renders mermaid at
  `securityLevel: strict`, so this is the injection boundary. The escaping mirrors the existing
  ```mermaid`` embed path (`kanban_tree.py:175-183`).
- Read-only command: only SELECTs from `tasks` + `task_links`; no state mutation, no new
  privileges.

## Self-review notes
- Watch for title injection into mermaid labels (escape `"` and `[`).
- Ensure cycle-termination guard present even though DB prevents cycles.

## Handoff
- Branch: <name>
- PR: <url>
- Handoff brief: <path>
- Awaiting Rocio OK before staging/prod deploy.