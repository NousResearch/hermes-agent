# Handoff: kanban task-tree renderer → Rocio security review

**Build brief**: briefs/miguel/2026-09-10_kanban-tree.md
**Branch**: `feat/kanban-task-tree` (2 commits: `1c0dec60c1` feat + docs/handoff)
**Worktree**: `~/.hermes/hermes-agent/.worktrees/kanban-tree` — exact head via `git rev-parse feat/kanban-task-tree` there
**PR**: *not yet opened — push blocked on credentials (admin-lock)*

## What changed
Adds `hermes kanban tree [task_id]` to render the parent/child task hierarchy so every
subtask's original root task is visible — the lineage gap where decomposed/swarmed children
looked orphaned on the board. Read-only: derived entirely from the durable `task_links`
edges + the `tasks` table.

Output modes:
- no arg → whole board as a forest (roots + nested children, Jira-style `├─`/`└─`)
- `task_id` → that root's subtree only
- `--json` → nested `{id, title, status, assignee, children}`
- `--mermaid` → `flowchart TD` graph (the desktop app already renders mermaid fences)
- `--archived` → include archived tasks

New sibling module `hermes_cli/kanban_tree.py` (mirrors `kanban_decompose`/`kanban_specify`
pattern). Thin handler + parser entry in `kanban.py` / `kanban_parser.py`.

## Why
BOSS asked how to trace a kanban subtask back to its original parent in the Hermes desktop
and dashboard. The relationship is stored (`task_links`) but no surface displayed it. `tree`
closes the gap in CLI, TUI, and desktop/dashboard (all three render the same output).

## Security surfaces (focus for Rocio)

| Surface | File / line | Notes |
|---|---|---|
| Input validation | `kanban_tree.py:104` `build_forest` | `task_id` validated (unknown → ValueError). Optional positional, no shell. |
| Output escaping | `kanban_tree.py:171` `_mermaid_label` | **Primary surface.** Task titles flow verbatim into mermaid labels. Escaped: `\`, `"`, `[`, `]`, and CR/LF→space. Desktop renders mermaid at `securityLevel: strict` — this is the injection boundary; a hostile title must not break out of a node label into diagram text. ASCII/JSON paths carry the same title text but are not an injection context (JSON via `json.dumps`, ASCII is display-only). |
| Auth / authz | `kanban.py:1236` `_cmd_tree` | Read-only command; no mutation; inherits the existing kanban CLI authz (same `run_slash`/`kanban_command` gate as `list`). |
| Cryptography | N/A | No crypto. |
| Parameterized queries | `kanban_tree.py:52,60` `_load_adjacency` | Both queries are fixed-SQL with no user-controlled interpolation. |
| Secrets in code | none | No secrets added; no `.env` / token handling in the change. |
| Logging / audit trail | none | Read-only view command; no security-relevant events generated (matches `list`). |
| Network / API surface | none | No new endpoints; no network I/O. |

## Dependencies added
- None. Stdlib (`sqlite3`, `json`) + existing `hermes_cli.kanban_db`. No dependency footprint
  change (see `dev-dependency-hygiene`).

## Test coverage
- `tests/hermes_cli/test_kanban_tree.py` — **15 passed**
- Coverage of acceptance criteria:
  - full-board forest + parent chain: `test_build_forest_nests_child_under_parent`,
    `test_build_forest_multiple_roots_are_forest`, `test_cli_tree_text` ✅
  - `tree <root>` scoped subtree: `test_build_forest_root_subtree_scoped`,
    `test_cli_tree_scoped_subtree` ✅
  - `--json` nested: `test_render_json_parses_and_nests`, `test_cli_tree_json` ✅
  - `--mermaid` edges + escaping: `test_render_mermaid_edges_follow_lineage`,
    `test_render_mermaid_escapes_label_injection`, `test_cli_tree_mermaid` ✅
  - decomposed-graph render: covered by the general `task_links` builder tests + live smoke test;
    not a dedicated decompose-live test (noted in self-review)
  - cycles terminate: `test_build_forest_terminates_on_cycle` ✅
  - all existing kanban tests pass: `test_kanban_cli`, `test_kanban_cli_dispatch_passthrough`,
    `test_kanban_graph_identity`, `test_kanban_swarm`, `test_kanban_boards`, `test_kanban_db`,
    `test_kanban_core_functionality`, `test_kanban_decompose`, `test_kanban_decompose_db` all green ✅
- Lint: `ruff check` clean on all 4 changed files.

## Self-review findings
- **Mermaid label escaping is the one thing to scrutinize first** (Rocio) — see the output-escaping
  row above. I escaped the four characters that can break a `flowchart TD` node label through the
  desktop's `securityLevel: strict` renderer. If you want belt-and-suspenders beyond that, I can
  also apply a stricter allowlist filter to labels.
- The smoke test used hand-wired parent/child chains (fine — the builder reads the same `task_links`
  edges regardless of how they were created). A dedicated decompose→tree e2e would be the natural
  follow-up if you want it; it wasn't my acceptance criterion.
- Read-only command verified: no INSERT/UPDATE/DELETE in `kanban_tree.py` or `_cmd_tree`.

## What I'm asking Rocio to verify
- [ ] Push `feat/kanban-task-tree` to `origin` and open the PR to `main`
      (credentials were blocked by admin-lock for me; this is also the build's security gate).
- [ ] No high/critical security findings
- [ ] Dependency footprint acceptable (none added)
- [ ] Input validation comprehensive
- [ ] Output escaping correct (mermaid labels)
- [ ] No secrets in code
- [ ] Approval to proceed to staging

## Awaiting
- Rocio's OK before staging deploy
- BOSS authorization before production deploy