# Wave-1 Region Analysis — Free-Function Module Tail (kanban_db R5 recipe, 2026-08-05)

Companion to `wave2-blind-cluster-mapping.md` (which covers class-body mixin godfiles like the slack adapter). This is the recipe for a **module whose region is a tail of top-level FREE FUNCTIONS** — no classes, no `self`, no MRO concerns. Proven on `hermes_cli/kanban_db.py` R5 (8241–10275, 45 top-level defs, 11 clusters, 2,035 lines) against pin `01a1037d1e`. Deliverable: `R5-analysis.md`, 35 KB, written EARLY and verified.

## What differs from the class-body recipe

1. **AST inventory is simpler**: collect top-level `FunctionDef`s only (no `col_offset == 4` class-method filter — there are no classes in the window). Still count by script, not by eye. Record module-level *state* too (kanban R5: one `set` global `_retagged_workspace_roots` at 8938) — it is an extraction constraint, not a def.
2. **The file's own `# ----` section banners ARE the cluster boundaries** (kanban R5 had 8 banners for 11 clusters; the 3 banner-less gaps were small leaf clusters). Read the whole region first and trust the banners — they matched the AST-derived clusters 1:1.
3. **Reverse-caller scan MUST filter by import surface, not bare-name regex.** A repo-wide `grep -rln '\b<fn>\b'` on generic names (`_positive_int`, `_looks_like_path`) returns files that merely DEFINE their own same-named function (`agent/tool_guardrails.py`, `tools/wake_word.py`) — false positives. Decide "is this function consumed externally" with: (a) `from hermes_cli.kanban_db import X` / `from hermes_cli import kanban_db as kb` occurrences, then (b) `kb\.<fn>` / `kanban_db\.<fn>` attribute-access greps. Name regex alone is only a *candidate* list.
4. **Cross-check your cluster map against the campaign's `shard-plan.json` during analysis** (find it at `<campaign>/godfile/<god>/shard-plan.json` or the earlier shard-plan dir). It classifies every def (move/stay/shared + proposed module name). Agreement is a strong sanity check on cluster boundaries; disagreements (plan says `stay` where you see a cluster) are worth an explicit note. kanban R5: 45/45 defs matched plan s5 targets; the dispatcher core (`dispatch_once`/`_dispatch_once_locked`) is plan-`stay` — the analysis must respect that and exclude it from extraction slices even though it dominates the window.
5. **Extract the intra-window call graph via AST** (walk each window function, collect `ast.Call` to window names) — this is what proves cohesion AND finds the zero-outgoing-edge leaf clusters that are the cleanest slices. kanban R5: C6 (stats: `board_stats`/`_to_epoch`/`task_age`) had **zero pre-R5 in-file deps and zero outgoing edges** — the decisive cleanest-slice signal.
6. **Test-surface scan per function by name** (grep test files for each def name). Caveat learned: **0 direct hits ≠ untested** — `_dispatch_once_locked` has zero by-name test hits but is exercised through `dispatch_once` stubs; `_worker_terminal_timeout_env`/`_rotate_worker_log` are covered indirectly via `_default_spawn` tests. Report both direct-hit counts and the integration path that covers them.

## Seam design for free-function extraction

- **Re-export shim is the whole seam**: move the functions to `hermes_cli/<mod>.py` and leave in the godfile at the original location:
  ```python
  # (moved to hermes_cli/stats_mixin.py — extraction #78632)
  from hermes_cli.stats_mixin import board_stats, task_age, _to_epoch  # noqa: F401
  ```
  Zero caller edits required — every `kb.<fn>` / `from hermes_cli.kanban_db import <fn>` consumer keeps working.
- **Import-direction / cycle analysis before naming the new module**: for each cluster, list what it needs from the godfile's PRE-R5 region. One-way (new module imports nothing from godfile) = clean. Back-edges (`write_txn`, `kanban_db_path`, `Event`, `_CTX_MAX_*` constants) = top-level `from hermes_cli.kanban_db import X` in the new module + top-level shim = **import cycle**. Mitigations: lazy imports inside functions (the file's own established pattern for `profile_exists`/`load_config`), or move the shared pre-R5 helpers/constants to a `kanban_core.py`/`kanban_constants.py` slice first. kanban R5: C6/C8/C9/C10/C11 are one-way; C5/C7/C4 have back-edges → wave-2+.
- **Module-level state must travel with its function** (kanban R5: `_retagged_workspace_roots` set belongs to `_retag_legacy_worker_sessions` — splitting them silently loses the in-process dedup).
- **Note constants living OUTSIDE all region windows** (`_CTX_MAX_*` @331-335, `DEFAULT_LOG_*`/`KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS` @6744-6753) — a `kanban_constants.py` pre-slice is its own campaign slice, flag it for the planner.

## Cleanest-slice scoring (free-function tails)

Rank by: (1) zero pre-R5 in-file deps, (2) zero outgoing intra-R5 edges, (3) small diff (80–110 lines ideal for wave 1), (4) existing direct test surface + real production callers (CLI + dashboard endpoints beat test-only), (5) shard-plan `move` agreement, (6) no model/constant coupling (string annotations under `from __future__ import annotations` need no import). kanban R5 winner: C6 stats → `stats_mixin.py`; runner-up C11 runs accessor (108 lines, widest 11-file test surface, single `Run` model dep via lazy import). Do NOT lead with the biggest or most-gateway-critical cluster (C7 notify subs, 417 lines) or the env-contract spawner (C4).

## Risk catalog to include in every function-tail analysis

- **Cross-file runtime env contracts** (not imports, so invisible to grep-by-import): `_default_spawn` sets `HERMES_KANBAN_GOAL_MODE=1` + `-Q` flag, consumed in `cli.py` (`if os.environ.get("HERMES_KANBAN_GOAL_MODE") == "1"` → `_run_kanban_goal_loop_q`). Any move of the spawner must not change env keys/flags. Grep the consumers (`cli.py`, `gateway/`, `tools/`) for `HERMES_*` names the region writes.
- **CLI interdependency surface**: count the godfile's consumers and how many R5 members each calls (`hermes_cli/kanban.py` calls 17 R5 members via `kb.`; `gateway/kanban_watchers.py` 6; dashboard + tui_gateway + tools more). Every extraction must keep the godfile namespace intact via shims — this is the campaign's #1 discipline for this god.
- **Read-only probe semantics**: `count_notify_subs` opens its own `sqlite3.connect(uri + "?mode=ro")` (no DB creation, missing table → 0). Any refactor must preserve that — do not "normalize" it onto `connect()` (which creates the DB).
- **Boundary straddle**: `dispatch_once` (8204–8271) straddles R4/R5 (starts 37 lines before the 8241 window edge) — coordinate with the R4 inventory (`r4_inventory.py` window was 6181–8240); plan classifies it `stay` in both, so no conflict, but record it.

## Gotchas hit on Windows this session

- **`git worktree add /c/tmp/ws-kb-r5` silently creates `C:/c/tmp/ws-kb-r5`** (MSYS leading-slash path treated as repo-relative). Fix: remove with the REGISTERED form `git worktree remove C:/c/tmp/ws-kb-r5 --force`, `git worktree prune`, re-add with native `C:/tmp/...`. (Same as the slack r5b pitfall in SKILL.md — it recurs per-god.)
- **`execute_code`/python scripts run from the session cwd, NOT the worktree** — relative paths silently miss. Use absolute `C:/tmp/ws-<god>-r<N>/...` paths inside analysis scripts.
- **`awk 'NR>=A && /regex/'` for boundary def-lists**: confirm the last pre-region def line so you can name the straddler precisely.
