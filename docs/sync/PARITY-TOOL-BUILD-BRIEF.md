# Build brief: `hermes-parity` (worker instructions)

You are building a new internal CLI tool in THIS worktree (branch `feat/hermes-parity`).
The authoritative spec is `docs/sync/2026-07-10-hermes-parity-SPEC.md` — read it FULLY
first, including the v2 delta block at the top (those review verdicts are binding).

## Deliverables

1. **`scripts/hermes_parity/`** — Python package, stdlib-only, Python 3.11+.
   - `__init__.py`, `__main__.py` (argparse CLI, `python -m hermes_parity <cmd>`)
   - Modules by concern: `gitops.py` (all subprocess git calls), `state.py` (atomic
     JSON state, tree-SHA invalidation), `buckets.py` (conflict bucketing),
     `forkdelta.py` (fork-delta computation + manifest cross-reference),
     `lint_unbound.py` (unbound-name AST linter), `gates.py` (the 6-stage ladder),
     `bisect.py` (baseline classification), `cli.py` (subcommand wiring, output).
   - Subcommands: `status`, `start`, `gates`, `bisect`, `finish`, `clean` — exactly as
     specced (v2 ladder: markers, imports, unbound-lint, manifest+forkdelta, tests,
     linuxonly; CI-owned checks printed as a reminder checklist only).
2. **`docs/sync/fork-features.json`** — seed manifest. Schema:
   `[{"feature": str, "tests": [pytest-node...], "paths": [repo-relative glob...], "why": str}]`
   `paths` is what fork-delta coverage is checked against. Seed with these 5 entries
   (find the real pytest node IDs by grepping the repo's tests/):
   - cron-subagent approval gate reads ContextVar not raw env
     (tests/gateway/test_cron_subagent_session.py, TestCronSubagentApprovalGating;
     paths: hermes_cli/approval.py, hermes_cli/gateway/session_context.py)
   - systemd restart exits 0, darwin/launchd exits 75
     (tests/gateway/test_gateway_shutdown.py systemd+launchd restart tests;
     paths: hermes_cli/gateway/run.py)
   - hygiene compaction announces in-chat on success
     (tests/gateway/test_session_hygiene.py msgcount announce test; paths: hermes_cli/gateway/run.py)
   - messaging + moa toolsets present (grep tests for send_message/mixture_of_agents
     toolset presence tests; paths: hermes_cli/tools*.py or wherever toolsets register)
   - telegram polling never drops pending updates on recovery
     (tests/gateway/test_telegram_polling_invariants.py or similar INV-2 guard tests;
     paths: hermes_cli/gateway/adapters/telegram*.py)
   If a pytest node you expected doesn't exist, find the closest real one and use it —
   every node in the manifest MUST actually collect (verify with pytest --collect-only).
3. **`tests/scripts/test_hermes_parity.py`** — unit tests for the pure logic:
   bucketing (build tmpdir git fixture repos with synthetic conflicts), fork-delta
   computation, unbound-name linter (true positive on the known incident pattern:
   a name used after its only binding was removed; plus a no-false-positive case on
   comprehensions/walrus/globals/nonlocal/star-imports — if star-import present, SKIP
   the file), state atomicity + tree-SHA invalidation, bisect classification matrix
   (all 4 classes) with a fake runner injected (no real pytest spawning in unit tests).
   Tests must pass under the repo's `scripts/run_tests.sh tests/scripts/`.
4. **`docs/sync/README-hermes-parity.md`** — short operator doc: lifecycle walkthrough,
   one section per subcommand, the landing rules (merge --merge never squash; BEHIND →
   merge fork/main never rebase), and the "add a manifest entry every postmortem" rule.

## Hard constraints

- stdlib only. No PyYAML, no requests, no gh-API calls from Python (shell out to `git`
  only; `finish` PRINTS a `gh pr create` command, never runs it).
- Never mutate the repo the tool is invoked from except: creating/removing worktrees
  under the configured worktree root, and writes inside those worktrees.
- Every destructive step prints the exact rollback command BEFORE executing.
- All state writes atomic (tmp file + os.replace). State lives in the merge worktree
  as `.parity-state.json`.
- `finish --force` requires `--force-reason`; it still runs gates and prints failures.
- Startup: verify `git --version` >= 2.38 for `status --predict` (merge-tree
  --write-tree); other commands degrade gracefully without it.
- Gates emit human table AND append JSONL records to `<worktree>/gates.jsonl`.
- Repo conventions apply (AGENTS.md): no change-detector tests, behavior contracts,
  type hints, no new deps.
- Defaults: worktree root = `~/.hermes/worktrees` (override via `--worktree-root`),
  remotes literally named `fork` (ours) and `origin` (upstream) — validate at startup
  and fail with a clear message if missing.
- The `tests` gate shells to `scripts/run_tests.sh` (repo-root relative in the merge
  worktree); `--fast` skips it. Do NOT reimplement a test runner.

## Definition of done (self-check before you stop)

- `python -m hermes_parity --help` and every subcommand `--help` work from repo root.
- `python -m hermes_parity status` runs against THIS repo live and prints real numbers.
- `scripts/run_tests.sh tests/scripts/test_hermes_parity.py` fully green.
- Every pytest node in fork-features.json collects (`pytest --collect-only <node>`).
- `python -m py_compile scripts/hermes_parity/*.py` clean.
- Do NOT commit — leave changes in the working tree for review.
