# BUILD BRIEF: hermes_parity tooling v2 (implement docs/sync/2026-07-16-hermes-parity-tooling-SPEC.md)

You are in a git worktree on branch `feat/parity-tooling-v2` (off fork/main). Implement the
spec at `docs/sync/2026-07-16-hermes-parity-tooling-SPEC.md` (v0.4, Momus clean APPROVE).
Read it FULLY first. All work is in `scripts/hermes_parity/` + `tests/scripts/`.

## Scope (items 1-5; item 6 is documented-only, skip)
1. **bisect fix** (`bisect.py`): per-side faithful venvs (baseline = fork/main's own venv;
   dual-venv baseline run when requirements/lock diff non-empty → MERGE REGRESSION (dep-driven)),
   N=3 stability loop, NONDETERMINISTIC + flaky sub-classes per the spec's classification table,
   both-sides pytest output persisted (bounded, $HOME/$VENV-rewritten, routed through
   agent.redact.redact_sensitive_text(force=True)) into gates.jsonl.
   NOTE: resolving "fork/main's own venv" = the shared checkout venv IS fork/main's venv in this
   fleet (deps rarely change); implement the dual-venv path as: detect requirements diff via
   `git diff <baseline>..HEAD -- requirements*.txt pyproject.toml uv.lock`; when empty (the
   common case) a single shared venv is faithful for both sides — document this in the code.
2. **lint-manifest** (new `lint_manifest.py` + CLI subcommand): zero-match path RED (glob-aware),
   non-collecting test nodeid RED (pytest --collect-only), vacuous-coverage floor (0 covered +
   non-empty manifest + ≥1 fork-delta file touched → RED absent --vacuous-ok ack).
   Also emit a GitHub Actions workflow snippet in the module docstring (CI wiring is a follow-up,
   not this PR).
3. **catchup subcommand** (`cli.py` + new `catchup.py`): refuse >50 commits; fast gates; test
   selection by AST import-graph closure over the FULL changed set seeded from the PRE-MERGE
   fork/main tip (use stdlib ast + module-path resolution — resolve __init__ re-exports by
   parsing __init__.py's import-from statements transitively; do NOT add a new third-party dep
   unless importlab is already in the venv — check first); per-changed-file coverage map in the
   report; UNTESTED-BY-CATCHUP flags; manifest tests unconditional; finish-compatible state.
4. **ack ergonomics** (`cli.py` + `state.py`): `ack --from-file` (path<TAB>reason lines; bare
   path inherits --reason which becomes REQUIRED when any bare line exists); `forkdelta
   --emit-uncovered` snapshot emitter; distinct `ack --vacuous-ok` kind stored separately from
   path acks. NO --all-uncovered flag (TOCTOU — deliberately not implemented).
5. **tests-stage SKIP semantics** (`gates.py`): no-corpus default = loud SKIP, exit 2 (distinct
   from failure 1), `{"gate":"tests","status":"SKIP"}` in gates.jsonl; `--full` runs
   run_tests.sh; finish() treats a SKIP-status tests gate as NOT-satisfied.

## Test requirements (the spec's RED-proofs are the deliverable, not decoration)
In `tests/scripts/test_hermes_parity.py` (extend) or a new sibling file:
- item 1: the SIX-fixture corpus (4 verdict cells + guaranteed-alternation NONDETERMINISTIC via
  persisted on-disk parity counter + dep-bump fixture classifying MERGE REGRESSION (dep-driven)).
  Use tiny synthetic git repos (tmp_path fixtures) — do NOT run against the real fork history.
- item 2: break-a-path → RED; fix → GREEN; vacuous-coverage floor RED; --vacuous-ok clears it.
- item 3: C1-shape fixture (non-conflicted signature change w/ fork caller NOT manifest-covered)
  → RED via closure; __init__-re-export fixture → RED; >50-commit refusal; coverage-map lines
  present in report output.
- item 4: from-file parsing (tab-reason + bare + missing --reason error); snapshot TOCTOU test
  (paths added after emit are NOT acked).
- item 5: SKIP exit=2 + jsonl status; finish refuses on SKIP.

## Constraints
- Python 3.11, stdlib-first. Match the existing hermes_parity code style (small modules, no
  classes where functions do).
- Run the suite with: HOME=/Users/alexgierczyk ~/.hermes/hermes-agent/venv/bin/python -m pytest
  tests/scripts/ -q -o addopts="" -p no:randomly — must be GREEN before you stop.
- Also run: python3.11 -m py_compile scripts/hermes_parity/*.py
- Commit locally on this branch with clear per-item commits. DO NOT push. DO NOT open a PR.
  STOP when tests are green and work is committed. The orchestrator verifies and lands.
