# Handoff: UA Tier 1 T1-002 Supabase Migration Markers Complete

## Context

- Branch: `feat/ua-tier1-static-signals`
- Prior bead commit: `502fbc9ff feat(code-scan): add tier1 static signals schema`
- Bead: `.beads/ua-tier1-002-supabase-migration-markers.md`
- Execution mode: Codex/gpt-5.5 via `codex exec -m gpt-5.5 --dangerously-bypass-approvals-and-sandbox` after sandboxed Codex was blocked by `bwrap: loopback: Failed RTM_NEWADDR` and JC asked to continue serial execution.

## Work Completed

- Added `extract_supabase_migration_markers(rel_path, content, max_per_type=50)` in `scripts/code-scan/static_signals.py`.
- Added strict tests in `tests/code_scan/test_static_signals.py`.
- Marker inventory is line-oriented heuristic matching only; no SQL AST parsing and no policy/security validation.
- Required marker types covered:
  - `enable_rls`
  - `create_policy`
  - `drop_policy`
  - `using_clause`
  - `with_check_clause`
  - `auth_uid`
  - `auth_role`
  - `anon_role`
  - `authenticated_role`
  - `permissive_true`
  - `security_definer`
  - `service_role`
  - `grant_statement`
  - `revoke_statement`
  - `create_function`
- Scope preserved: no `run_ua.py`, report, context, dependency, or external target repo edits.

## Verification

- Codex RED:
  - `python -m pytest tests/code_scan/test_static_signals.py -q`
  - Failed before implementation with `ImportError: cannot import name 'extract_supabase_migration_markers'`.
- Hermes RED reconstruction:
  - Restored `scripts/code-scan/static_signals.py` from `HEAD` while retaining the new tests.
  - `python -m pytest tests/code_scan/test_static_signals.py -q`
  - Result: expected failure, `T1_002_RED_EXIT=2`, ImportError for missing `extract_supabase_migration_markers`.
- GREEN focused:
  - `python -m pytest tests/code_scan/test_static_signals.py -q`
  - Result: PASS, `15 passed in 0.31s`.
- FULL:
  - `python -m pytest tests/code_scan -q`
  - Result: PASS, `1061 passed in 140.00s (0:02:20)`.
- Compile:
  - `python -m py_compile scripts/code-scan/static_signals.py`
  - Result: PASS.
- Diff hygiene:
  - `git diff --check`
  - Result: PASS.
- Static/test-quality scan on added lines:
  - `hardcoded_secret=0`
  - `shell_injection=0`
  - `eval_exec=0`
  - `unsafe_deserialization=0`
  - `sql_format_injection=0`
  - `vacuous_or_true=0`
  - `explicit_placeholder_terms=0`
  - `STATIC_AND_TEST_QUALITY_SCAN_PASS`
- Diff artifact:
  - `/tmp/ua-tier1-artifacts/ua-tier1-002-supabase-migration-markers-diff.patch`
  - `291 lines / 11868 bytes`.

## Reviewer

- Independent reviewer verdict: PASS.
- Blockers: none.
- Reviewer notes: implementation is spec-compliant, evidence-boundary preserving, strict enough in tests, scope-clean, and safe to commit.

## Next Recommended Action

Commit and push T1-002, then begin T1-003 only after the push succeeds.
