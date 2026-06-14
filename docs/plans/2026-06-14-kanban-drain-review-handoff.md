# Kanban Drain Review Handoff

## Scope

This branch implements the kanban drain controller slices documented in
`docs/plans/2026-06-13-kanban-drain-controller-source-of-truth.md`.

Implemented surfaces:

- Read-only drain classification report for nonterminal board cards.
- Structured `block_class` and `block_metadata` persistence on blocked cards.
- `review_required` routing from `blocked` to `review`.
- Review Packet PASS/FAIL consumption with required evidence validation.
- `timeout_gave_up` reslicing into idempotent parent-linked child tasks.
- `superseded_duplicates` archive handling with canonical evidence checks.
- Optional review-profile cap override reporting for dispatcher dry runs.

## Safety Boundaries

- Unknown, human-held, credential-held, and prod-risk cards remain report-only.
- Drain commands are dry-run by default and require explicit `--apply`.
- PASS consumption completes only source cards still blocked for review-required.
- FAIL consumption creates scoped rework without marking the source done.
- Timeout/gave-up reslice preserves source status and failure counters.
- Superseded archive requires canonical evidence, duplicate evidence, and no unique acceptance criteria.
- Cap override is opt-in and scoped to review rows; ordinary ready rows retain the normal dispatcher cap behavior.

## Verification

Fresh local commands run on 2026-06-14:

```bash
scripts/run_tests.sh tests/hermes_cli/test_kanban_drain.py tests/hermes_cli/test_kanban_db.py
```

Result: blocked before collection because `.venv/bin/python` has no `pytest` module.

```bash
env -i PATH="$PATH" HOME="$HOME" TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONHASHSEED=0 \
  venv/bin/python scripts/run_tests_parallel.py \
  tests/hermes_cli/test_kanban_drain.py tests/hermes_cli/test_kanban_db.py
```

Result: 200 tests passed, 0 failed.

```bash
venv/bin/python -m py_compile \
  hermes_cli/kanban_db.py hermes_cli/kanban_drain.py hermes_cli/kanban.py \
  tests/hermes_cli/test_kanban_drain.py tests/hermes_cli/test_kanban_db.py
```

Result: exit 0.

```bash
git diff --check
```

Result: exit 0.

## Remaining Unproven Boundaries

- Full repository test suite was not run.
- `scripts/run_tests.sh` currently selects `.venv` first; this checkout's `.venv` lacks `pytest`.
- No live board dry-run/apply was executed against production kanban data.
- No remote CI or PR checks have run yet.
