---
name: hermes-ops-review
description: Use when running a Hermes optimization, ops, testing, or review workflow that needs redacted status receipts, wrapper-backed gateway verification, focused tests, secret scans, build-log updates, and judge-ready evidence.
---

# Hermes Ops Review

Use this workflow for Hermes repo changes that need operational evidence without
touching private runtime state.

## Boundaries

- Preserve `ai.hermes.gateway` and `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- Do not mutate private memory, live logs, caches, provider facts, credentials,
  Docker config, historical artifacts, or launchd state unless explicitly asked.
- Do not dump raw `.env`, auth files, Keychain values, launchd environment,
  private memory, raw logs, cron prompt bodies, or `hermes status --all`.
- Treat optional provider warnings from `hermes doctor` as expected unless the
  current task is specifically about provider setup.

## Workflow

1. Read the current control-plane docs for the target phase:
   `AGENTS.md`, `docs/HERMES_EXECUTION_PLAN.md`,
   `docs/HERMES_BUILD_LOG.md`, `docs/HERMES_SECURITY_MODEL.md`, and
   `docs/HERMES_TESTING_PLAN.md`.
2. Check repo state with `git status --short --branch`; do not revert unrelated
   dirty work.
3. Capture redacted operator receipts:

```bash
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health > /tmp/hermes-ops-status.md
./venv/bin/python -m hermes_cli.main ops status --json --no-health > /tmp/hermes-ops-status.json
./venv/bin/python -m json.tool /tmp/hermes-ops-status.json > /tmp/hermes-ops-status.pretty.json
```

4. When live local health is relevant, run:

```bash
./venv/bin/python -m hermes_cli.main ops status --markdown
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
```

5. Run the smallest focused tests that cover the changed surface. For CLI or
   workflow edits, include parser/receipt tests when available.
6. Scan changed files, generated receipts, and rollback patches for secret
   patterns before reporting.
7. Update the build log and execution plan with files changed, commands run,
   validation results, known issues, fixes, and judge results.
8. Run the three-judge cycle: Architecture, Reliability/Security, and
   Tooling/UX. Fix required failures and judge again.

## Evidence To Report

- Current phase and exact slice.
- Files changed.
- Commands run.
- Test and smoke results.
- Secret-scan and rollback-check results.
- Judge results with PASS/FAIL, evidence, required fixes, optional
  improvements, and confidence.
- Remaining blockers and the next recommended prompt.
