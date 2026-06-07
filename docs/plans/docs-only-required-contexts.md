# Docs-only required contexts

The `main` branch ruleset requires a fixed set of status contexts before a PR can merge. Required contexts must not be skipped just because a PR is docs-only; otherwise GitHub reports the PR as blocked even when the produced docs-safe checks are green.

## Policy

- Required check names stay stable.
- Docs-only PRs run the same required contexts that branch protection names.
- Workflows may internally no-op when their scanner has no relevant files, but the required context must be emitted.
- Do not add fake runtime file changes to trigger checks.
- Do not weaken branch protection or remove required contexts to merge docs-only PRs.

## Current alignment

These required-context workflows run for docs-only pull requests:

- `Lint (ruff + ty)` emits `ruff + ty diff`, `ruff enforcement (blocking)`, and `Windows footguns (blocking)`.
- `Contributor Attribution Check` emits `check-attribution`.
- `Tests` emits `test (1)` through `test (6)` and `e2e`.
- `Supply Chain Audit` emits `Scan PR for critical supply chain risks` and `Check PyPI dependency upper bounds`.

This keeps docs-only PRs on the normal protected path without claiming runtime, launch, production, or protected approval readiness.
