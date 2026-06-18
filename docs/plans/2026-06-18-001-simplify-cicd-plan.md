---
title: "ci: Simplify CI/CD into PR Gate, Main Deploy, and Release Publish"
status: active
date: 2026-06-18
type: ci
target_repo: hermes-agent
origin: user request after CI/CD audit: reduce workflow sprawl and unblock continuous deployment
---

# ci: Simplify CI/CD into PR Gate, Main Deploy, and Release Publish

## Summary

Hermes Agent's CI/CD is intentionally defensive, but the current shape makes continuous deployment harder than it needs to be. The repo has many active workflows where PR validation, post-merge deployment, scheduled maintenance, and release publishing are mixed across separate files and GitHub's Actions UI also shows additional dynamic/legacy workflows.

The simplification target is not weaker validation. The target is a smaller operational surface:

1. **PR Gate** — one required PR workflow that answers: "Can this change enter main?"
2. **Main Deploy** — one post-merge workflow that answers: "What did current main deploy?"
3. **Release Publish** — one release workflow that answers: "What user-facing artifacts were published?"

Everything else should either be folded into one of those three, become a job inside them, or be explicitly documented as a GitHub-managed/dynamic workflow that is not part of Hermes' deploy train.

---

## Current Inventory

### Current branch rulesets

GitHub rulesets currently protect `main` through two active branch rulesets:

| Ruleset | Enforcement | Relevant rule | Current required signal |
|---|---|---|---|
| `protect-main` | active | required status checks | `test (1)` through `test (6)`, `check-attribution`, `ruff + ty diff`, `ruff enforcement (blocking)`, `Windows footguns (blocking)`, `Scan PR for critical supply chain risks`, `Check PyPI dependency upper bounds`, `check-common-ancestor` |
| `dep-version-gate` | active | pull request reviewer rule | Team approval for dependency/version files: `pyproject.toml`, `uv.lock`, `flake.lock`, `**/package.json`, `**/package-lock.json` |

This means the first simplification target is concrete: collapse the thirteen required status contexts in `protect-main` into one Hermes-owned `PR Gate` context, while preserving the separate dependency-review ruleset.

### PR validation workflows

These currently run on PRs and contribute to the large required-check surface:

| Workflow | Current file | Current role | Simplification destination |
|---|---|---|---|
| Tests | `.github/workflows/tests.yml` | Python tests, sharded via `scripts/run_tests_parallel.py`, plus e2e job | `pr-gate.yml` |
| Lint (ruff + ty) | `.github/workflows/lint.yml` | Advisory ruff/ty diff, blocking ruff, Windows footgun scan | `pr-gate.yml` |
| Typecheck | `.github/workflows/typecheck.yml` | TS package typecheck matrix plus desktop build | `pr-gate.yml` |
| Docs Site Checks | `.github/workflows/docs-site-checks.yml` | Docusaurus / skills docs build, currently PR-wide | `pr-gate.yml` with path-aware job |
| Docker / shell lint | `.github/workflows/docker-lint.yml` | hadolint + shellcheck | `pr-gate.yml` with Docker path-aware job |
| Docker Build and Publish | `.github/workflows/docker-publish.yml` | PR Docker build/smoke/integration plus main/release publish | Split: PR jobs to `pr-gate.yml`; publish jobs to `main-deploy.yml` / `release.yml` |
| uv.lock check | `.github/workflows/uv-lockfile-check.yml` | `uv lock --check` against merged PR state | `pr-gate.yml` |
| Supply Chain Audit | `.github/workflows/supply-chain-audit.yml` | Critical diff scanner, dependency bounds, MCP catalog label gate | `pr-gate.yml` |
| OSV-Scanner | `.github/workflows/osv-scanner.yml` | lockfile vulnerability scan, non-blocking findings | `pr-gate.yml` for PR status; scheduled Security tab scan can remain separate only if explicitly desired |
| History Check | `.github/workflows/history-check.yml` | reject unrelated-history PRs | `pr-gate.yml` |
| Contributor Attribution Check | `.github/workflows/contributor-check.yml` | contributor email mapping check | `pr-gate.yml` |

### Post-merge / deployment workflows

| Workflow | Current file | Current role | Simplification destination |
|---|---|---|---|
| Deploy Site | `.github/workflows/deploy-site.yml` | GitHub Pages docs deploy; Vercel hook on release/manual | `main-deploy.yml` for Pages, `release.yml` for production release hook |
| Build Skills Index | `.github/workflows/skills-index.yml` | scheduled/manual skills-index artifact and deploy-site trigger | `main-deploy.yml` scheduled job or explicit maintenance job inside same workflow |
| Skills Index Freshness Check | `.github/workflows/skills-index-freshness.yml` | scheduled stale-index issue signal | keep only if it remains a clear maintenance signal; otherwise fold into skills-index job summary |
| Docker Build and Publish | `.github/workflows/docker-publish.yml` | main/release Docker Hub publish | `main-deploy.yml` for main images; `release.yml` for tag images |

### Release / artifact workflows

| Workflow | Current file | Current role | Simplification destination |
|---|---|---|---|
| Publish to PyPI | `.github/workflows/upload_to_pypi.yml` | CalVer tag/manual PyPI publish + Sigstore + GitHub Release attach | `release.yml` |
| Build Windows Installer | `.github/workflows/build-windows-installer.yml` | admin-only manual signed Windows installer | `release.yml` as manual/admin-gated job |

### GitHub-managed or dynamic workflows visible in Actions UI

GitHub reports additional active workflows that are not present in `origin/main:.github/workflows`, e.g. CodeQL, Dependabot, Copilot, and other dynamic/legacy entries. These should not be counted as Hermes-owned deploy train files unless their source path exists in repo.

Action: document them separately as **GitHub-managed/dynamic**, then avoid making deploy decisions from the raw Actions workflow count alone.

---

## Problems to Solve

### P1. PR validation and deployment are mixed

`docker-publish.yml` currently owns both PR Docker validation and main/release Docker publication. That makes the workflow hard to reason about and makes the PR gate look like a deployment pipeline.

**Target:** PR validation belongs in `pr-gate.yml`; publication belongs in `main-deploy.yml` or `release.yml`.

### P2. Many workflows exist only to keep required checks from going pending

Several files intentionally avoid PR path filters because required checks can remain pending when a path-gated workflow does not run. The reason is valid, but the result is a broad CI surface.

**Target:** use one always-present `pr-gate.yml` with internal `classify` and path-aware jobs. Required check points to `PR Gate`, not every specialized workflow.

### P3. Main Docker publish serializes continuous deployment

Main/release Docker runs are intentionally not cancelled so every main merge gets an image. This is safe but creates a 8-16 minute post-merge queue during rapid merges.

**Target decision needed:** either preserve one-image-per-main-commit or switch to a deploy train that publishes the latest approved main on a cadence/manual dispatch. Do not hide this trade-off inside YAML.

### P4. External `action_required` runs pollute deploy visibility

Fork/approval-required PRs can show many `action_required` checks. These are not the same as failed trusted CI and should be separated in any deploy dashboard/report.

**Target:** `PR Gate` summary should classify `action_required` / untrusted PR state separately from failing validated CI.

### P5. Lockfile checks are correct but train-hostile

`uv.lock check` runs against GitHub's merged PR state, which is the right safety property. In rapid merge trains it can force follow-up PRs to rebase/regenerate `uv.lock`.

**Target:** isolate dependency/lockfile PRs in the train and allow only one dependency-touching PR in the ready-to-merge lane at a time.

---

## Target Architecture

```text
.github/workflows/
  pr-gate.yml          # required PR check surface
  main-deploy.yml      # post-merge deployment from main
  release.yml          # tag/release/manual artifact publishing
```

Optional exceptions must justify why they cannot be jobs inside those three:

- GitHub-managed dynamic workflows: CodeQL, Dependabot, Copilot.
- Low-frequency maintenance signal that cannot be represented in `main-deploy.yml` without confusing deploy state.

---

## Proposed `pr-gate.yml` Shape

```yaml
name: PR Gate

on:
  pull_request:
    branches: [main]

jobs:
  classify:
    outputs:
      python: ...
      frontend: ...
      docs: ...
      docker: ...
      deps: ...
      mcp_catalog: ...

  history:
  contributor:
  supply-chain:
  dependency-bounds:
  uv-lock:
  lint:
  windows-footguns:
  typecheck:
  tests:

  docs:
    if: needs.classify.outputs.docs == 'true'

  docker-lint:
    if: needs.classify.outputs.docker == 'true'

  docker-smoke:
    if: needs.classify.outputs.docker == 'true' || needs.classify.outputs.python == 'true'

  summary:
    if: always()
```

Important: the `summary` job must fail if any required internal job failed, and succeed when a non-applicable job was intentionally skipped. That preserves a single required check without path-gated pending behavior.

---

## Proposed `main-deploy.yml` Shape

```yaml
name: Main Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  classify:
  docker-publish:
    if: needs.classify.outputs.docker_or_python == 'true'
  docs-pages:
    if: needs.classify.outputs.docs_or_skills == 'true'
  skills-index:
    if: github.event_name == 'schedule' || needs.classify.outputs.skills == 'true'
  summary:
```

Open decision: keep one Docker image per main commit, or batch/cancel superseded main runs. If continuous deployment throughput is the priority, make this an explicit release-train policy rather than an accidental consequence of `cancel-in-progress`.

---

## Proposed `release.yml` Shape

```yaml
name: Release Publish

on:
  release:
    types: [published]
  push:
    tags: ['v20*']
  workflow_dispatch:

jobs:
  docker-release:
  pypi:
  sigstore:
  github-release-assets:
  vercel-production-hook:
  windows-installer:
    if: inputs.build_windows_installer == 'true'
```

Release publishing should be the only place where PyPI, release-tag Docker images, Sigstore artifacts, and optional signed installer artifacts are considered together.

---

## Migration Plan

### Phase 1 — Inventory and branch-protection alignment

1. Land this plan.
2. Export current branch protection / required check names from GitHub settings.
3. Classify existing checks as:
   - required PR gate,
   - advisory PR signal,
   - deploy,
   - release,
   - maintenance,
   - GitHub-managed/dynamic.
4. Decide the final required check name: `PR Gate`.

Stop condition: do not delete existing workflows until branch protection can be updated safely.

### Phase 2 — Introduce `pr-gate.yml` without removing old workflows

1. Add `pr-gate.yml` with internal jobs copied from the existing workflows.
2. Keep old workflows active temporarily.
3. Verify `PR Gate` returns the same blocking result as the old required set for at least several PRs.
4. Update branch protection to require `PR Gate` instead of the old per-workflow checks.

Stop condition: if `PR Gate` summary cannot faithfully propagate internal failures, do not remove old required checks.

### Phase 3 — Split deploy concerns

1. Move main Docker publish jobs out of `docker-publish.yml` into `main-deploy.yml`.
2. Move docs Pages deploy out of `deploy-site.yml` into `main-deploy.yml`.
3. Keep release-specific Docker/PyPI/signing behavior in `release.yml`.
4. Verify main push deploys are still observable from one workflow summary.

Stop condition: no release publishing behavior should change silently.

### Phase 4 — Delete or archive old workflows

After branch protection and deploy observability are proven:

- remove `tests.yml`, `lint.yml`, `typecheck.yml`, `docs-site-checks.yml`, `docker-lint.yml`, `uv-lockfile-check.yml`, `history-check.yml`, `contributor-check.yml`, `supply-chain-audit.yml` after their jobs are represented inside `pr-gate.yml`;
- remove PR responsibilities from `docker-publish.yml`, then remove the file after deploy behavior is represented in `main-deploy.yml` / `release.yml`;
- remove `deploy-site.yml`, `skills-index.yml`, `upload_to_pypi.yml`, and `build-windows-installer.yml` after equivalent jobs exist and have passed live dry-runs.

---

## Acceptance Criteria

- GitHub branch protection has at most one Hermes-owned required PR workflow check: `PR Gate`.
- PR authors can tell from one workflow summary whether the PR is blocked by tests, docs, Docker, deps, security, external approval, or contributor attribution.
- Main deploy status is visible from one workflow: `Main Deploy`.
- Release publish status is visible from one workflow: `Release Publish`.
- Docker publish, Pages deploy, PyPI publish, Sigstore signing, and Windows signing still have real successful runs after migration.
- The old workflow count in `.github/workflows` is reduced substantially, not replaced with more files.

---

## Non-goals

- Do not weaken supply-chain protections.
- Do not remove SHA pinning for GitHub Actions.
- Do not change dependency pinning policy.
- Do not make Docker publish behavior less observable.
- Do not hide failing checks behind a green summary job.
- Do not update branch protection blindly from code; branch protection changes must be coordinated with repository settings.
