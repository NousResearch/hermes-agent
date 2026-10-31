---
name: github-actions
description: Author secure and reproducible GitHub Actions workflows.
version: 1.1.0
author: Tugrul Guner (@tugrulguner), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, CI/CD, Actions, Workflows, Automation, Security]
    related_skills: [github-auth, github-pr-workflow, github-repo-management]
---

# GitHub Actions Skill

Author and revise secure, reproducible GitHub Actions workflow YAML. This skill
covers workflow design, permissions, immutable dependencies, matrices, caching,
artifacts, and concurrency. It does not cover run diagnosis or secret
management; use `github-pr-workflow` and `github-repo-management` for those
operations.

## When to Use

- Create or modify `.github/workflows/*.yml`.
- Design CI matrices, job dependencies, path filters, or manual inputs.
- Add caches, artifacts, release jobs, or deployment environments.
- Audit a workflow for excessive permissions or unsafe event handling.
- Replace floating action tags with immutable commit SHAs.

Do not use this skill merely to inspect a failed run or manage repository secrets. Load the related GitHub skill instead so operational procedures have one source of truth.

## Prerequisites

- Use `read_file` to inspect `CONTRIBUTING.md`, `AGENTS.md`, and existing workflows before editing.
- Use `search_files` to find repository test, lint, build, and package-manager commands; never invent them.
- Use `github-auth` when authenticated GitHub access is required.
- Preserve the repository's established runner versions and action pins unless the task explicitly upgrades them.

GitHub-hosted runners execute Linux, macOS, or Windows jobs, so this authoring skill is available on every supported Hermes platform.

## How to Run

1. Inspect the repository's existing workflow conventions and required checks.
2. Identify the smallest event, permissions, and path scope that satisfies the task.
3. Resolve each external action version to a full 40-character commit SHA.
4. Write the workflow using real repository commands and deterministic dependency installation.
5. Validate YAML, action pins, expression placement, and changed-file scope locally.
6. Push for review; never merge or enable a deployment automatically unless explicitly requested.

## Quick Reference

| Concern | Preferred pattern |
|---|---|
| Permissions | top-level `permissions: contents: read`; elevate per job |
| Dependencies | full action commit SHA plus trailing version comment |
| Pull requests | `pull_request`; avoid privileged `pull_request_target` |
| Repeated pushes | `concurrency` with `cancel-in-progress: true` |
| Matrices | `fail-fast: false` when every result matters |
| Dependencies | lockfile-backed install such as `npm ci` or project equivalent |
| Secrets | environment variables, never direct shell interpolation |
| Deployments | protected GitHub environment with explicit job permissions |
| Artifacts | short retention and non-secret contents |
| Time bounds | set `timeout-minutes` on every job |

## Procedure

### 1. Discover Repository Commands

Read existing automation before drafting:

- package manifests and lockfiles
- existing workflows and reusable workflows
- contribution instructions
- test scripts and required check names
- supported runtime versions

A workflow must call the same commands contributors run locally. Do not replace an existing package manager, test runner, or build path merely to make the YAML shorter.

### 2. Choose the Safest Event

Use the narrowest trigger that covers the intended lifecycle:

```yaml
name: CI

on:
  pull_request:
    branches: [main]
    paths:
      - "src/**"
      - "tests/**"
      - "pyproject.toml"
      - ".github/workflows/ci.yml"
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

Treat `pull_request_target` as privileged. Code from a fork must not be checked out and executed in that event while write permissions or secrets are available.

### 3. Pin Every External Action

A tag such as `actions/checkout@v6` is mutable. Use a complete commit SHA and retain a version comment for update tooling and reviewers:

```yaml
steps:
  - name: Check out source
    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2

  - name: Set up Python
    uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405 # v6.2.0
    with:
      python-version: "3.11"
```

Resolve pins from the action's upstream release or repository and verify that the commit belongs to the expected publisher. Never shorten the SHA.

### 4. Define a Minimal CI Job

Substitute only commands confirmed from the target repository:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Check out source
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2

      - name: Set up Python
        uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405 # v6.2.0
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run tests
        run: scripts/run_tests.sh tests/ -q
```

If the repository does not use `uv` or `scripts/run_tests.sh`, replace those lines with its actual documented commands.

### 5. Add a Matrix Only When It Proves Compatibility

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ["3.11", "3.12"]

runs-on: ${{ matrix.os }}
timeout-minutes: 30
```

Avoid multiplying jobs that run identical code without validating a supported platform or runtime. Quote version-like YAML values so `3.10` is not parsed as a number.

### 6. Cache Derived Dependencies, Not Secrets

Prefer setup actions' built-in cache support when available. For a custom cache, derive keys from lockfiles:

```yaml
- name: Restore dependency cache
  uses: actions/cache@0400d5f644dc74513175e3cd8d07132dd4860809 # v4.2.4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

Never cache credentials, signing keys, `.env` files, or untrusted executable output that a later privileged job consumes.

### 7. Pass Untrusted Context Through Environment Variables

Do not inject pull-request titles, branch names, issue bodies, or other attacker-controlled expressions directly into shell source.

```yaml
- name: Report pull request title
  env:
    PR_TITLE: ${{ github.event.pull_request.title }}
  run: printf '%s\n' "$PR_TITLE"
```

This keeps expression expansion in data rather than executable shell syntax. Quote shell variables and use structured command arguments where possible.

### 8. Upload Bounded Artifacts

```yaml
- name: Upload test report
  if: always()
  uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
  with:
    name: test-report-${{ github.run_id }}
    path: reports/
    if-no-files-found: error
    retention-days: 7
```

Review artifact paths for secrets and excessive size. Do not upload the workspace root, home directory, Git credentials, or environment files.

### 9. Gate Deployments

Deployment jobs should:

- depend on successful build and test jobs
- use a protected GitHub environment
- request only the permission they need
- use OIDC instead of long-lived cloud credentials when supported
- deploy an identified artifact rather than rebuilding mutable source

```yaml
jobs:
  deploy:
    needs: [test, build]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    timeout-minutes: 20
    environment: production
    permissions:
      contents: read
      id-token: write
    steps:
      - name: Check out deployment scripts
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2

      - name: Download reviewed artifact
        uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          name: deploy-package
          path: dist/

      - name: Deploy reviewed artifact
        run: ./scripts/deploy.sh dist/
```

The job-level permission elevation must not leak into untrusted build or test jobs.

## Pitfalls

- **Floating tags:** `@v4` and branch references are mutable; pin a full commit SHA.
- **Broad permissions:** omit `write-all`; grant one permission at the narrowest job scope.
- **Privileged fork execution:** never execute fork code with `pull_request_target` secrets.
- **Expression injection:** pass untrusted `${{ github.event.* }}` data through `env`.
- **Invented commands:** derive install, test, and build commands from the repository.
- **Cache poisoning:** do not restore untrusted executable caches into privileged jobs.
- **Secret artifacts:** inspect exact upload paths and use short retention.
- **Runaway jobs:** set `timeout-minutes` and concurrency cancellation.
- **YAML coercion:** quote runtime versions, permissions, and values that resemble numbers.
- **Duplicate operations:** defer run diagnosis and secret management to related GitHub skills.

## Verification

- [ ] The workflow file parses as YAML and appears under `.github/workflows/`.
- [ ] Every external `uses:` reference has a 40-character SHA and version comment.
- [ ] Install, lint, test, build, and deploy commands exist in the repository.
- [ ] Top-level permissions are read-only or empty; write permissions are job-scoped.
- [ ] Fork-originated code cannot access secrets or write-capable tokens.
- [ ] Untrusted GitHub context is passed as data, not interpolated into shell source.
- [ ] Jobs have appropriate timeouts and concurrency behavior.
- [ ] Cache and artifact paths contain no credentials or environment files.
- [ ] Deployment jobs use protected environments and reviewed artifacts.
- [ ] The repository's workflow-specific tests and lint checks pass.
