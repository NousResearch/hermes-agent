# Oflow local CI safety and isolation model

This document describes the local CI pilot added for Oflow review branches. The
workflow is intentionally narrow: it gives reviewers repeatable repository-side
validation evidence without touching runtime systems, credentials, production
services, providers, or databases.

## Runner isolation posture

- Manual-only trigger: `.github/workflows/oflow-local-ci-safety.yml` uses
  `workflow_dispatch` only. It is not attached to `push`, `pull_request`,
  schedules, deployments, or release events.
- Read-only repository permission: workflow permissions are limited to
  `contents: read`, and checkout uses `persist-credentials: false`.
- Branch guard: the job exits on `main`/`master` so the workflow cannot be used
  as evidence for direct main mutation.
- Concurrency guard: only one run per ref stays active; newer manual reruns
  cancel stale runs for the same branch.
- Job bound: the validation job has an explicit timeout so a runner cannot hang
  indefinitely.
- Artifact-only outputs: review evidence is written under
  `artifacts/oflow-local-ci/` and uploaded as a GitHub Actions artifact. The
  workflow does not create comments, labels, commits, deployments, releases, or
  status mutations beyond the Actions run itself.

## Local-only checks

The helper at `scripts/review/oflow_local_ci.py` provides reusable, read-only
commands for:

- denylist scanning of changed files;
- Python `py_compile` over changed Python files;
- `git diff --check` whitespace validation;
- workflow YAML structure validation;
- focused pytest subsets for local fixture tests;
- CI summary artifact generation.

The denylist treats the following as hard failures when they appear in changed
operational files: `.env` access, deploy script invocation, service-manager
restart commands such as `systemctl`, `docker compose up`/`docker compose down`,
SSH, and sqlite/postgres mutation commands. Secret-bearing paths are reported
without reading their contents.

## Runtime and production hard stops

This pilot is repository validation only. It must not:

- merge;
- deploy;
- restart services;
- inspect `.env` files;
- inspect secrets or credentials;
- monitor runtime systems;
- probe production;
- call model/provider or trading APIs;
- SSH to any host;
- access databases;
- migrate, backfill, or mutate data;
- affect trading/order flow.

The summary artifact records these hard stops explicitly so PR handoffs can cite
machine-readable evidence.

## Future approval gates

If Oflow later needs broader CI behavior, add a separate approval layer instead
of widening this pilot in place. Future work should require an explicit human
approval gate before any capability that can mutate runtime state, register or
manage runners, use credentials, call providers, access databases, deploy,
restart services, or probe production.
