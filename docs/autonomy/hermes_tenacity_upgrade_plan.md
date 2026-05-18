# Hermes Tenacity Upgrade Plan

## Snapshot

- Upgrade snapshot root: `/Users/preston/.local/state/hermes-operator/tenacity-upgrades/20260514T165811Z`
- Hermes source root: `/Users/preston/.hermes/hermes-agent`
- Active product repo: `/Users/preston/robinhood/crypto_bot`
- Source backup for obsolete Hermes backup file: `/Users/preston/.local/state/hermes-operator/hermes-source-backups/20260514T165811Z`
- Official Hermes runtime backup: `/Users/preston/.hermes/backups/pre-update-2026-05-14-115856.zip`

## Before State

- Hermes source branch observed at session start: `main`
- Expected Hermes autonomy branch from forensics: `hermes/gitea-ci-pr-lifecycle`
- Hermes autonomy branch HEAD used for this work: `1619d3421ae3f9ef683cc8ad2d0e49460b112402`
- Hermes Agent version before update: `Hermes Agent v0.13.0 (2026.5.7)`
- Hermes Agent installed HEAD before update: `faa13e49f81480771ceeb55991bb0c27edf1a5fb`
- Hermes update check before update: `644 commits behind origin/main`
- Gateway before update: captured in `hermes-gateway.before.txt`

## Update Method

The official updater was used:

```bash
hermes update --backup
```

The updater created a pre-update archive, pulled 644 commits, updated Python and
Node dependencies, rebuilt the web UI, synced bundled skills, verified config,
and restarted the Hermes gateway service `ai.hermes.gateway`.

No manual destructive update path was used.

## After State

- Hermes Agent version after update: `Hermes Agent v0.13.0 (2026.5.7)`
- Hermes Agent installed HEAD after update: `b08f53a75893ec4dfa6c470e9f27bc039fce6f07`
- Hermes update check after update: `1 commit behind origin/main`
- OpenAI SDK after update: `2.24.0`
- Gateway after update: service loaded and running with PID recorded in `hermes-gateway.after.txt`
- Config check after update: config version `23`, up to date
- Bundled skills after update: `kanban-worker` and `kanban-orchestrator` present

## Gateway Restart And Resume Observation

The official update reported:

```text
Service restarted
Restarted ai.hermes.gateway
```

The post-update gateway status reported the launchd service loaded and running.
Gateway auto-resume semantics were not exercised by sending Telegram messages,
because this session was not authorized to send or edit Telegram messages.

## Feature Availability Checks

- Persistent `/goal`: available as a local documented slash command in
  `website/docs/user-guide/features/goals.md` and slash-command reference.
- Kanban: `hermes kanban --help` exposes durable SQLite-backed boards,
  dependencies, assignment, comments, dispatch, runs, logs, heartbeats, and
  assignee discovery.
- Worker lanes: `kanban-worker` and `kanban-orchestrator` bundled skills were
  found under `/Users/preston/.hermes/skills/devops/`.
- Hooks: `hermes hooks --help` exposes shell hook listing, test, doctor, and
  consent allowlist management; local docs also describe plugin hooks for
  `pre_tool_call`, `pre_llm_call`, `transform_terminal_output`, and
  `transform_tool_result`.
- Codex runtime: the Hermes custom wrapper
  `/Users/preston/.local/bin/hermes-codex-audit` remains the bounded audit
  runtime for crypto_bot until a native equivalent is explicitly proven.

## Reconciled Readiness Semantics

- Missing historical dev13 product artifacts
  `docs/development/hermes_coding_work_packet_template.md` and
  `docs/implementation/hermes_pm_checkpoint_13b_plan.md` are warnings under the
  Tenacity-native control plane unless a future selected task proves they are
  current global safety prerequisites.
- Missing `scripts/validation/validate-security-evidence-wrapper.py` is a
  task-scoped validator warning by default. It becomes a blocker only when the
  selected task explicitly requires that validator.
- Global validator blockers remain hard blockers for
  `scripts/validation/validate-secrets-discipline.sh` and
  `scripts/validation/validate-governance-baseline.sh`.
- Source/runtime parity for installed custom skills, plugin, and wrapper is a
  hard blocker.
- Native Kanban import has not been performed. In the current phase this means
  `ready_for_next_task` remains false, while a clean preview may make
  `ready_to_request_board_import` true.

## Remaining Remote Blockers

- PR evidence for S006 is locally ready, but no PR exists.
- CI/check evidence is absent for the S006 HEAD.
- Runner status and branch protection visibility require additional authority.

## Rollback

To roll back the installed Hermes Agent runtime to the official pre-update
archive:

```bash
hermes import /Users/preston/.hermes/backups/pre-update-2026-05-14-115856.zip
```

If a source-level rollback is required for the installed agent checkout:

```bash
git -C /Users/preston/.hermes/hermes-agent checkout faa13e49f81480771ceeb55991bb0c27edf1a5fb
/Users/preston/.hermes/hermes-agent/venv/bin/python -m pip install -e /Users/preston/.hermes/hermes-agent
hermes gateway restart
```

If a Hermes autonomy-source rollback is required:

```bash
git -C /Users/preston/.hermes/hermes-agent switch hermes/gitea-ci-pr-lifecycle
git -C /Users/preston/.hermes/hermes-agent reset --hard 1619d3421ae3f9ef683cc8ad2d0e49460b112402
```

Do not run the `reset --hard` command unless the Operator explicitly requests a
source rollback.

To reinstall custom runtime assets from Hermes source after rollback or repair:

```bash
cd /Users/preston/.hermes/hermes-agent
python3 tools/install_user_assets.py --format json
```
