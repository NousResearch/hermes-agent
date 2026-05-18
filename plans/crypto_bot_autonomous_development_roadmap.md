# Crypto Bot Autonomous Development Roadmap

## Goal

Bring `crypto_bot` to supervised autonomous development under Hermes, with the
default Hermes profile acting as operator/supervisor and `crypto-bot-pm` acting
as the PM worker lane. The end state is branch-local autonomous development
with evidence gates, Codex sidecar audit, native Hermes Kanban lifecycle, and
Gitea CI/CD evidence. Remote mutation, PR creation, runner actions, and merge
remain gated until each authority is explicitly enabled and verified.

## Current Verified State

- `crypto-bot-pm` profile exists and uses `gpt-5.5` through `openai-codex`.
- `crypto-bot-pm` plugin and skill are installed in the profile.
- Codex sidecar wrapper exists at `/Users/preston/.local/bin/hermes-codex-audit`.
- Control-plane assets were found on a legacy staging branch in a separate
  Hermes checkout. That branch/location is not the target operating model.
- The active Hermes control-plane location is
  `/Users/preston/.hermes/hermes-agent` on its normal branch.
- `/Users/preston/robinhood/crypto_bot` is on
  `hermes/dev13-006-daemon-trust-contract-mapping` at `8be208b...`.
- The product repo current branch does not need to carry PM script source for
  the Hermes PM bridge; the source has been consolidated under the active
  `crypto-bot-pm` plugin.
- Latest Kanban import audit artifact classifies the board as valid with
  90 cards and 101 dependencies, recommends the S006 PR pilot path, and keeps
  S007A blocked.

## Phase 0: Run Hermes Outside The Codex Sandbox

Objective: stop treating this Codex sandbox as the control plane.

The immediate blocker is not a missing scheduler. It is that this session cannot
write to the real Hermes home, Hermes control-plane checkout, or `crypto_bot`
worktree. Adding cron/watchdog infrastructure before the base control plane is
proven functional would add technical debt around an unstable surface.

Near-term operating model:

- Run the default Hermes profile from a normal terminal or service context, not
  from this sandbox.
- Keep `crypto-bot-pm` as the worker profile with the same Codex OAuth config as
  the default profile.
- Use the default profile to run the PM/control-plane recovery and verification
  commands directly.
- Do not add new cron jobs, launch daemons, or periodic watchdogs until the
  read-only PM tools, Kanban checks, self-check, and readiness checks pass from
  the unsandboxed Hermes runtime.

Expected exit: default Hermes can read/write the required local paths under the
normal user account, and the control-plane recovery work can proceed without this
Codex session needing elevated filesystem access.

## Phase 1: Restore PM Control Plane

Objective: make read-only PM status tools operational from one Hermes checkout
and one active `crypto_bot` branch, with no runtime dependency on
`/Users/preston/hermes`, `hermes/control-plane-stabilization`, or a PM tooling
sidecar worktree.

1. Port the required Hermes control-plane assets into the active Hermes checkout
   at `/Users/preston/.hermes/hermes-agent` on its normal branch.
2. Keep PM bridge script source under
   `/Users/preston/.hermes/hermes-agent/plugins/crypto-bot-pm/scripts/hermes_pm`
   so the plugin does not depend on product-branch PM scripts or bytecode in
   `__pycache__`.
3. Keep `/Users/preston/hermes` and `hermes/control-plane-stabilization` out of
   the runtime path. After consolidation is verified, that staging checkout can
   be ignored or removed.
4. Run `/Users/preston/.hermes/hermes-agent/tools/install_user_assets.py` to
   refresh default runtime assets from the active checkout.
5. Re-run `scripts/setup_crypto_bot_pm_profile.sh` to sync the profile-local
   plugin and skill.
6. Smoke-test from the active profile:
   - `crypto_bot_pm_status`
   - `crypto_bot_pm_development_workstream`
   - `crypto_bot_pm_development_slice`
7. Run:
   - `python3 tools/crypto_bot_control_plane_self_check.py --format json`
   - `python3 tools/crypto_bot_autonomy_readiness.py --format json`

Expected exit: plugin tools succeed; self-check no longer fails on runtime
asset drift or missing plugin backing scripts.

Helper:

```bash
cd /Users/preston/.hermes/hermes-agent
scripts/prepare_crypto_bot_autonomy_control_plane.sh
```

The helper now enforces this single-source policy. It does not switch branches
or create a sidecar worktree; it exits with a missing-assets list if the active
branches have not been consolidated yet.

## Phase 2: Default Profile Supervises `crypto-bot-pm`

Objective: default runs the oversight loop; `crypto-bot-pm` handles PM evidence.

Default profile duties:

- Keep the standing objective active in chat or TUI.
- Run read-only reality checks before dispatch.
- Assign PM/control-plane tasks to `crypto-bot-pm`.
- Review worker evidence before unblocking downstream tasks.
- Ask the operator for exact approval only when policy requires it.

Worker profile duties:

- Report current PM status through plugin tools.
- Comment Kanban evidence and blockers.
- Stop before secrets, Gitea writes, workflow/runner actions, PR creation,
  push, merge, runtime service actions, or trading surfaces.

Do not dispatch product implementation cards until the readiness tools agree
that local autonomy is allowed and S006 remote lifecycle no longer blocks the
next task.

## Phase 3: S006 Remote Lifecycle

Objective: close the S006 PR/CI lifecycle before S007A or later product work.

Required checks before any pilot:

- `crypto_bot_control_plane_self_check.py` passes.
- `crypto_bot_kanban_import_audit.py` still reports 90 cards and 101 links.
- `crypto_bot_pr_ci_audit.py` confirms current PR state without mutation.
- `crypto_bot_autonomy_readiness.py` says the next action is to request the
  exact S006 PR pilot decision.

Pilot rules:

- Use only the Hermes-owned Gitea PR pilot adapter.
- Run adapter self-check, dry-run, and create-pr-only preflight with the same
  runtime and command shape.
- Require exact operator approval before any PR creation attempt.
- Do not push again if the exact approved remote branch SHA already exists.
- After pilot, use read-only discovery to prove PR existence, PR number, source
  branch/head, target branch, and CI state.

## Phase 4: Gitea CI/CD Readiness

Objective: make CI evidence reliable before autonomous remote progression.

Gates:

- Gitea API is reachable read-only.
- PR status/check APIs are readable.
- Workflow files validate locally.
- Local action mirrors are present or explicitly justified.
- Runner labels and container/network contracts are verified.
- Runner recovery remains gated by exact operator approval.
- No token material is printed or embedded in artifacts.

Runner recovery, if authorized, starts with:

```bash
python3 /Users/preston/.hermes/hermes-agent/tools/crypto_bot_gitea_runner_recovery.py --inspect
```

Execution requires the helper's exact approval phrase and must not start
workflows, mutate PR metadata, edit workflow files, merge, or touch product
runtime files.

## Phase 5: Branch-Local Autonomous Development

Objective: enable safe local implementation loops after S006 remote lifecycle
is closed.

Loop:

1. Default selects a ready Kanban card and verifies dependencies.
2. `crypto-bot-pm` records selected-task source and allowlisted paths.
3. Worker creates a non-protected local branch.
4. Worker edits only allowlisted non-secret files.
5. Worker runs targeted validators, `git diff --check`, and `ruff check` where
   appropriate.
6. Worker commits locally.
7. Worker renders a sidecar audit prompt and runs
   `/Users/preston/.local/bin/hermes-codex-audit --mode audit-readonly`.
8. Worker runs `crypto_bot_completion_gate.py`.
9. Completion is allowed only on a `PASS` gate and matching post-commit sidecar
   evidence.

## Phase 6: Controlled Remote Automation

Objective: move from read-only remote evidence to limited remote operations only
after policy enables each authority.

Authority ladder:

1. Read-only Gitea discovery.
2. PR evidence packet generation.
3. Controlled remote branch push.
4. One PR creation pilot.
5. PR update/comment/status mutation.
6. CI runner/workflow recovery.
7. Merge-to-main.

Each rung needs its own policy flag, exact command shape, preflight, evidence
packet, and operator approval until the project policy explicitly promotes it.

## Phase 7: Merge And Release Automation

Objective: eventually allow autonomous merge/release only after CI and merge
gates are proven reliable.

Prerequisites:

- Current CI green evidence on the exact PR head.
- Merge readiness dry-run passes.
- Protected branch policy is understood and documented.
- No unresolved evidence issues.
- Operator has explicitly enabled merge authority.
- Post-merge verification exists and is read-only safe.

Until then, autonomous development may prepare evidence and recommendations but
must not merge, deploy, publish, or touch live trading/runtime surfaces.
