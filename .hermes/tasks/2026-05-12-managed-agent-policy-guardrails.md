# Managed-Agent Policy Guardrails

Status: done
Owner: Hermes main / Multica Codex worker
Started: 2026-05-12 07:35
Branch: feat/managed-agent-policy-guardrails
Multica project: Hermes Agent / 966bde1c-667a-42e9-8088-56e3cbe97b79
Multica parent: JEF-222 / 014e2fe4-947f-4a9f-b772-190621f0473e
Multica implementation: JEF-223 / 1eb698b5-0177-44a2-936c-05a6f986db12
Multica review gate: JEF-224 / e6608eb6-2c1c-43a7-a3f9-a9fc1a577b6f

## Scope

Continue the Managed-Agent/Kanban worker work after the sandbox-capable worker foundation. This phase makes policy contracts explicit, visible, and testable:

- Add structured policy-contract fields to worker policy descriptors.
- Inject a machine-readable `HERMES_KANBAN_POLICY_CONTRACT` env var into spawned workers.
- Include a clear worker-policy contract section in worker context.
- Surface concise policy/capability information in CLI show output.
- Preserve honest boundary: local workers still report `os_sandbox=false` and `policy_enforced_by=contract`.

## Non-goals

- No Docker/VM/OS sandbox implementation in this phase.
- No user config mutation.
- No production deploys.
- No secrets, `.env`, tokens, DB dumps, raw media/model weights in `.hermes`.

## Affected files

- `.hermes/plans/2026-05-12-managed-agent-policy-guardrails-plan.md`
- `.hermes/tasks/2026-05-12-managed-agent-policy-guardrails.md`
- `hermes_cli/kanban_db.py`
- `hermes_cli/kanban.py`
- `tests/hermes_cli/test_kanban_db.py`
- `tests/hermes_cli/test_kanban_cli.py`

## Acceptance Criteria

- Policy descriptors/contract expose allowed/forbidden operations and enforcement level.
- Dispatcher run metadata/env includes the policy contract.
- Worker context warns/instructs correctly for `read_only`, `test_only`, `code_edit`, and `sandbox_strict`.
- CLI show/json gives operators enough policy/capability context.
- Tests prove no OS/container sandbox overclaim.
- Focused Kanban tests and existing local-patch tests pass.

## Verification plan

```bash
source venv/bin/activate
python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py -o 'addopts='
python -m pytest -q tests/test_hermes_memory_provider.py tests/agent/test_auxiliary_temperature_retry.py tests/agent/test_prompt_builder.py -o 'addopts='
git diff --check
```

## Discussion / decisions

- 2026-05-12: Dragon approved continuing next phase. Main Hermes defined the next low-risk phase as policy-contract guardrails, not OS sandboxing.
- 2026-05-12: Plan written before code so Multica worker can execute from GitHub-visible ledger.

## Status log

- 2026-05-12 07:35 — in_progress: branch created and plan/task ledger drafted.
- 2026-05-12 07:38 — in_progress: Multica trace created: parent JEF-222, implementation JEF-223, review gate JEF-224. Implementation not assigned until ledger is committed/pushed for agent visibility.
- 2026-05-12 08:19 — review: implemented policy-contract descriptors/helper, worker env injection, run metadata, worker-context guardrail section, and CLI show/json visibility. Local workers remain contract/path scoped with `os_sandbox=false`; no OS/container sandbox enforcement is claimed. Focused verification passed; ready for JEF-224 review gate.
- 2026-05-12 08:24 — review: JEF-224 read-only Codex review passed on commit `14b93331a`; reviewer reran required tests plus isolated `HERMES_HOME` CLI smoke and reported no blocking findings.
- 2026-05-12 08:28 — done: controller reran required verification on canonical branch, confirmed `git diff --check` clean, and closed Multica parent/child/review issues as complete on the feature branch. No merge to `main` performed.
- 2026-05-12 08:28 — done: copied plan and ledger backup to `/Users/jeffphoon/mnt/G/project-files/hermes-agent/.hermes/`.
- 2026-05-12 08:30 — done: contained stale comment-triggered JEF-224 run `7facac46-6584-4b63-b2cc-7ee0b3c71ec8` after closeout comment; canonical review run remains `53e6fb87-8283-4a3a-9b39-95030ffeceeb`.

## Review evidence

- JEF-223 implementation run `36bf8c12-48f0-4c96-a0f4-8abade32a82c` completed with commit `14b93331a feat: add kanban worker policy contracts`.
- JEF-224 review run `53e6fb87-8283-4a3a-9b39-95030ffeceeb` completed with `Review pass`; no code changes by reviewer.
- JEF-224 reviewer verification: Kanban DB/CLI `140 passed`; related regressions `135 passed, 1 skipped`; Kanban tools `198 passed`; `git diff --check` clean; isolated `HERMES_HOME` CLI smoke passed.

## Verification evidence

```bash
source venv/bin/activate && python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py -o 'addopts='
# 140 passed in 3.92s

source venv/bin/activate && python -m pytest -q tests/test_hermes_memory_provider.py tests/agent/test_auxiliary_temperature_retry.py tests/agent/test_prompt_builder.py -o 'addopts='
# 135 passed, 1 skipped in 1.18s

git diff --check
# clean

Controller closeout verification on 2026-05-12 08:28 +08:

source venv/bin/activate && python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py -o 'addopts='
# 140 passed in 3.99s

source venv/bin/activate && python -m pytest -q tests/test_hermes_memory_provider.py tests/agent/test_auxiliary_temperature_retry.py tests/agent/test_prompt_builder.py -o 'addopts='
# 135 passed, 1 skipped in 0.95s

git diff --check
# clean

source venv/bin/activate && python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py tests/tools/test_kanban_tools.py -o 'addopts='
# 198 passed in 6.91s
```
