# /ooo Ouroboros Native Integration Supervised Implementation Plan

> **For Hermes:** Use `subagent-driven-development` and `superpowers-dispatching-parallel-agents`. The controller session is the commander/supervisor. Do not paste this plan into chat; keep it as the working document.

**Goal:** Raise the current Discord `/ooo` Ouroboros integration from a usable skill-bridge implementation (~76/100) to a verified native command integration (95-100/100).

**Architecture:** Replace `/ooo` skill-injection routing with a gateway-owned native Ouroboros command router that calls Ouroboros MCP tools directly, persists Discord thread/user session state, and returns Discord-safe receipts/results. Keep skill bridge only as an explicit fallback for unsupported/non-native flows.

**Tech Stack:** Hermes Agent gateway, Discord adapter, Python 3.11, Ouroboros MCP server, pytest, Hermes config/MCP tooling.

---

## 0. Current Evidence Baseline

Recorded on 2026-06-22 before this plan was saved.

- [x] Current repo: `/home/tohoho/tohohowsl/HERMES/hermes-agent`
- [x] Branch: `toh/attach-light-usage-last`
- [x] Hermes config path: `/home/tohoho/tohohowsl/HERMES/.hermes/config.yaml`
- [x] Gateway running: PID 10814, manual foreground mode under tmux.
- [x] Discord connected: gateway state says `discord.state=connected`.
- [x] Ouroboros MCP enabled in Hermes config: `mcp_servers.ouroboros.enabled=true`.
- [x] `hermes mcp test ouroboros`: connected, 29 tools discovered.
- [x] Ouroboros version: `0.42.6.dev1`.
- [x] Discord API global commands: 55.
- [x] Discord API has `/ooo`: true.
- [x] `/ooo` has 21 subcommands: `help`, `interview`, `seed`, `run`, `evaluate`, `status`, `pm`, `qa`, `unstuck`, `evolve`, `ralph`, `auto`, `cancel`, `resume-session`, `setup`, `config`, `brownfield`, `publish`, `welcome`, `tutorial`, `update`.
- [x] Gateway log contains real `/ooo interview ...` invocation.
- [x] Current targeted gateway slash tests: `35 passed, 2 warnings`.
- [x] Current score estimate: `76/100`.

Current dirty state before this plan:

- [ ] `gateway/run.py`
- [ ] `gateway/slash_commands.py`
- [ ] `hermes_cli/commands.py`
- [ ] `package-lock.json`
- [ ] `plugins/platforms/discord/adapter.py`
- [ ] `tests/gateway/test_discord_slash_commands.py`
- [ ] `.ouroboros/` untracked
- [ ] `.ouroboros_eval_artifact.md` untracked

---

## 1. Non-Negotiable Completion Definition

The work is complete only when every checked claim has fresh evidence.

### Functional completion

- [ ] Discord `/ooo` command remains registered in live Discord API.
- [ ] Discord `/ooo` still exposes all intended subcommands.
- [ ] `/ooo help` returns native static help without agent loop or skill injection.
- [ ] `/ooo interview` calls `ouroboros_interview` directly.
- [ ] `/ooo pm` calls `ouroboros_pm_interview` directly.
- [ ] `/ooo seed` calls `ouroboros_generate_seed` directly.
- [ ] `/ooo run` calls `ouroboros_start_execute_seed` by default.
- [ ] `/ooo evaluate` calls `ouroboros_start_evaluate` by default.
- [ ] `/ooo qa` calls `ouroboros_qa` directly.
- [ ] `/ooo status` can show thread-local recent session/job state and explicit session/job status.
- [ ] `/ooo job status` calls `ouroboros_job_status`.
- [ ] `/ooo job wait` calls `ouroboros_job_wait`.
- [ ] `/ooo job result` calls `ouroboros_job_result`.
- [ ] `/ooo cancel --job <id>` calls `ouroboros_cancel_job`.
- [ ] `/ooo cancel --execution <id>` calls `ouroboros_cancel_execution`.
- [ ] `/ooo evolve` calls `ouroboros_start_evolve_step` by default.
- [ ] `/ooo ralph` calls `ouroboros_start_ralph` by default.
- [ ] `/ooo auto` calls `ouroboros_start_auto` by default.
- [ ] `/ooo brownfield` calls `ouroboros_brownfield` for safe read/query actions.
- [ ] `/ooo setup`, `/ooo config`, `/ooo update`, `/ooo publish` enforce stop gates before environment-changing or external/publishing actions.
- [ ] Native unsupported/fallback cases clearly say fallback reason.
- [ ] Skill-injection path is not used for native commands.

### State completion

- [ ] Gateway stores recent Ouroboros IDs by Discord context.
- [ ] Stored fields include relevant IDs: `interview_session_id`, `pm_session_id`, `auto_session_id`, `last_job_id`, `last_execution_id`, `last_lineage_id`, `last_seed_id`, `updated_at`.
- [ ] State is scoped by platform/guild/channel/thread/user so different threads do not mix.
- [ ] State survives gateway restart.
- [ ] Corrupt state file/db is handled without crashing the gateway.

### UX completion

- [ ] Discord responses are short, operational, and Korean-first where gateway context expects Korean.
- [ ] Long jobs return immediate receipts with `job_id` and next command.
- [ ] Long outputs are truncated or summarized safely; Discord message length failures are avoided.
- [ ] CLI ANSI/rich formatting is stripped before Discord display.
- [ ] Errors show command/tool, reason, and next action; no raw tracebacks by default.

### Verification completion

- [ ] `py_compile` passes for touched Python files.
- [ ] Targeted gateway tests pass.
- [ ] New native router/state tests pass.
- [ ] `hermes mcp test ouroboros` passes.
- [ ] Live Discord API confirms `/ooo` registration after gateway restart.
- [ ] Live Discord smoke passes for at least `/ooo help`, `/ooo interview`, `/ooo status`, and one job/status flow if safe.
- [ ] `git status --short --branch` is inspected and only intended changes remain.

### Score gates

- [ ] 85/100: cleanup + `/ooo run` route corrected + basic native router tests.
- [ ] 92/100: interview/pm/seed/run/evaluate/qa/status/job/cancel native MCP routes done.
- [ ] 95/100: evolve/ralph/auto/brownfield done + state persistence + live smoke.
- [ ] 100/100: all subcommands policy-complete, stop gates correct, tests/live smoke/repo state clean.

---

## 2. Stop Gates

The controller must stop and ask for explicit approval before:

- [ ] pushing to main/master
- [ ] merge/rebase that changes published history
- [ ] force-push/history rewrite
- [ ] release/tag
- [ ] destructive delete/reset
- [ ] changing Discord bot permissions or application authority
- [ ] publishing GitHub issues or any public/external posting
- [ ] running Hermes/Ouroboros update that changes installed code
- [ ] running setup/config commands that alter credentials/provider/account settings
- [ ] deploying/applying the change to VPS/Oracle/other hosts
- [ ] broad persistent automation outside this bounded task

Controller may do local file edits/tests/gateway restart for this WSL Hermes repo unless a command is destructive or external-authority-changing.

---

## 3. Supervision Model

The controller session owns:

- [ ] task decomposition
- [ ] final decisions
- [ ] exact git staging/commit/push decisions
- [ ] final verification
- [ ] live Discord smoke coordination
- [ ] stop-gate enforcement

Subagents do:

- [ ] focused code reading/investigation
- [ ] bounded implementation in assigned file set
- [ ] tests/reviews scoped to assigned area
- [ ] self-contained reports with exact files, commands, results, and blockers

Subagents must not:

- [ ] commit
- [ ] push
- [ ] restart gateway unless explicitly assigned
- [ ] run destructive git commands
- [ ] publish external content
- [ ] edit unrelated files
- [ ] claim completion without command output

---

## 4. Phase A — Read-only Parallel Reconnaissance

Use 3 subagents in parallel. This phase must be read-only except temporary scratch notes under `/tmp` if needed.

### Agent A1 — MCP invocation path reconnaissance

**Goal:** Find the correct Hermes-internal way for gateway code to call an enabled MCP tool directly.

**Scope:** Read-only.

**Files/areas to inspect:**

- `model_tools.py`
- `tools/registry.py`
- `agent/*mcp*`
- `hermes_cli/*mcp*`
- `gateway/run.py`
- existing tests involving MCP calls

**Required output:**

- [ ] exact internal API/function to call, or reason no reusable API exists
- [ ] minimal code shape for `call_ouroboros_tool(tool_name, args, timeout)`
- [ ] required imports and lifecycle concerns
- [ ] error/timeout behavior
- [ ] recommended test strategy with fake MCP client
- [ ] risks/blockers

**Completion condition:** Controller can implement MCP adapter without guessing.

---

### Agent A2 — Discord/gateway `/ooo` routing/state reconnaissance

**Goal:** Map the current `/ooo` flow and identify exact safe insertion points for native routing and state persistence.

**Scope:** Read-only.

**Files/areas to inspect:**

- `plugins/platforms/discord/adapter.py`
- `gateway/slash_commands.py`
- `gateway/run.py`
- `gateway/session.py`
- gateway state/persistence helpers
- current tests under `tests/gateway/`

**Required output:**

- [ ] current flow from Discord slash interaction to `_handle_ooo_command`
- [ ] where skill injection currently happens
- [ ] exact wrapper shape to keep `slash_commands.py` thin
- [ ] recommended state persistence location and schema
- [ ] how to scope state by Discord guild/channel/thread/user
- [ ] risks around running-agent/interaction timeouts
- [ ] tests to prove no state bleeding between threads

**Completion condition:** Controller can create router/state files and connect wrapper safely.

---

### Agent A3 — Test/cleanup/score audit

**Goal:** Identify the exact tests, cleanup actions, and score criteria needed to move from 76 to 95+ without scope creep.

**Scope:** Read-only.

**Files/areas to inspect:**

- `tests/gateway/test_discord_slash_commands.py`
- current dirty diff
- `package-lock.json` diff
- `.ouroboros/` and `.ouroboros_eval_artifact.md`
- pytest conventions in repo
- command registry tests if any

**Required output:**

- [ ] which existing tests to update
- [ ] which new tests to add
- [ ] exact commands to run
- [ ] whether `package-lock.json` should be reverted/excluded
- [ ] whether `.ouroboros*` files should be removed/excluded/kept
- [ ] final verification checklist
- [ ] score rubric with pass/fail mapping

**Completion condition:** Controller knows cleanup/test scope before implementation.

---

## 5. Phase B — Core Native Router Skeleton

Do this after Phase A results are reviewed.

### Task B1: Create native router file

**Objective:** Add a separate module for `/ooo` native routing so `gateway/slash_commands.py` stays thin.

**Files:**

- Create: `gateway/ouroboros_native.py`
- Modify: `gateway/slash_commands.py`
- Test: `tests/gateway/test_ouroboros_native.py`

**Checklist:**

- [ ] Define `OooCommand` dataclass.
- [ ] Define `OooContext` dataclass.
- [ ] Define `OooResult` dataclass.
- [ ] Implement `parse_ooo_args(raw: str) -> OooCommand`.
- [ ] Implement alias support: `h`, `?`, `init`, `resume`.
- [ ] Implement usage errors for unknown commands/flags.
- [ ] Implement `format_ooo_result` helpers.
- [ ] Keep `/ooo help` fully native.
- [ ] Wire `_handle_ooo_command` to new router for `/ooo help`.

**Completion condition:**

- [ ] `/ooo help` no longer needs skill injection.
- [ ] malformed args return usage, not traceback.
- [ ] tests cover parser/aliases/help/unknown command.

---

### Task B2: Remove accidental MCP commands from CLI fast-path

**Objective:** Prevent MCP-first commands from being routed through CLI just because a similarly named CLI subcommand exists.

**Files:**

- Modify: `gateway/slash_commands.py`
- Test: `tests/gateway/test_ouroboros_native.py`

**Checklist:**

- [ ] Remove `run` from CLI fast-path.
- [ ] Remove or reconsider `qa` from CLI fast-path if native MCP route is available.
- [ ] Keep only truly safe CLI-backed read-only commands, if any.
- [ ] Make `evaluate` explicitly MCP-only.
- [ ] Add regression tests showing `/ooo run` chooses MCP route.

**Completion condition:**

- [ ] `/ooo run` cannot accidentally call `ouroboros run` CLI path.
- [ ] test proves run/evaluate route to MCP path.

---

## 6. Phase C — MCP Adapter

### Task C1: Implement MCP tool caller

**Objective:** Provide direct MCP tool calls from gateway-owned `/ooo` router.

**Files:**

- Modify/Create: `gateway/ouroboros_native.py`
- Possibly Modify/Create: `gateway/ouroboros_mcp.py` if separation is cleaner
- Test: `tests/gateway/test_ouroboros_native.py`

**Checklist:**

- [ ] Implement `call_ouroboros_tool(tool_name: str, args: dict, timeout: float)`.
- [ ] Use Hermes-supported MCP client/registry path identified by Agent A1.
- [ ] Add fake client injection for tests.
- [ ] Normalize result to `OooResult`.
- [ ] Normalize errors: missing server, missing tool, validation, timeout, exception.
- [ ] Apply sensitive-output redaction where available.
- [ ] Strip raw tracebacks from Discord-facing message.

**Completion condition:**

- [ ] fake MCP success test passes.
- [ ] fake MCP timeout test passes.
- [ ] fake MCP validation/error test passes.
- [ ] actual `hermes mcp test ouroboros` still passes.

---

## 7. Phase D — State Persistence

### Task D1: Create Ouroboros gateway state store

**Objective:** Persist recent Ouroboros IDs per Discord context.

**Files:**

- Create: `gateway/ouroboros_state.py`
- Modify: `gateway/ouroboros_native.py`
- Test: `tests/gateway/test_ouroboros_state.py` or same native test file

**Schema:**

```json
{
  "interview_session_id": "...",
  "pm_session_id": "...",
  "auto_session_id": "...",
  "last_job_id": "...",
  "last_execution_id": "...",
  "last_lineage_id": "...",
  "last_seed_id": "...",
  "updated_at": "..."
}
```

**Checklist:**

- [ ] Define context key fields: platform, guild_id, channel_id, thread_id, user_id.
- [ ] Implement load.
- [ ] Implement atomic save.
- [ ] Implement patch/update.
- [ ] Implement corrupt file fallback with backup.
- [ ] Add tests for two separate threads/users.
- [ ] Add tests for restart persistence.

**Completion condition:**

- [ ] `/ooo status` can use recent state.
- [ ] state survives process restart in tests.
- [ ] state does not bleed between threads/users.

---

## 8. Phase E — Native Core Commands

### Task E1: Native `/ooo interview`

**Files:**

- Modify: `gateway/ouroboros_native.py`
- Modify: `gateway/ouroboros_state.py` if needed
- Test: `tests/gateway/test_ouroboros_native.py`

**Checklist:**

- [ ] `/ooo interview <initial_context>` starts new interview via `ouroboros_interview(initial_context=..., cwd=...)`.
- [ ] Store returned `session_id` as `interview_session_id`.
- [ ] `/ooo interview --session <id> <answer>` continues explicit session.
- [ ] `/ooo interview answer <text>` continues recent session.
- [ ] Missing recent session for answer returns usage/error.
- [ ] Response includes question/summary and session ID.
- [ ] No call to `_build_skill_message`.

**Completion condition:**

- [ ] native direct MCP test passes.
- [ ] continuation test passes.
- [ ] no-skill-injection regression test passes.

---

### Task E2: Native `/ooo pm`

**Checklist:**

- [ ] `/ooo pm <context>` starts PM interview via `ouroboros_pm_interview`.
- [ ] Store returned `session_id` as `pm_session_id`.
- [ ] `/ooo pm answer <text>` continues recent PM session.
- [ ] `/ooo pm generate` calls `ouroboros_pm_interview(action="generate")` with correct session.
- [ ] Missing session returns usage.

**Completion condition:**

- [ ] PM start/answer/generate tests pass.

---

### Task E3: Native `/ooo seed`

**Checklist:**

- [ ] Parse `--session <id>`.
- [ ] If omitted, use recent interview session first, PM session second.
- [ ] Parse `--force`.
- [ ] Call `ouroboros_generate_seed(session_id=..., force=...)`.
- [ ] Store seed ID/content pointer if returned.
- [ ] Ambiguity gate failure is shown clearly.

**Completion condition:**

- [ ] explicit session test passes.
- [ ] recent session fallback test passes.
- [ ] force flag test passes.

---

### Task E4: Native `/ooo run`

**Checklist:**

- [ ] Parse `--seed-path`, `--seed`, `--session`, `--max-iterations`, `--skip-qa`.
- [ ] Default to `ouroboros_start_execute_seed`.
- [ ] Add idempotency key based on Discord interaction/message context when available.
- [ ] Store returned `job_id`, `execution_id`, `session_id`.
- [ ] Missing seed returns usage unless recent seed is available.
- [ ] Return immediate receipt.

**Completion condition:**

- [ ] `/ooo run` uses start MCP tool, not CLI.
- [ ] job receipt test passes.
- [ ] missing seed test passes.

---

### Task E5: Native `/ooo evaluate`

**Checklist:**

- [ ] Parse positional `session_id artifact`.
- [ ] Parse flags `--session`, `--artifact`, `--consensus`.
- [ ] Default to `ouroboros_start_evaluate`.
- [ ] Store returned `job_id`.
- [ ] Artifact path handling is explicit; no destructive writes.
- [ ] Missing session/artifact returns usage.

**Completion condition:**

- [ ] start evaluate test passes.
- [ ] missing args test passes.

---

### Task E6: Native `/ooo qa`

**Checklist:**

- [ ] Parse `--artifact`, `--bar`, `--type`, `--threshold`.
- [ ] Call `ouroboros_qa`.
- [ ] Apply default quality bar only when safe and explicit in response.
- [ ] Return score/verdict/suggestions.

**Completion condition:**

- [ ] QA MCP call test passes.
- [ ] default quality bar behavior is tested.

---

### Task E7: Native `/ooo status` and `/ooo job`

**Checklist:**

- [ ] `/ooo status` with no args shows recent context state or health usage.
- [ ] `/ooo status --session <id>` calls `ouroboros_session_status`.
- [ ] `/ooo status --job <id>` calls `ouroboros_job_status`.
- [ ] `/ooo job status <id>` calls `ouroboros_job_status`.
- [ ] `/ooo job wait <id>` calls `ouroboros_job_wait`.
- [ ] `/ooo job result <id>` calls `ouroboros_job_result`.
- [ ] no explicit job ID may fallback to recent `last_job_id` for status/result, but must say it did.

**Completion condition:**

- [ ] all job/status tests pass.
- [ ] recent fallback tests pass.

---

### Task E8: Native `/ooo cancel`

**Checklist:**

- [ ] `/ooo cancel --job <job_id>` calls `ouroboros_cancel_job`.
- [ ] `/ooo cancel --execution <execution_id>` calls `ouroboros_cancel_execution`.
- [ ] `/ooo cancel` with no explicit ID does not cancel automatically.
- [ ] Response tells exact command to confirm cancellation by ID.

**Completion condition:**

- [ ] explicit job cancel test passes.
- [ ] explicit execution cancel test passes.
- [ ] no-ID safety test passes.

---

## 9. Phase F — Loop and Advanced Commands

### Task F1: Native `/ooo evolve`

**Checklist:**

- [ ] Parse `--lineage`, `--seed`, `--project-dir`, `--max` where applicable.
- [ ] Default to `ouroboros_start_evolve_step`.
- [ ] Store `job_id` and `lineage_id`.
- [ ] Require lineage ID unless a safe recent lineage exists.

**Completion condition:**

- [ ] start evolve test passes.

---

### Task F2: Native `/ooo ralph`

**Checklist:**

- [ ] Parse `--lineage`, `--max-generations`, `--timeout`.
- [ ] Call `ouroboros_start_ralph`.
- [ ] Store `job_id` and lineage.
- [ ] Enforce bounded max/defaults; no unbounded loop.

**Completion condition:**

- [ ] start ralph test passes.

---

### Task F3: Native `/ooo auto`

**Checklist:**

- [ ] Parse `--goal`, `--resume`, `--cwd`, `--domain`, `--complete-product`.
- [ ] Call `ouroboros_start_auto`.
- [ ] Store `job_id` and `auto_session_id`.
- [ ] Return status command.

**Completion condition:**

- [ ] start auto test passes.

---

### Task F4: Native `/ooo brownfield`

**Checklist:**

- [ ] Parse safe actions: list/query/default/scan.
- [ ] Call `ouroboros_brownfield`.
- [ ] For register/set-default, require explicit path/name.
- [ ] Do not mutate defaults from vague command.

**Completion condition:**

- [ ] safe query test passes.
- [ ] mutation-without-explicit-args test blocks.

---

### Task F5: Safety-gated setup/config/update/publish

**Checklist:**

- [ ] `/ooo config` returns current safe info/usage, not uncontrolled GUI or settings mutation.
- [ ] `/ooo setup` returns doctor/setup guidance unless explicitly approved.
- [ ] `/ooo update` checks version or gives approval-required message; does not upgrade automatically.
- [ ] `/ooo publish` defaults to preview/dry-run; does not create GitHub issues without explicit approval.

**Completion condition:**

- [ ] tests prove no side effect without explicit approval.

---

## 10. Phase G — Verification and Live Smoke

### Local commands

Run from `/home/tohoho/tohohowsl/HERMES/hermes-agent`.

- [ ] `./venv/bin/python -m py_compile gateway/ouroboros_native.py gateway/ouroboros_state.py gateway/slash_commands.py gateway/run.py plugins/platforms/discord/adapter.py hermes_cli/commands.py`
- [ ] `./venv/bin/python -m pytest tests/gateway/test_discord_slash_commands.py tests/gateway/test_ouroboros_native.py tests/gateway/test_ouroboros_state.py -q -o addopts=`
- [ ] `hermes mcp test ouroboros`
- [ ] `hermes doctor`
- [ ] `git diff --stat`
- [ ] `git status --short --branch`

### Live Discord smoke

- [ ] Restart gateway from outside the running gateway process if needed.
- [ ] Query Discord API and confirm `/ooo` exists.
- [ ] Query Discord API and confirm expected subcommands.
- [ ] Run `/ooo help` in Discord.
- [ ] Run `/ooo interview 테스트용 요구사항 정리` in Discord.
- [ ] Confirm response contains no skill document/frontmatter.
- [ ] Confirm response contains question/result and session ID if returned by MCP.
- [ ] Run `/ooo status` in same thread.
- [ ] If a safe background job exists, run `/ooo job status <job_id>`.

### Completion condition

- [ ] All verification commands are fresh and recorded.
- [ ] Live smoke proves Discord side, not just unit tests.
- [ ] Final score can be assigned with evidence.

---

## 11. Cleanup Policy

- [ ] Revert or exclude `package-lock.json` unless a reviewed explanation shows it belongs to this task.
- [ ] Do not commit `.ouroboros/` or `.ouroboros_eval_artifact.md` unless explicitly justified.
- [ ] Do not let subagents commit.
- [ ] Controller stages exact approved files only after inspection.
- [ ] Controller asks before commit/push if not already explicitly approved for those side effects.

---

## 12. Final Report Template

Use this only after verification, and keep it concise in chat.

```text
1#

실행 단위:
- Discord /ooo Ouroboros native integration

변경:
- <file list>

검증:
- py_compile: <result>
- pytest: <result>
- hermes mcp test ouroboros: <result>
- Discord API /ooo: <result>
- live Discord smoke: <result>

점수:
- 이전: 76/100
- 현재: <XX>/100

미완료/stop gate:
- <none or list>

다음:
- <commit/PR/deploy/etc. if approved>
```

---

## 13. First Dispatch Wave

The first wave is read-only reconnaissance to avoid parallel agents editing the same files and corrupting the working tree.

- [ ] A1 MCP invocation path reconnaissance dispatched.
- [ ] A2 Discord/gateway routing/state reconnaissance dispatched.
- [ ] A3 Test/cleanup/score audit dispatched.

After A1-A3 summaries return, controller updates this plan if needed and dispatches implementation agents in non-overlapping slices.
