# AION PR Intake Adapter v0.1 surface pack / AION PR 入口适配器 v0.1 表层包

## 中文摘要

本表层包把 PR intake 标准化为 Hermes 原生 Kanban 语义的静态合约：GitHub label 与 PR metadata 只能作为 intake mirror（入口镜像），不得被解释为执行触发。真正的执行触发必须同时具备 Hermes Kanban `task.assignee` 与 dispatcher pickup/run 证据。GitHub 继续作为正式 ledger/evidence surface；GitHub Actions 只能运行 CI/checkers，不承载业务流调度。

## English Summary

This surface pack standardizes PR intake as a static contract over native Hermes Kanban semantics: GitHub labels and PR metadata are intake mirrors only and must not be interpreted as execution triggers. Execution requires Hermes Kanban `task.assignee` plus dispatcher pickup/run evidence. GitHub remains the formal ledger/evidence surface; GitHub Actions may run CI/checkers only and must not dispatch business flow.

## Contract / 合约

A valid PR intake packet must satisfy all of the following:

1. `github_pr_metadata.intake_mirror_only` is `true`.
2. `github_pr_metadata.label_claims_execution` is `false`.
3. `execution_trigger.type` is exactly `hermes_kanban_task_assignee_and_dispatcher_pickup`.
4. `execution_trigger.kanban_task.assignee` is non-empty.
5. `execution_trigger.dispatcher_pickup.picked_up` is `true` and has non-empty `run_id` and `claimed_by`.
6. `audit.verdict_required_before_merge` is `true` and `audit.current_head_verdict` is one of `PASS`, `CONDITIONAL_PASS`, or `BLOCK`.
7. Every protected-surface / runtime / production flag remains `false`.
8. `non_claims.full_unattended_ready` is `false`.

## Trigger semantics / 触发语义

| Surface | Allowed role | Not allowed |
| --- | --- | --- |
| GitHub label | Intake mirror, routing hint, ledger pointer | Execution source, scheduler, business-flow dispatcher |
| PR metadata/body | Evidence mirror and audit request | Runtime enablement, Required Mode enablement, source-writeback activation |
| Hermes Kanban task.assignee | Executor assignment | Audit PASS, merge permission, production authorization |
| Hermes dispatcher pickup/run evidence | Execution trigger and executor-of-record evidence | Runtime/gateway/cron/live scanner activation |
| GitHub Actions | CI/checker execution | Business-flow dispatch or actor scheduling |

## Fail-closed cases / 失败关闭覆盖

The checker and fixtures intentionally fail closed for:

- label-only execution claim / 仅凭 label 声称执行
- missing `task.assignee` / 缺少任务 assignee
- missing dispatcher pickup / 缺少 dispatcher pickup 证据
- missing current-head audit verdict / 缺少当前 head 审计结论
- forbidden runtime flag / 禁止的 runtime 标志
- full-unattended overclaim / full unattended 过度声明

## Risk boundary / 风险边界

This pack is static docs/schema/checker/fixtures/tests evidence only. It does not add or activate a runtime, gateway, cron job, Required Mode, live scanner, source writeback, production/payment/database/webhook/secret/customer-data access, external executor sandbox, real API execution, issue closure, merge, or full unattended-ready declaration.

## Evidence artifacts / 证据产物

- Contract schema: `schemas/aion_pr_intake_adapter_v0_1.schema.json`
- Checker: `scripts/aion_pr_intake_adapter_v0_1_check.py`
- Fixtures: `tests/fixtures/aion-pr-intake-adapter-v0.1/`
- Tests: `tests/test_aion_pr_intake_adapter_v0_1_check.py`
- Static evidence packet: `docs/governance/aion-pr-intake-adapter-v0.1.evidence.json`

## Audit request / 审计请求

Audit target: `bafuxunan / 八府巡按`. Please audit whether this PR preserves the #446/#447 standard: label/PR metadata are mirrors only; Hermes Kanban task assignment plus dispatcher pickup is the execution trigger; GitHub Actions are CI/checkers only; all runtime/production/full-unattended boundaries remain false.
