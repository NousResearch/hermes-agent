# AgentOps 完整自愈平台 Luna 分阶段实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each approved phase task-by-task. This document is the program plan; every phase requires its own task-level implementation plan and explicit authorization before code changes begin.

**Goal:** 分阶段交付一个独立、可审计、可灰度和可回滚的 Hermes AgentOps 自愈控制面，同时确保任何阶段都不会提前获得下一阶段的写权限。

**Architecture:** 独立 `agentopsd` Sidecar 负责 Fleet、采集、事故、策略、修复、验证和发布；Hermes 插件只提供 Bridge、CLI 和 Dashboard 接入。第一版使用本地 SQLite WAL 与 Unix Domain Socket，所有自动写操作由结构化 ActionPlan、Policy Engine、Target Lease、验证和回滚共同约束。

**Tech Stack:** Python 3、Hermes General Plugin、FastAPI-compatible dashboard plugin API、SQLite WAL、pytest、macOS launchd、Git worktree、现有 Hermes Dashboard/Plugin/Kanban/Cron 能力。

## Global Constraints

- Phase 0/1 已获本轮任务授权，且仅可在隔离 worktree 实施；Phase 2-7 仍为规划，未经新的书面授权不得创建或修改实现文件。
- 禁止覆盖、清理、提交或重置当前工作区内任何用户已有修改。
- 控制面必须独立于 Gateway 进程和模型可用性。
- 第一版存储固定为本地 SQLite WAL；不引入 Postgres、Redis、NATS 或云端控制面。
- 控制 API 默认只使用 Unix Domain Socket；localhost TCP 必须显式开启并认证。
- 所有权限默认 `observe_only`；升级和迁移不得自动开启写权限。
- LLM 只能输出符合 Schema 的建议，不能直接执行 Shell、发布或写生产数据。
- R1 仅允许逐项审核的结构化 Runbook；R2 仅在隔离 worktree；R3 仅部署不可变 Artifact；R4 每次人工审批。
- 所有动作必须有前置条件、幂等键、Target Lease、超时、验证、审计和回滚。
- 插件不得加入 AgentOps 专用核心分支；若通用 Hook 不足，通用扩展必须独立提交和审阅。
- Feishu 数据操作优先确定性 CLI、官方 API 或 Connector；UI 自动化只能作为明确说明原因的受限回退。
- 每个阶段使用独立 `codex/agentops-phase-<n>-<name>` 或用户指定的分支；不得直接在当前 dirty `main` 上实施。
- 每个阶段必须先提交 task-level implementation plan，再由 Molly 授权、Luna 实施、Codex 审阅。

---

## 1. 计划定位

这是一份项目级交付计划，不是一条授权执行的命令。平台包含多个可独立验收的子系统，若把全部内容写成一次性代码任务，会导致：

- 一次变更跨越 R0-R4 权限；
- 事故存储、策略和执行器无法分别审阅；
- 还没验证检测准确度就提前开放自动修复；
- dirty deployment 问题尚未解决就进入自动发布；
- Luna 难以获得清晰、有限的任务上下文。

因此每个 Phase 都是独立子项目。后一个 Phase 只能在前一个 Gate 有书面结论后开始。

## 2. 角色与责任

| 角色 | 责任 | 无权事项 |
|---|---|---|
| Molly | 冻结产品决策、批准阶段启动、审批 R3/R4 和权限扩大 | 审批不能绕过硬安全不变量 |
| Luna | 编写阶段 task plan、实现、测试、迁移、运行证据和交接 | 不得自行开启下一阶段权限 |
| Codex | 审阅规格、代码、测试、威胁控制、迁移、验证和回滚 | 审阅结论不替代 Molly 的生产审批 |
| AgentOps | 在已批准范围内采集或执行 | 不拥有未登记通用权限 |

## 3. 总体时间与依赖

| Phase | 名称 | 预计工程时间 | 依赖 | 门禁 |
|---|---|---:|---|---|
| 0 | 规格冻结与基线盘点 | 2-3 天 | 无 | G0 |
| 1 | Control Plane Foundation | 1-2 周 | G0 | G1 |
| 2 | Observer 与 Fleet Coverage | 1 周 | G1 | G2 |
| 3 | Incident Ops 与 Dashboard | 1-2 周 + 7 天观察 | G2 | G3 |
| 4 | Safe R1 Remediation | 1-2 周 | G3 | G4 |
| 5 | R2 Code Repair Sandbox | 1-2 周 | G4 | G5 |
| 6 | R3 Release、Canary 与 Rollback | 1-2 周 | G5 + 发布基线 | G6 |
| 7 | Domain Review Packs | 1-2 周 | G3；写权限依风险另行批准 | G7 |

总工程周期预计 6-10 周，7 天只读观察是独立的自然时间门禁，不能用模拟数据替代。

## 4. Phase 0：规格冻结与基线盘点

### 目标

冻结术语、范围、首批 Target、当前控制器、当前部署版本和权限矩阵，确保 Luna 不在错误基线上开始实施。

### 已有规格文件

- `docs/superpowers/specs/2026-08-09-agentops-self-healing-platform-prd.md`
- `docs/superpowers/specs/2026-08-09-agentops-self-healing-technical-architecture.md`
- `docs/superpowers/specs/2026-08-09-agentops-self-healing-security-threat-model.md`
- `docs/superpowers/plans/2026-08-09-agentops-self-healing-luna-phased-plan.md`

### Luna 在获批后交付

- [ ] 输出首批 Target 清单：五个 Hermes Profile、对应 launchd Label、日志目录、仓库和关键 Cron。
- [ ] 输出所有现有 watchdog、agentops、Cron 的职责矩阵，标注读取者和写控制者。
- [ ] 为每个 Target 标记 `criticality` 和初始 `authority_mode=observe_only`。
- [ ] 输出当前部署 SHA、dirty 状态、上游差异和运行路径，不修改现状。
- [ ] 将此前历史问题映射到 Probe、Regression Test、Runbook 或 Manual-only Policy。
- [ ] 输出 Phase 1 的任务级实施计划，列出每个新文件、接口、测试和提交边界。

### G0 审阅证据

- 四份规格无相互矛盾。
- 首批 Target 和控制器职责没有遗漏。
- “哪一个控制器可以写哪一个 Target”有唯一答案。
- R1-R4 默认权限得到 Molly 书面确认。
- 当前用户修改被列为受保护资产。

### 退出结果

`G0: approved`、`G0: changes_requested` 或 `G0: rejected`。只有 `approved` 可进入 Phase 1。

## 5. Phase 1：Control Plane Foundation

### 目标

交付无修复能力的独立控制面骨架、稳定领域类型、事件契约、SQLite Store、UDS 健康 API 和安全启动模式。

### 计划文件边界

```text
plugins/agentops/plugin.yaml
plugins/agentops/__init__.py
plugins/agentops/cli.py
plugins/agentops/control/__init__.py
plugins/agentops/control/daemon.py
plugins/agentops/control/config.py
plugins/agentops/control/models.py
plugins/agentops/control/events.py
plugins/agentops/control/store.py
plugins/agentops/control/api.py
plugins/agentops/control/audit.py
tests/plugins/agentops/unit/test_config.py
tests/plugins/agentops/unit/test_models.py
tests/plugins/agentops/unit/test_events.py
tests/plugins/agentops/unit/test_store.py
tests/plugins/agentops/unit/test_audit.py
tests/plugins/agentops/contract/test_control_api.py
tests/plugins/agentops/integration/test_daemon_restart.py
```

### 必须产出的接口

```python
load_agentops_config(path: Path) -> AgentOpsConfig
open_store(config: AgentOpsConfig) -> AgentOpsStore
append_event(event: EventEnvelope) -> AppendResult
append_audit(event: AuditEvent) -> int
get_health() -> ControlPlaneHealth
run_daemon(config: AgentOpsConfig, stop_event: threading.Event) -> int
```

### Luna 在获批后交付

- [ ] 先写 Event、Store、配置默认拒绝和审计链的失败测试。
- [ ] 实现 Schema v1 和 canonical JSON hash。
- [ ] 实现 SQLite migration、WAL、busy timeout、事务和备份前置检查。
- [ ] 实现 event spool 的写入、重放、幂等和 quarantine。
- [ ] 实现 UDS `/v1/health`，不得包含写端点。
- [ ] 实现 `agentops doctor --json` 的无模型自检。
- [ ] 验证 daemon 重启不丢事件、不重复事件、不改变权限。
- [ ] 输出迁移、恢复、卸载和 launchd 安装设计；G1 前不实际安装 launchd。

### 禁止能力

- Target 写操作。
- LLM Reviewer。
- 任意 Shell 执行。
- Dashboard 审批。
- 自动安装 launchd。

### G1 审阅证据

- Event/API/Store 契约测试通过。
- 配置缺失、迁移失败和审计异常时启动为 `observe_only`。
- 合成 Secret 不进入数据库和日志。
- event spool 重放保持幂等。
- 数据库备份和恢复测试成功。
- 无 Gateway、无模型时 `/v1/health` 正常。

### 回滚

停止测试 daemon，删除测试 launchd 草案但不触碰现有服务，恢复 Phase 1 前的 AgentOps 数据库快照。Hermes Gateway 不应因 Phase 1 回滚发生变化。

## 6. Phase 2：Observer 与 Fleet Coverage

### 目标

在无写权限下纳管首批 Target，稳定采集日志、进程、launchd、Cron、SQLite 和 Git 信号。

### 计划文件边界

```text
plugins/agentops/bridge.py
plugins/agentops/control/registry.py
plugins/agentops/control/cursors.py
plugins/agentops/control/redaction.py
plugins/agentops/control/collectors/base.py
plugins/agentops/control/collectors/logs.py
plugins/agentops/control/collectors/processes.py
plugins/agentops/control/collectors/launchd.py
plugins/agentops/control/collectors/cron.py
plugins/agentops/control/collectors/sqlite_health.py
plugins/agentops/control/collectors/git_state.py
plugins/agentops/review_packs/runtime_core/manifest.yaml
tests/plugins/agentops/unit/test_registry.py
tests/plugins/agentops/unit/test_cursors.py
tests/plugins/agentops/unit/test_redaction.py
tests/plugins/agentops/contract/test_collector_protocol.py
tests/plugins/agentops/integration/test_log_rotation.py
tests/plugins/agentops/integration/test_fleet_inventory.py
```

### 必须产出的接口

```python
register_target(spec: TargetSpec) -> Target
record_target_snapshot(snapshot: TargetSnapshot) -> None
collect(target: Target, cursor: Cursor | None) -> CollectionBatch
redact_signal(signal: RawSignal, policy: RedactionPolicy) -> Signal
commit_collection(batch: CollectionBatch) -> None
```

### Luna 在获批后交付

- [ ] 为五个 Hermes Profile 建立稳定 Target ID 和只读快照。
- [ ] 实现日志 inode/offset 游标、轮转和截断恢复。
- [ ] 实现进程与 launchd 采集，不执行 restart/stop/start。
- [ ] 实现 Cron“运行状态 + 业务断言”分离的数据模型。
- [ ] 实现 SQLite/WAL 与 Git dirty/upstream 只读采集。
- [ ] 实现 Bridge 事件失败时的本地有界缓冲；Bridge 故障不得影响 Gateway。
- [ ] 用合成 Token、Cookie、用户内容验证双层脱敏。
- [ ] 输出所有 Target 的首份 Fleet Snapshot 和覆盖报告。

### G2 审阅证据

- 五个首批 Profile 资产覆盖率 100%。
- 一个日志错误跨多个日志文件不会重复生成相同 Signal。
- 日志轮转、截断和重启后游标准确。
- Cron 退出码 0、业务断言失败时产生 unhealthy Signal。
- Collector 超时或崩溃不阻止其他 Collector。
- Bridge 关闭时 Gateway 对话路径不受影响。
- 所有 Target 仍为 `observe_only`。

### 回滚

停用 Bridge 与 Collector 配置，保留只读证据导出；不改动任何被监控 Target。

## 7. Phase 3：Incident Ops 与 Dashboard

### 目标

把 Signal 聚合成可管理 Incident，提供 Review Agent 建议、Dashboard、日报周报，并完成至少 7 天只读准确度评估。

### 计划文件边界

```text
plugins/agentops/control/incidents/fingerprint.py
plugins/agentops/control/incidents/correlator.py
plugins/agentops/control/incidents/state_machine.py
plugins/agentops/control/incidents/service.py
plugins/agentops/control/review/schema.py
plugins/agentops/control/review/reviewer.py
plugins/agentops/control/reporting/digest.py
plugins/agentops/control/reporting/notifier.py
plugins/agentops/dashboard/manifest.json
plugins/agentops/dashboard/plugin_api.py
plugins/agentops/dashboard/agentops.js
plugins/agentops/dashboard/agentops.css
tests/plugins/agentops/unit/test_fingerprint.py
tests/plugins/agentops/unit/test_incident_state_machine.py
tests/plugins/agentops/unit/test_correlator.py
tests/plugins/agentops/unit/test_review_schema.py
tests/plugins/agentops/integration/test_incident_lifecycle.py
tests/plugins/agentops/integration/test_dashboard_proxy_auth.py
```

### 必须产出的接口

```python
fingerprint(signal: Signal, version: int = 1) -> str
correlate(signal: Signal) -> IncidentDecision
transition_incident(incident_id: str, to_state: IncidentState, reason: str) -> Incident
review_incident(evidence: EvidenceBundle) -> ReviewProposal
render_daily_digest(window: TimeWindow) -> Digest
```

### Luna 在获批后交付

- [ ] 实现稳定指纹和跨 Target 时间关联。
- [ ] 实现 Incident 状态机、合并、拆分、抑制、重开和通知节流。
- [ ] 实现 Reviewer 的脱敏输入、Schema 输出和预算熔断。
- [ ] 模型不可用时，事故停在可理解状态，规则流程继续。
- [ ] Dashboard 只代理 UDS，不保存长期控制 Token，不实现聊天界面。
- [ ] 生成每日摘要、每周趋势和“应沉淀资产”列表。
- [ ] 使用历史日志回放和人工标注集合评估聚合质量。
- [ ] 线上只读观察至少 7 个连续自然日。

### G3 审阅证据

- 已标注事故集合聚合准确率不低于 90%。
- 注入 P0/P1 故障漏报率为 0。
- 通知按 Incident 去重，没有逐日志风暴。
- 历史 `NoneType not iterable`、Gateway 重复实例、MCP 假绿和断连能形成正确事故。
- 恶意日志指令不会形成可执行动作。
- 模型故障不影响采集、事故状态和日报生成。
- 连续 7 天没有任何写 Target 动作。

### 回滚

停用 Dashboard tab 和 Reviewer，保留 Fleet 与 Signal；Incident 数据可只读导出，不影响 Gateway。

## 8. Phase 4：Safe R1 Remediation

### 目标

只对白名单、确定性、可逆的运维问题开放自动修复，建立 Policy、Approval、Lease、Executor、Verification、熔断和 Kill Switch。

### 计划文件边界

```text
plugins/agentops/control/policy/engine.py
plugins/agentops/control/policy/approvals.py
plugins/agentops/control/policy/kill_switch.py
plugins/agentops/control/remediation/actions.py
plugins/agentops/control/remediation/leases.py
plugins/agentops/control/remediation/orchestrator.py
plugins/agentops/control/remediation/executors/ops.py
plugins/agentops/control/verification/checks.py
plugins/agentops/control/verification/service.py
plugins/agentops/runbooks/runtime/*.yaml
tests/plugins/agentops/unit/test_policy_matrix.py
tests/plugins/agentops/unit/test_approvals.py
tests/plugins/agentops/unit/test_leases.py
tests/plugins/agentops/unit/test_kill_switch.py
tests/plugins/agentops/fault_injection/test_registered_restart.py
tests/plugins/agentops/fault_injection/test_stale_lock.py
tests/plugins/agentops/fault_injection/test_retry_circuit_breaker.py
tests/plugins/agentops/fault_injection/test_rollback_failure.py
```

### 必须产出的接口

```python
decide(plan: ActionPlan, context: PolicyContext) -> PolicyDecision
acquire_lease(target_id: str, owner: str, ttl_seconds: int) -> Lease
execute_plan(plan_id: str, approval: ApprovalReceipt | None) -> Execution
verify_execution(execution_id: str) -> VerificationSummary
rollback_execution(execution_id: str) -> RollbackResult
set_kill_switch(scope: KillSwitchScope, enabled: bool, actor: Actor) -> None
```

### 首批候选 Runbook

每个候选必须单独审阅，列表不构成自动执行授权：

1. `restart_registered_launchd_service`
2. `clear_stale_lock_for_dead_owner`
3. `checkpoint_sqlite_wal_with_threshold`
4. `rotate_registered_log`
5. `quarantine_optional_integration`

### Luna 在获批后交付

- [ ] 先实现默认拒绝 Policy 决策表和全部拒绝路径测试。
- [ ] 实现 plan hash、ApprovalReceipt、过期/消费/撤回。
- [ ] 实现 Target Lease、fencing token 和幂等执行。
- [ ] 实现注册动作，不接受任意 Shell 字符串。
- [ ] 为每个 Runbook 写前置条件、验证、回滚和故障注入。
- [ ] 实现全局/风险/Target/动作 Kill Switch。
- [ ] 实现修复失败两次后的自动熔断和 `manual_only` 降级。
- [ ] 先在完全模拟 Target 运行，再对一个明确非关键 Target 申请有限 canary。

### G4 审阅证据

- 未登记动作、Target、参数和 Runbook 全部拒绝。
- 竞争执行只有一个有效 fencing token。
- Kill Switch 阻止新写执行但保留采集。
- 动作退出 0 但强制验证失败时不标记 resolved。
- 重试次数有界，失败后熔断。
- 回滚失败触发 P0 和 Target Kill Switch。
- Owner + Reviewer 对每个开放的 R1 Runbook 留下批准记录。

### 回滚

关闭 `global_write_enabled`，撤回所有 R1 Runbook 授权，将 Target 退回 `observe_only`。保留执行和验证审计。

## 9. Phase 5：R2 Code Repair Sandbox

### 目标

允许 Luna 或代码 Agent 在隔离 worktree 中复现、搜索上游、补失败测试、生成最小 patch 和证据包；不允许部署。

### 计划文件边界

```text
plugins/agentops/control/remediation/executors/code_sandbox.py
plugins/agentops/control/remediation/worktrees.py
plugins/agentops/control/remediation/upstream_search.py
plugins/agentops/control/remediation/test_selector.py
plugins/agentops/control/remediation/evidence_bundle.py
tests/plugins/agentops/unit/test_worktree_boundaries.py
tests/plugins/agentops/unit/test_command_templates.py
tests/plugins/agentops/integration/test_reproduce_patch_verify.py
tests/plugins/agentops/integration/test_dirty_tree_protection.py
tests/plugins/agentops/security/test_sandbox_secrets.py
```

### 必须产出的接口

```python
create_repair_worktree(repo: Path, base_sha: str, execution_id: str) -> Worktree
run_reproduction(worktree: Worktree, spec: ReproductionSpec) -> ReproductionResult
search_upstream(issue: IncidentContext) -> tuple[UpstreamCandidate, ...]
run_test_plan(worktree: Worktree, plan: TestPlan) -> TestReport
build_evidence_bundle(execution_id: str) -> EvidenceBundle
```

### Luna 在获批后交付

- [ ] 验证 base SHA 存在且当前部署快照未变化。
- [ ] 创建仅位于 AgentOps worktrees 根下的隔离目录。
- [ ] 禁止访问无关凭证、生产写接口和其他工作区。
- [ ] 上游搜索只生成候选，不自动 cherry-pick。
- [ ] 修复流程强制要求“失败测试 → 最小修复 → 目标测试 → 影响回归”。
- [ ] EvidenceBundle 包含复现、diff、测试、上游候选、风险和回滚建议。
- [ ] 输出 branch/patch 交给 Luna 和 Codex，不部署、不切换当前服务。

### G5 审阅证据

- 主工作区和用户 dirty 文件内容、mtime、Git 状态完全未改变。
- 沙箱逃逸和 Secret Canary 测试失败即阻止产物。
- 没有失败复现或回归测试的 patch 不具备发布资格。
- Target SHA 变化时计划变为 `STALE_SNAPSHOT`。
- EvidenceBundle 可以由独立 Reviewer 复现。

### 回滚

撤销 R2 authority，归档 EvidenceBundle，把隔离 worktree 移至可恢复归档或按批准的保留策略清理；不触碰主工作区。

## 10. Phase 6：R3 Release、Canary 与 Rollback

### 目标

建立清洁、不可变、可追溯的发布基线，并只对批准 Artifact 执行逐环灰度与自动回滚。

### 强制前置项目

当前实时仓库存在用户修改且与上游差距较大。进入 Phase 6 前必须单独完成 Release Baseline 决策：明确运行中的代码来源、如何构建干净 Artifact、用户修改如何保留、不同 Profile 如何切换版本。未通过该决策，G6 自动失败。

### 计划文件边界

```text
plugins/agentops/control/releases/artifacts.py
plugins/agentops/control/releases/builder.py
plugins/agentops/control/releases/manager.py
plugins/agentops/control/verification/rollout.py
plugins/agentops/control/verification/health_gates.py
tests/plugins/agentops/unit/test_artifact_manifest.py
tests/plugins/agentops/unit/test_rollout_state_machine.py
tests/plugins/agentops/integration/test_atomic_version_switch.py
tests/plugins/agentops/fault_injection/test_canary_rollback.py
tests/plugins/agentops/fault_injection/test_stale_artifact.py
```

### 必须产出的接口

```python
build_release(base_sha: str, test_report: TestReport) -> ReleaseArtifact
verify_artifact(artifact: ReleaseArtifact) -> ArtifactVerification
deploy_to_ring(artifact_id: str, ring: RolloutRing, approval: ApprovalReceipt) -> Deployment
evaluate_health_gates(deployment_id: str) -> GateSummary
rollback_deployment(deployment_id: str) -> RollbackResult
```

### Luna 在获批后交付

- [ ] Artifact 包含 source、dependency、build、test、config compatibility 和 rollback hash。
- [ ] dirty tree 不能作为构建输入或部署目标。
- [ ] 版本切换采用原子链接/服务配置切换，不原地覆盖。
- [ ] Rollout ring 按资产登记的 criticality 决定，不按 Profile 名称猜测。
- [ ] 每个 ring 有技术、行为、业务和观察窗口强制门禁。
- [ ] 强制门禁失败停止扩大并自动回滚。
- [ ] 完成一次注入失败的真实 canary 回滚演练。

### G6 审阅证据

- 每次部署可追溯到 Artifact、SHA、测试、审批和配置快照。
- 不可变校验可发现 Artifact 篡改。
- 失败 canary 没有进入下一环。
- 回滚恢复上一版本并通过独立健康验证。
- 控制面重启后 rollout 状态可恢复，不重复部署。
- `default` 等关键 Target 未经明确审批不进入 R3。

### 回滚

恢复上一 Artifact 和服务指针，验证 Gateway 与业务合成探针，撤销该 Artifact 的发布资格。若回滚失败，按安全模型触发 P0 和 Kill Switch。

## 11. Phase 7：Domain Review Packs

### 目标

把已知业务问题转化为版本化、可回放、默认 no-write 的 Review Pack，先完成 Book Workflow，再扩展其他流程。

### 计划文件边界

```text
plugins/agentops/review_packs/model_and_tools/manifest.yaml
plugins/agentops/review_packs/experience/manifest.yaml
plugins/agentops/review_packs/code_health/manifest.yaml
plugins/agentops/review_packs/book_workflow/manifest.yaml
plugins/agentops/review_packs/book_workflow/fixtures/*.json
plugins/agentops/review_packs/book_workflow/assertions/*.py
tests/plugins/agentops/review_packs/test_model_and_tools.py
tests/plugins/agentops/review_packs/test_experience.py
tests/plugins/agentops/review_packs/test_code_health.py
tests/plugins/agentops/review_packs/test_book_workflow.py
```

### 必须产出的接口

```python
load_review_pack(path: Path) -> ReviewPack
run_review_pack(pack: ReviewPack, target: Target, no_write: bool = True) -> ReviewPackResult
compare_to_baseline(result: ReviewPackResult, baseline: Baseline) -> RegressionDecision
```

### Book Workflow 强制断言

- [ ] 明确书籍身份不可被替换。
- [ ] dry-run 与生产入口通过相同核心链。
- [ ] R1/R2/R3 规则实际执行并留下证据。
- [ ] 压缩失败或过短结果不能作为合格产物。
- [ ] 每个产物保留来源、规则版本和处理版本。
- [ ] 不同记录之间不能串写。
- [ ] 弱匹配进入人工确认。
- [ ] 写回需独立 R4、幂等键和 read-back。
- [ ] Prompt/规则修改通过黄金样本、对照样本和回归样本。

### Luna 在获批后交付

- [ ] 从历史问题构建脱敏黄金 fixture，不直接复制敏感业务数据。
- [ ] 业务断言是确定性规则；LLM Judge 只作为辅助 Signal。
- [ ] Review Pack 默认 `no_write=true`，写连接器不在探针路径中。
- [ ] 失败结果能阻止对应 Artifact 扩大 rollout。
- [ ] 每次历史事故解决后都有流程将其加入适当 Review Pack。

### G7 审阅证据

- 已知 Book Workflow 历史失败样本全部被捕获。
- 正常样本误报率在 Molly 接受的基线内，并有逐例说明。
- Review Pack 版本、fixture、断言和基线可回放。
- `no_write` 测试证明探针不会修改飞书或其他生产数据。
- 业务强制断言失败能够阻止发布。

### 回滚

回退 Review Pack 版本或把有争议断言降为报告型 Signal；不得通过删除历史结果掩盖回归。强制门禁变化需要 Owner + Reviewer 审批。

## 12. 每阶段的 Luna 交付包

Luna 每次请求审阅时必须提交一个完整交付包：

```text
1. Scope：本阶段实施了什么、明确没实施什么
2. Files：新增和修改文件清单
3. Interfaces：公开接口和 Schema 版本
4. Tests：命令、完整结果、失败注入结果
5. Security：权限变化、凭证与数据路径
6. Migration：安装、升级、恢复和卸载
7. Evidence：真实运行或模拟运行证据
8. Rollback：已执行的回滚演练及结果
9. Known limits：已确认但不影响本 Gate 的限制
10. Next request：下一阶段需要的明确授权
```

“测试通过”“已经修复”或“可以发布”必须附最新命令输出或结构化证据，不能只提供自然语言结论。

## 13. Codex 审阅清单

### 架构

- 组件是否仍独立于 Gateway 和模型？
- 是否在核心文件加入了 AgentOps 专用耦合？
- 文件和接口是否保持单一职责？
- 状态恢复和版本兼容是否明确？

### 安全

- 新权限是否默认关闭？
- 不可信内容是否可能绕过 Schema/Policy？
- Action、Target、路径、参数是否严格限定？
- Approval、Lease、fencing、幂等和 Kill Switch 是否真实生效？
- 凭证和用户内容是否最小化、脱敏和限期保留？

### 修复与发布

- 是否先复现并加入回归测试？
- 是否搜索并评估上游已有修复？
- 是否触碰实时 dirty worktree？
- 是否有独立验证而不是信任执行退出码？
- 是否能够在失败时恢复上一可用状态？

### 测试

- 单元、契约、集成、故障注入和业务回归是否覆盖本阶段？
- 测试是否包含失败路径、竞态、重启恢复和恶意输入？
- 测试证据是否来自当前提交而非旧运行？

## 14. Gate 结论格式

每个 Gate 只允许以下结论：

```text
APPROVED
- 可进入的下一阶段：Phase N
- 新增允许权限：明确列出或写“无”
- 限制条件：明确列出

CHANGES_REQUESTED
- 阻塞项：逐项列出并关联证据
- 允许继续的只读工作：明确列出
- 禁止事项：明确列出

REJECTED
- 原因：架构、安全或产品方向不成立
- 恢复点：应回到哪个版本/阶段
```

沉默、聊天中的模糊同意、测试绿灯或 PR 合并都不自动构成下一阶段权限授权。

## 15. 实施开始前的最终检查

用户未来要求 Luna 开始某阶段时，执行顺序固定为：

1. 确认前一 Gate 已 `APPROVED`。
2. 读取四份规格的当前版本。
3. 读取仓库 `AGENTS.md` 和适用技能。
4. 检查工作区和现有用户修改。
5. 创建隔离 worktree 和 `codex/agentops-phase-<n>-<name>` 分支。
6. 为该 Phase 写任务级 implementation plan，包含精确文件、失败测试、实现步骤、验证命令和提交边界。
7. Molly 明确授权该 Phase。
8. Luna 按 TDD 小步提交。
9. Luna 生成完整交付包。
10. Codex 按本计划审阅并给出 Gate 结论。

本文件自身不执行上述任何一步，也不授权 Luna 开始任何 Phase；每个 Phase 均需 Molly 另行明确授权。
