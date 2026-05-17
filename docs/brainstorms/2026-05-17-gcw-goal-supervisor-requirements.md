---
date: 2026-05-17
topic: gcw-goal-supervisor
category: brainstorm
---

# GCW Goal Supervisor Requirements

## Summary

本方案定义一个 Evidence-Aware GCW Goal Supervisor：用 Hermes `/goal` 负责 GCW 长任务的跨 turn 持续推进和恢复，但只允许 GCW ledger/status/close-run 证据决定完成状态。`/subgoal` 第一版用于承接中途补充的验收标准、审批提醒和证据要求，但不直接改写 GCW workflow、AC queue 或 validator gates。

---

## Problem Frame

GCW 已经有正式 PMO 契约：issue URL 是入口，ledger/status 是 truth source，PMO 只调度 bounded child executors，最终 closeout 必须经过 validator、completion guard、final evidence 和明确 terminal state。这个体系解决了“做完了吗”的可信度问题，但在长任务上仍有执行摩擦：跨 turn、跨天、等待 worker、等待审批或恢复会话时，用户往往需要反复说“继续”“查一下状态”“别忘了某个验收条件”。

Hermes `/goal` 刚好补上持续推进能力：它能让一个目标跨 turn 存活，自动判断是否继续，并支持 pause/resume/status。Hermes `/subgoal` 则补上中途补充标准的轻量入口。但 `/goal` 的 judge 默认只看最近 assistant response，这与 GCW 的机器证据原则存在错层风险：如果不约束，Hermes 可能把“看起来完成”误当作 GCW done。

---

## Actors

- A1. 用户 / Owner：发起 `/gcw <issue-url>` 或要求继续某个 GCW 任务，补充中途标准或审批决策。
- A2. Hermes Goal Supervisor：负责持续 readback、推进、状态汇报和 continuation 控制。
- A3. GCW PMO：dispatcher-only 状态机，读取 ledger/status，调度 child executors，执行 validator/approval/closeout gates。
- A4. Child Executors：Codex / Hermes SubAgent / 其他 bounded executor，产出 phase report、handoff、测试、PR、部署或其他 evidence。
- A5. Goal Judge：Hermes auxiliary judge，只基于 supervisor 的证据摘要判断 goal 是否继续或停止。

---

## Key Flows

- F1. GCW goal 启动
  - **Trigger:** 用户发起或继续一个明确的 `/gcw <issue-url>` 工作。
  - **Actors:** A1, A2, A3
  - **Steps:** Supervisor 验证这是 GCW 类目标；确认有 canonical issue URL；读取或初始化 GCW run 状态；设置 `/goal` 的持续目标；首轮输出 ledger-backed status。
  - **Outcome:** 有一个 active supervisor goal，且它绑定到一个 GCW issue/run，而不是 free-form 聊天目标。
  - **Covered by:** R1, R2, R3

- F2. 每轮 evidence-aware continuation
  - **Trigger:** `/goal` continuation 触发，或用户要求“继续/状态”。
  - **Actors:** A2, A3, A4, A5
  - **Steps:** Supervisor 先 readback GCW status/ledger/worker evidence；判断当前是否可推进；必要时触发 PMO 下一步；输出固定证据摘要；Goal Judge 只基于这份摘要决定 continue 或 stop。
  - **Outcome:** 长任务能自动持续推进，但不会绕过 GCW truth source。
  - **Covered by:** R4, R5, R6, R7, R8

- F3. 补充标准与审批提醒
  - **Trigger:** 用户在 active GCW goal 中添加 `/subgoal`，例如补充 smoke、截图、审批、证据格式或边界约束。
  - **Actors:** A1, A2, A5
  - **Steps:** Supervisor 将该 subgoal 纳入下一轮 continuation 和 judge 可见标准；在状态摘要中标记其是否只是提醒，还是需要后续规划升级为正式 GCW gate。
  - **Outcome:** 中途补充不会丢在聊天里，但第一版不自动修改 GCW validator/AC 体系。
  - **Covered by:** R9, R10, R11

- F4. Terminal-state stop
  - **Trigger:** GCW closeout、completion guard、worker readback 或 PMO 状态进入 `done`、`blocked`、`needs_user`、`approval_required`；`partial` 仅在被明确标记为 owner-facing final handoff 时停止，否则继续。
  - **Actors:** A2, A3, A5, A1
  - **Steps:** Supervisor 显示 terminal-state evidence；Goal Judge 停止 auto-continuation；只有 `done` 被表达为成功，其余状态必须表达为停止并报告原因/下一步输入。
  - **Outcome:** `/goal` 停止不等于 GCW 成功，terminal state 语义被完整保留。
  - **Covered by:** R12, R13, R14

---

## Requirements

**Goal supervisor boundary**
- R1. GCW supervisor goal 必须绑定 canonical issue URL；没有 issue URL 的正式 GCW 工作不得进入 supervisor auto-continuation。
- R2. Supervisor 不得替代 `/gcw`、GCW PMO、validator、completion guard 或 close-run；它只负责持续推进、readback、状态汇报和 continuation 控制。
- R3. 普通 Hermes `/goal` 仍可用于非 GCW 轻任务；GCW 类 goal 必须明确标识为 GCW supervisor，而不是 free-form goal。

**Evidence-aware continuation**
- R4. 每个 GCW supervisor turn 必须先读取 GCW truth source，再输出状态；不得仅依据聊天记忆、child executor 自述或上一轮 assistant response 继续推进或判定完成。
- R5. 每轮状态摘要必须覆盖：issue URL、当前 GCW phase/run 状态、ledger/status readback、worker 状态、validator/closeout 结果、missing gates、terminal-state candidate、关键 evidence 链接或路径。
- R6. Goal Judge 只能基于 supervisor 的 evidence summary 判断是否继续或停止；不能直接把实现描述、PR 存在、issue 关闭、worker 自称 done 等单一信号当作 GCW 成功。
- R7. 如果 GCW 状态显示 workflow 只初始化但没有真实 phase_started / worker evidence，Supervisor 必须报告 initialization-only 风险并继续 readback/恢复，而不是声称工作已经启动。
- R8. 当 worker 仍 pending/running，或 artifact/manifest/status 缺失时，Supervisor 必须继续或报告 blocked/partial；不得进入成功完成。

**Subgoal handling**
- R9. `/subgoal` 第一版用于追加 judge-visible 补充标准、审批提醒、证据要求或边界约束。
- R10. `/subgoal` 第一版不得自动改写 GCW workflow template、AC queue、validator gates 或 approval gates。
- R11. 当 subgoal 看起来应成为正式 GCW gate 时，Supervisor 必须在状态中显式标记“需要后续升级为 GCW gate/AC”的决策点，而不是静默当作已生效的机器门。

**Terminal-state semantics**
- R12. 只有 GCW `done` 且 closeout/evidence 满足要求时，Supervisor goal 才能表达为成功完成。
- R13. GCW `blocked`、`needs_user`、`approval_required` 必须停止 auto-continuation 并报告原因、证据和下一步需要的输入；这些状态不得被表达为成功。`partial`、stale worker、missing artifact、PR-only/local-only evidence、缺 validator/closeout gate 默认继续，除非 GCW artifact 明确将其定义为 owner-facing final handoff。
- R14. 如果 Goal Judge 因模型错误、解析失败或证据不足无法可靠判断，系统应保守继续或暂停，并要求 readback/人工确认，而不是直接成功。

**Resume and status behavior**
- R15. `/goal resume` 或用户说“继续这个 GCW”时，Supervisor 的第一步必须是 GCW readback，而不是直接继续执行。
- R16. `/goal status` 对 GCW supervisor 应返回 PMO 可读的短状态：issue、phase/run、terminal candidate、missing gates、worker/evidence 状态和下一步。
- R17. 当真实用户消息进入 active GCW goal 时，该消息应优先于 queued continuation；如果消息补充约束，应引导或转化为 subgoal，而不是丢失。

---

## Acceptance Examples

- AE1. **Covers R1, R2, R3.** Given 用户要求“用 GCW 继续这个需求”但没有 issue URL，when Supervisor 尝试启动 GCW goal，then 它拒绝进入 GCW auto-continuation，并要求提供 canonical issue URL。
- AE2. **Covers R4, R5, R6.** Given child executor 回复“done”，但 status/ledger 没有 completion guard 或 close-run evidence，when `/goal` judge 评估本轮，then Supervisor 摘要必须显示 missing gates，judge 不得把目标判为成功。
- AE3. **Covers R7.** Given `/gcw` run 只有 workflow initialized 和 `next_action=continue PMO dispatch`，when Supervisor readback，then 它报告 initialization-only 风险并继续恢复/调度，而不是说工作已开始。
- AE4. **Covers R9, R10, R11.** Given 用户中途 `/subgoal 必须带 active smoke 证据`，when 下一轮 continuation 运行，then smoke 要求出现在 judge-visible criteria 和状态摘要中，但不得声称 GCW validator 已经新增 smoke gate；若需要正式 gate，必须显式标为后续升级点。
- AE5. **Covers R12, R13.** Given GCW closeout 返回 `needs_user`，when Supervisor 汇报，then `/goal` 停止并说明需要什么用户输入，不能显示“Goal achieved”。
- AE6. **Covers R15, R16.** Given 用户隔天说“继续昨天那个 GCW”，when Supervisor 恢复，then 首轮输出 readback 状态和下一步，而不是直接启动新的 executor 或复述旧聊天结论。

---

## Success Criteria

- 长 GCW 任务不再需要用户反复输入“继续”，Supervisor 能在安全边界内持续推进或停止汇报。
- PMO 状态汇报从聊天记忆转为 ledger/status/evidence-backed，能清楚说明当前 phase、worker、missing gates、terminal candidate 和下一步。
- `/goal` 的成功判定不污染 GCW done：没有 closeout/evidence 的任务不会被误报成功。
- `/subgoal` 能承接中途补充标准，并明确区分“judge-visible 标准”和“正式 GCW gate”。
- 下游 `gh:plan` 不需要重新发明 terminal-state mapping、evidence summary、subgoal v1 边界或 resume/readback 行为。

---

## Scope Boundaries

- 不在第一版自动改 GCW workflow templates、routing rules、validator gates、AC queue 或 approval gate 体系。
- 不用 `/goal` 替代 `/gcw`；GCW 仍是正式 PMO entrypoint。
- 不允许无 issue URL 的正式 GCW supervisor goal。
- 不做 UI 看板、移动端状态页或完整 PMO dashboard。
- 不让 Goal Judge 单独决定 GCW `done`；它只判断 supervisor evidence summary 是否达到停止条件。
- 不把 child executor 自报 done、PR 存在、issue closed、local Kanban done 等单一信号作为 Story/GCW 成功依据。

---

## Key Decisions

- 选择方案 B：Evidence-Aware Supervisor，而不是轻包装或完整 subgoal-to-gate sync。理由是它能挡住最大错层风险，同时避免第一版过度改造 GCW。
- `/goal` 的角色是持续推进和恢复，不是验收权威。完成权威仍是 GCW ledger/status/close-run/final evidence。
- `/subgoal` v1 是 judge-visible criteria 和 PMO 提醒，不是正式机器 gate。正式 gate sync 留给后续阶段。
- Terminal-state mapping 必须保留 GCW 语义：`done` 是成功；`blocked|needs_user|approval_required` 是非成功停止并报告；`partial` 默认继续，只有被 GCW artifact 明确标为 owner-facing final handoff 时才停止。

---

## Dependencies / Assumptions

- GCW run 必须能提供可读的 status/ledger/worker/evidence readback；如果 active deployment schema 不稳定，planning 需先定义兼容读取策略。
- Hermes `/goal` judge 当前主要看最近 response，因此 supervisor response 的 evidence summary 格式会直接影响判定质量。
- 用户确认第一版按方案 B 推进，暂不做 subgoal 到正式 AC/validator gate 的自动同步。
- 当前需求文档聚焦产品/行为契约；具体命令、文件、schema、prompt 细节由 `gh:plan` 决定。

---

## Outstanding Questions

### Resolve Before Planning

- 无。当前 scope 已足够进入 planning。

### Deferred to Planning

- [Affects R4-R6][Technical] Supervisor evidence summary 采用现有 `/goal` continuation prompt、GCW PMO status line，还是新增专门的 GCW supervisor status contract？
- [Affects R7-R8][Technical] 如何检测 initialization-only、stale worker、missing artifact 等 readback 风险，并映射到 continue/partial/blocked？
- [Affects R9-R11][Technical] `/subgoal` 文本如何在状态摘要中标识为 soft criterion、approval reminder 或 candidate formal gate？
- [Affects R12-R14][Technical] Goal Judge 的 prompt 是否需要 GCW-specific override，以避免把 blocked/needs_user 当成 achieved？
