---
date: 2026-05-17
topic: hermes-goal-subgoal-gcw
focus: Hermes /goal 与 /subgoal 对 GCW 的参考价值和结合建议
---

# Ideation: Hermes Goal/Subgoal × GCW 结合建议

## Codebase Context

- Hermes v0.13 引入 `/goal`：跨 turn 持久目标，after-turn auxiliary judge 判断 done/continue，自动注入 continuation prompt，默认 20 turn budget，支持 `/goal status|pause|resume|clear`，状态可随 `/resume` 恢复。
- Hermes v0.14 引入 `/subgoal`：给 active `/goal` 追加用户补充标准，judge prompt 和 continuation prompt 都能看到；在 Gateway 中作为 control-plane mutation，可 mid-run 安全添加/删除/清空。
- GCW 是更正式的 PMO 状态机：issue URL mandatory，PMO dispatcher-only，ledger/status 是 truth source，child executors 有界，validator/approval/final evidence 是硬门，terminal success 只有 `done`；`blocked|needs_user|approval_required` 是非成功停止；`partial` 默认继续，除非 artifact 明确说明它是 owner-facing final handoff。
- 关键错层风险：`/goal` 的 judge 主要看最近 assistant response；GCW 的完成必须看 `status.json`、`ledger-updates.jsonl`、validator/close-run、PR/deploy/smoke/issue comment 等机器证据。因此 `/goal` 可做外层持续推进器，不能替代 GCW validator/closeout。
- HKTMemory 召回未发现同主题历史 ideation；近 30 天本 repo ideation 仅有 BOSS 招聘浏览器 profile 相关，主题不重合。

## Ranked Ideas

### 1. GCW Goal Supervisor：用 `/goal` 做长程 GCW 外层监督循环
**Description:** 为每个 `/gcw <issue-url>` 自动建立一个 supervisor goal：目标不是“做完需求”，而是“持续推进 GCW，直到 close-run 给出 `done`，或给出 `blocked|needs_user|approval_required`/owner-facing final `partial` 非成功停止，并每轮报告 ledger-backed status”。
**Warrant:** `direct:` Hermes `/goal` 已支持跨 turn 持续推进、judge、pause/resume/status；GCW contract 要求 PMO dispatcher-only、ledger/status truth、terminal states。
**Rationale:** 解决 GCW 长任务跨 turn/跨天时需要人反复说“继续”的问题，但不破坏 GCW 作为事实源和验收源的地位。
**Downsides:** 需要让 goal judge 明确只判断 GCW terminal-state 报告是否达到，而不是判断实现本身是否“看起来完成”。
**Confidence:** 92%
**Complexity:** Medium
**Status:** Unexplored

### 2. Evidence-Aware Goal Judge：让 `/goal` 只基于 GCW 证据摘要判 done
**Description:** GCW supervisor 每轮必须输出固定证据摘要：issue URL、current phase、status.json path、ledger tail、validator/close-run verdict、missing gates、final evidence links。`/goal` judge 只对这个摘要判定是否继续。
**Warrant:** `direct:` `/goal` judge 只天然看到最近 response；GCW truth 在 ledger/status/artifacts，不在聊天记忆。
**Rationale:** 这是防止“agent 自称完成”污染 GCW done 的关键护栏，也是 `/goal` 和 GCW 结合的最低可用条件。
**Downsides:** 需要标准化 supervisor turn 的输出格式；否则 judge 仍可能依据自然语言误判。
**Confidence:** 95%
**Complexity:** Low-Medium
**Status:** Unexplored

### 3. `/subgoal` → Acceptance Criteria Queue：把 mid-run 用户补充变成可验证门槛
**Description:** 用户在 GCW 执行中补充“必须带截图”“不要改 public API”“需要 active smoke”等要求时，先落为 `/subgoal`，再同步到 GCW workflow/AC/validator gate 队列；未同步前只能作为提醒，不能算正式 done gate。
**Warrant:** `direct:` `/subgoal` 可 mid-run 安全加入 judge 可见标准；GCW 的正式完成依赖 validator gates 和 final evidence。
**Rationale:** 解决 PMO/老板临时补充要求容易留在聊天里、没进入执行系统的问题，同时避免 `/subgoal` 文本标准被误当成 GCW 机器门。
**Downsides:** 需要定义 subgoal 到 AC/validator 的映射规则，以及哪些 subgoal 只作软约束。
**Confidence:** 90%
**Complexity:** Medium
**Status:** Unexplored

### 4. Approval-as-Subgoal：把审批门显式放入 active goal
**Description:** 当 GCW 进入 `workflow_pending_user_confirmation`、`approval_required`、生产变更、关闭 issue、发布等节点时，自动添加 approval subgoal，并让 supervisor goal 暂停/报告 `needs_user`，直到审批证据写入 ledger。
**Warrant:** `direct:` GCW 有 approval gates 和 `needs_user`；`/subgoal` 是 mid-run control-plane criteria，judge 可见。
**Rationale:** 让长程自动推进具备“刹车”，防止 `/goal` 的持续性越过 GCW 的人工授权边界。
**Downsides:** 若 approval subgoal 未与 ledger evidence 绑定，可能变成聊天提示而非真实审批门。
**Confidence:** 88%
**Complexity:** Low-Medium
**Status:** Unexplored

### 5. Goal/GCW Terminal-State Mapping：保留 `blocked|needs_user|partial` 语义，不要折叠成 goal success
**Description:** 建立明确映射：GCW `done` 才对应 goal achieved；GCW `blocked|needs_user|approval_required` 对应非成功 stop-with-report；GCW `partial` 默认继续，只有 artifact 明确把它定义为 owner-facing final handoff 时才 stop-with-report；`continue` 对应继续 PMO dispatch/readback。
**Warrant:** `direct:` `/goal` 内部是 done/continue；GCW final states 更丰富，并且 human override 不能把缺失机器门转为 done。
**Rationale:** 避免最危险的错层：Hermes 认为“任务已结束”就把 GCW 的 blocker/partial 当成功汇报。
**Downsides:** 需要改造文案和 judge prompt，让“停止自动继续”与“成功完成”分离。
**Confidence:** 94%
**Complexity:** Low
**Status:** Unexplored

### 6. GCW Resume Readback：`/goal resume` 先读 ledger/status，再恢复下一步
**Description:** 恢复一个 GCW goal 时，第一步不是继续执行，而是读取 `status.json`、`ledger-updates.jsonl`、worker manifests、issue comments，重建 current phase、pending gates 和 terminal risk，再决定是否继续。
**Warrant:** `direct:` `/goal` state survives resume；GCW 已知存在 initialization-only/no-op、worker stale、artifact missing 等恢复风险，必须 readback。
**Rationale:** 让“继续昨天那个 issue”真正可用，减少跨天长任务的上下文漂移和误报进度。
**Downsides:** readback 规则需要维护；如果 artifact schema 漂移，需要兼容层。
**Confidence:** 89%
**Complexity:** Medium
**Status:** Unexplored

### 7. Issue-URL-Bound Goal：GCW 类 goal 必须绑定 canonical issue
**Description:** 对所有 GCW supervisor goal 强制 issue URL：没有 URL 不启动 dispatcher、不建 ledger、不进入 auto-continuation；只允许普通 `/goal` 做非 GCW 的轻任务。
**Warrant:** `direct:` GCW canonical entrypoint 是 `/gcw <issue-url>` 且 issue URL mandatory；Hermes `/goal` 是 free-form text。
**Rationale:** 把 Hermes 长程执行纳入 demand-pool / GitHub Issue 事实源，避免 PMO 追踪黑洞。
**Downsides:** 对临时探索任务略重；需要保留“普通 goal”和“GCW goal”的分层。
**Confidence:** 91%
**Complexity:** Low
**Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | 自动 Issue Intake 到完整 GCW 任务包 | 和 Issue-URL-Bound Goal + subgoal/AC mapping 重叠，范围更大，容易滑向 planner。 |
| 2 | 自动拆分 Bounded Child Executors | GCW 已有 workflow/routing/phase contract，作为独立新想法重复现有机制。 |
| 3 | Goal-as-Compound-Interest Ledger | 表达有启发，但不够具体；已吸收到 Evidence-Aware Judge 和 Resume Readback。 |
| 4 | Final Evidence Compounding | 价值成立，但更像 `gh:compound` / knowledge-distiller 主题，不是 goal/subgoal 与 GCW 的核心结合点。 |
| 5 | PMO 状态面板 | 与 `/goal status` 汇总 ledger truth 相关，但作为产品面板需要更多 UI/PMO 设计，当前阶段并入 supervisor status 输出。 |
| 6 | Subgoal-Based Assumption Refactoring | 偏 brainstorm 变体；核心已由 `/subgoal` → AC Queue 覆盖。 |
| 7 | Approval Gate 变成 Subgoal | 保留为 Approval-as-Subgoal，但必须强调 ledger approval evidence。 |
| 8 | Pause/Resume as Option Value | 概念表达好，但功能落点由 GCW Resume Readback 覆盖。 |
| 9 | Bounded Child Executors as Goal Leverage Multipliers | 太接近 GCW 现有 dispatcher/child executor contract，缺少新机制。 |
| 10 | 自动维护 GCW Ledger / Status Truth | 过大且可能违反 PMO dispatcher-only；保留较窄的 evidence summary/readback。 |

## Recommended Next Step

建议先进入 `gh:brainstorm` 的种子不是“全量集成”，而是：

> GCW Goal Supervisor：用 Hermes `/goal` 做外层持续推进器，但以 GCW ledger/status/close-run 作为唯一完成依据，并用 `/subgoal` 接收 mid-run AC/approval 补充。

这条能直接回答“参考价值”和“结合建议”，且风险边界最清楚：`/goal` 负责持续性，GCW 负责事实源、执行契约和验收。
