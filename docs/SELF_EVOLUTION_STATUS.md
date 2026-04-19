# Hermes Self Evolution Status

Updated: 2026-04-19 17:03:27 +0800

## 结论
Hermes 的自主进化层已经从“原则/宣言”推进到“可运行的受控闭环”，并且这份总览文档已接入自动更新。

当前不只是会提出建议，还已经具备：
- state-based candidate generation
- 低风险 top-priority 执行
- 执行后 rerank
- history + latest snapshot
- report / retro / snapshot 对齐
- 独立 healthcheck
- cron 持续跟踪
- overview status doc 自动刷新

## 本轮变化摘要
- 已执行低风险动作：skill_patch / hermes-self-evolution-maintenance / Captured same-turn continuation discipline and asset maturity rules for candidate generation.
- 当前 top priority：skill / hermes-self-evolution-maintenance / low
- 闭环状态：candidate_generation=active, top_priority_execution=manual-low-risk-executed-in-live-session, rerank_after_execution=active
- snapshot 对齐：aligned_with_next_actions=True, run_stamp=2026-04-19T17-03-27+0800
- embedded healthcheck：no-output

## 本轮收敛说明
- 本轮收敛后 top priority：skill / hermes-self-evolution-maintenance / low
- 本轮已完成低风险动作：skill_patch / hermes-self-evolution-maintenance，因此本轮不是停留在建议，而是已执行后再收敛。
- 当前候选面已收敛到低优先级维护/复用项，没有新的高风险或高优先级漂移。
- surface consistency（可见面对齐视角）：healthy
- embedded healthcheck（报告内嵌视角）：no-output

## 当前漂移提示
- none

## 当前阶段
- 当前 status：`active-self-evolution-loop`
- 当前 top priority：`skill / hermes-self-evolution-maintenance`
- 当前 top priority level：`low`
- 最近一次已执行的低风险动作：`skill_patch / hermes-self-evolution-maintenance`
- 当前 execute-and-rerank：`active`
- 当前 rule engine：`state-based-v1.6`
- 当前 surface consistency：`healthy`
- 当前 embedded healthcheck：`no-output`

## 核心文档
- Charter:
  - `/Users/blank/.hermes/hermes-agent/docs/SELF_EVOLUTION.md`
- Skills policy:
  - `/Users/blank/.hermes/hermes-agent/docs/skills-policy.md`
- Retrospectives README:
  - `/Users/blank/.hermes/hermes-agent/docs/retrospectives/README.md`
- Weekly template:
  - `/Users/blank/.hermes/hermes-agent/docs/retrospectives/template-weekly.md`
- Healthcheck doc:
  - `/Users/blank/.hermes/hermes-agent/docs/SELF_EVOLUTION_HEALTHCHECK.md`
- This status doc:
  - `/Users/blank/.hermes/hermes-agent/docs/SELF_EVOLUTION_STATUS.md`

## 核心脚本
- Candidate actions generator:
  - `/Users/blank/.hermes/scripts/hermes_self_evolution_candidate_actions.py`
- Report generator:
  - `/Users/blank/.hermes/scripts/hermes_self_evolution_report.py`
- Healthcheck:
  - `/Users/blank/.hermes/scripts/hermes_self_evolution_healthcheck.py`

## 关键产物路径
- Next actions JSON:
  - `/Users/blank/.hermes/reports/self_evolution_next_actions.json`
- Status report JSON:
  - `/Users/blank/.hermes/reports/self_evolution_report.json`
- Auto retro:
  - `/Users/blank/.hermes/hermes-agent/docs/retrospectives/2026-04-19-auto-retro.md`
- Latest snapshot:
  - `/Users/blank/.hermes/hermes-agent/docs/retrospectives/status_snapshots/latest-candidate-actions-status.md`
- History snapshot:
  - `/Users/blank/.hermes/hermes-agent/docs/retrospectives/status_snapshots/2026-04-19T17-03-27+0800-candidate-actions-status.md`

## 当前闭环能力分层
### 1. Doctrine / policy
已经有：
- SELF_EVOLUTION charter
- skills policy

### 2. Reporting / retrospectives
已经有：
- JSON report
- Markdown auto retro
- weekly retrospective template

### 3. Candidate generation
已经有：
- state-based candidate rule engine
- human-readable status snapshot
- history snapshots + latest stable entry
- embedded health awareness
- mature asset downranking

### 4. Controlled convergence
已经有：
- top-priority low-risk execution
- execution after rerank
- executed action effect recording

### 5. Long-running tracking
已经有：
- cron-based asset check
- cron-based weekly report
- cron-based autonomous retrospective and skill maintenance suggestion
- cron-based self-evolution healthcheck

## 怎么人工检查这套系统是否健康
建议按这个顺序看：
1. 先跑 healthcheck：
   - `python3 /Users/blank/.hermes/scripts/hermes_self_evolution_healthcheck.py`
2. 再看 latest snapshot：
   - `/Users/blank/.hermes/hermes-agent/docs/retrospectives/status_snapshots/latest-candidate-actions-status.md`
3. 再看 report：
   - `/Users/blank/.hermes/reports/self_evolution_report.json`
4. 确认 report 里的 `snapshot_alignment.aligned_with_next_actions` 是否为 `true`
5. 看 `top_priority` 是否合理、是否长期卡死不变
