# Task Plan

## Goal

实现一套面向 QQ/NapCat 的情报员任务系统，让马噶能口头管理“员工/任务/状态/汇报”，同时保持底层群监听、采集、日报、清理与投递逻辑低耦合、可测试、可审计。

## Phases

| Phase | Status | Notes |
|---|---|---|
| 1. 补齐群策略/归档/日报底座 | completed | 已支持报告目标、即时快照、自动日报投递 |
| 2. 设计并实现 intel assignment 数据模型与状态机 | completed | 已新增 worker store、状态转换、群 membership 对账 |
| 3. 提供统一控制工具，支持招募/停用/查状态/立即汇报 | completed | 已新增 `qq_intel_control` |
| 4. 将 assignment 与群监听/日报/投递联动 | completed | 已接入 NapCat runtime overlay、调度对账、日报投递 |
| 5. 回归验证与部署说明 | completed | 已完成 ACP optional、配置版本漂移、quick command 超时清理、/provider 配置缺省 bug、Nous TLS deprecation、MCP 测试协程清理与子代理中断竞态收口；当前全量测试已绿 |

## Decisions

- 对外保持一个 bot：马噶。
- 对模型层优先暴露统一 control-plane 工具，不直接暴露大量底层实现细节。
- 群策略负责底层路由/采集约束；情报员 assignment 负责“谁在执行什么任务”和汇报行为。
- 原始采集按群共享，日报/快照可被多个 assignment 复用，避免重复打模型与重复采集。
- 当前“主动加群/好友申请处理”不伪造实现；assignment 先以 `awaiting_group_approval` 真实表达等待状态。

## Open Risks

- NapCat/OneBot 现有代码中没有现成的“主动加群”能力，assignment 的入群流程需要真实反映当前能力边界。
- 如果多个 assignment 监听同一群，自动日报投递应按 assignment 分发，不能只依赖单一群策略目标。
- 全量测试当前存在一批与本轮无关的既有失败与缺依赖，不能把“全量不绿”归因到本轮 QQ 情报员改动。
