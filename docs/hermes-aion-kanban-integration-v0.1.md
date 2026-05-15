# Hermes × AION Kanban 集成方案 v0.1

> English: Hermes native Kanban becomes the Chinese operational cockpit for the AION factory. GitHub remains the source of truth; Factory Report remains the ledger/health report; Discord remains the activity trace.

## 1. 定位

Hermes 原生 Web Kanban 不重造、不替代 GitHub、不替代 AION Factory Report。本方案只把 Hermes 已有概念翻译并映射为 AION 工厂语义：

| Hermes 原生概念 | AION 中文语义 | 说明 |
| --- | --- | --- |
| board | 项目看板 / 作战面板 | 一个 board 对应一个项目或工厂作战视图。 |
| lane / column | AION 状态列 | 见 state model 文档。 |
| card / task | 作战卡片 | 展示 GitHub 正本、负责人、审计、风险、证据。 |
| assignee / profile | 负责人 / 角色 | GM1、GM2、007、13爷、八府巡按、Cursor、Hermes、外部执行器、君主。 |
| tenant | 项目 | 用于按 AION 子项目过滤。 |
| archive | 已归档 | 完成并过冷却期后移出主战场。 |
| refresh | 刷新战报 | 只读刷新当前聚合状态。 |
| nudge dispatcher | 催动调度器 | v0.1 只读模式禁用；Phase 3 才允许门禁化启用。 |

### 1.1 Hermes 原生调用机制理解

Hermes Kanban 背后不是普通页面列表，而是一套“任务队列 + 调度器 + 多 Profile Worker”的调用机制：

- 任务、评论、运行记录、状态变化都写入 SQLite Kanban 数据库。
- 人和脚本可以通过 `hermes kanban ...`、Dashboard、slash command 操作。
- AI Worker 不靠 shell 调 `hermes kanban`，而是通过 `kanban_*` 工具读写任务，例如 `kanban_show`、`kanban_comment`、`kanban_complete`、`kanban_block`。
- Dispatcher 默认跑在 Hermes Gateway 里，每轮扫描 ready 任务，按 assignee/profile 拉起对应 Hermes profile。
- Worker 被拉起后会拿到当前 task、board、workspace、run_id、claim_lock 等环境变量。
- 每个 Worker 必须用 `kanban_complete` 或 `kanban_block` 收尾；否则会被视为 crash / timeout / gave_up。
- Dashboard API 是薄封装，读写最终还是走同一套 `kanban_db`，并通过 task_events / WebSocket 做实时更新。

因此 AION v0.1 只在 Dashboard 读层增加中文解释和总控摘要，不改变原生调度器、worker、claim、complete/block 的生命周期真相。

## 2. 三个数据源的职责

### 2.1 GitHub：任务正本

GitHub 是最终可信来源。Kanban 只展示，不把卡片本身当作完成依据。

必须优先展示：

- issue：任务正本
- PR：变更正本
- commit：提交证据
- check run / CI：验证证据
- review / audit comment：审计证据
- evidence link：可点击的 GitHub 证据链接

规则：没有 GitHub issue / PR / commit / CI / review evidence 的任务，不得显示为真正完成，只能显示为 `待补证据 / Evidence Missing` 或 `待审计`。

### 2.2 Discord：实时活动线

Discord 不是最终正本，只用于最近活动摘要：

- GM / 13爷派发
- 007 接单
- 八府巡按审计开始 / 完成
- 阻塞报告
- 完成报告
- 君主拍板记录
- 调度器被催动

v0.1 中 Discord trace 可以来自任务评论或外部同步写入的摘要字段。结论仍以 GitHub evidence 为准。

### 2.3 AION Factory Report：账本与健康报告

Factory Report 继续做长期账本和健康报告；Hermes Kanban 是一屏驾驶舱。

Kanban 顶部摘要应读取或复用 Factory Report 的关键指标：

- 总体状态
- 无人值守成熟度
- 总体评分
- 当前瓶颈
- 正在推进 / 待审计 / 阻塞 / 待君主 / 今日完成

## 3. v0.1 范围：只读集成

本版本目标是把 Hermes 原生 Kanban 改造成 AION 中文作战驾驶舱的第一版。

允许：

- 读取 Hermes Kanban task 数据。
- 从卡片正文、结果、评论中抽取 GitHub evidence link。
- 聚合 AION 状态列与君主摘要条。
- 展示 Discord/评论活动摘要。
- 展示风险等级、证据状态、下一关口、是否需君主。
- 中文化主要列名、按钮和卡片字段。

不允许：

- 自动改 GitHub。
- 自动关闭 issue。
- 自动 merge PR。
- 自动触发 production、payment、credits、webhook、external execution。
- 通过拖拽或 nudge dispatcher 让 v0.1 产生执行/写回副作用。

## 4. 页面结构

### 4.0 中文更新简报

顶部必须先给君主一段大白话中文简报，原则是“短、准、能判断是否要拍板”：

```text
过去一段时间发生了什么？
做了哪些工作？
AION 无人值守工厂是否正常运作？
运作过程中出现了哪些问题？
有没有 AAR / 复盘？
现在正在做什么？
未来计划做什么？
现有 AION 已经能实现哪些功能？
```

这段简报只放结论，不堆长细节；详细任务、负责人、证据、审计、AAR、归档状态全部交给 Kanban 卡片和详情页展示。

### 4.1 顶部君主摘要条

页面顶部显示：

```text
AION 帝国工厂驾驶舱
总体状态：运转中 / 部分阻塞 / 等君主拍板
无人值守成熟度：半自动 / 可审计自动 / 不可完全无人化
总体评分：xx / 100
当前瓶颈：GM2 / 007 / 八府巡按 / 外部依赖 / 君主
正在推进：xx
待审计：xx
阻塞：xx
待君主：xx
今日完成：xx
```

### 4.2 主看板

默认按状态列展示，也必须支持按角色、项目、风险、阻塞、待审计、君主待决过滤。

### 4.3 卡片

卡片只展示短字段；长文本放详情页。

示例：

```text
#266 L1 prepare-only 修复包
负责人：13爷 ｜ 审计：八府巡按
风险：L1 ｜ 证据：PR #267
下一关口：八府巡按 formal verdict
等待：21h ｜ 是否需要君主：否 ｜ 是否允许 merge：否
留痕：007 已提交，等待审计
```

## 5. 后续阶段

### Phase 1：只读集成（本 PR）

只展示，不反写。

### Phase 2：受控写回

允许低风险 comment 写回：dispatcher tick、ACK、blocked、audit requested。禁止 close / merge / production / payment / webhook / external execution。

### Phase 3：门禁化自动推进

L0/L1 自动推进，L2 自动推进但必须审计，L3 prepare-only，L4 进入君主列。

## 6. 验收证据

v0.1 PR 至少提供：

- 3 个真实 GitHub issue / PR 映射样例。
- 1 个待审计任务。
- 1 个阻塞任务。
- 1 个待君主拍板任务。
- 1 个已完成/归档任务。
- 测试输出。
- 截图或 API JSON 证据。

## 7. 归档机制

Hermes 原生 archive 只表示“移出默认看板/主战场”，不是 AION 的最终档案。AION 归档必须同时满足：

- GitHub 正本链接存在。
- evidence、审计、completion packet、AAR 已齐全。
- 已写入 AION archive index 或 Factory Report evidence index。
- Kanban 卡片能反查 archive packet。

详细机制见：`docs/hermes-aion-archive-mechanism-v0.1.md`。

## 8. 安全边界

v0.1 是只读驾驶舱。任何真实财务后果、真实法律后果、production、payment、credits、webhook、secret、DB mutation、customer data、irreversible operation、high-risk merge 都不得由普通调度器直接推进，必须进入 `待君主拍板`。
