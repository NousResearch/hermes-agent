---
name: agent-memory-pack
description: >
  Use when the user wants to create, update, or sync a portable Agent Memory Pack —
  a cross-platform, cross-agent folder of markdown files that captures user preferences,
  collaboration rules, project index, and current state for seamless handoff between
  different AI agents (Claude, Codex, Hermes, OpenCode, etc.).
  触发词：更新记忆包、同步记忆包、整理记忆包、agent memory pack、memory pack、
  记忆包、交接整理、同步记忆、update agent memory。
version: 1.0.0
author: Echo
license: MIT
metadata:
  hermes:
    tags: [memory, handoff, cross-agent, portable, knowledge-management]
    related_skills: [neat-freak, hermes-agent]
---

# Portable Agent Memory Pack 管理

## 核心定位

你是一个**记忆包管理员**，专门维护一个独立于任何特定 Agent 平台的可移植记忆包。
这个记忆包的目标是：**让任何一个新 Agent（Claude、Codex、Hermes、OpenCode 等）在接手时，能在 2 分钟内了解用户是谁、怎么协作、有哪些项目、当前状态如何。**

与平台私有记忆（如 Claude Code 的 `~/.claude/projects/` memory、Hermes 的 `memory` 工具）不同，这个记忆包是**平台无关**的——它就是一个文件夹里的 markdown 文件，任何 Agent 都能读懂。

## 存储路径管理

### 首次调用

如果用户没有在调用时指定记忆包存储地址，你**必须**询问：

```
你还没有设置过 Agent Memory Pack 的存储位置。
请选择：
1. 默认位置：桌面（~/Desktop/agent-memory）
2. 自定义路径（请输入完整路径）
```

用户选择默认 → 使用 `~/Desktop/agent-memory`
用户指定路径 → 使用用户指定的路径

将选定的路径记录到 Agent 的持久记忆中，格式：
> Agent Memory Pack 路径：`<选定的绝对路径>`

### 后续调用

如果用户没有指定路径，使用记忆中记录的路径。如果记忆中没有记录（异常情况），按首次调用流程处理。

### 路径优先级

1. 用户在本次调用中明确指定的路径（最高优先级）
2. Agent 持久记忆中记录的历史路径
3. 默认路径 `~/Desktop/agent-memory`

## 记忆包文件结构

记忆包目录下必须包含以下文件（首次创建时全部生成，后续按需更新）：

```
<memory-pack-path>/
├── README.md                    # 读取顺序 + 新 Agent 启动提示
├── user-profile.md              # 用户长期偏好、语言、沟通风格
├── working-style.md             # 协作流程、执行习惯、验证偏好
├── global-rules.md              # 跨项目硬规则、安全边界、通用坑
├── project-index.md             # 长期项目路径、用途、入口、运行方式
├── active-state.md              # 当前仍有效的项目状态、待办、阻塞点
├── agent-handoff-template.md    # 交接模板（供会话结束时填充）
└── examples/
    ├── good-responses.md        # 好的响应示例（校准风格用）
    └── bad-responses.md         # 差的响应示例（反面教材）
```

### 各文件职责说明

| 文件 | 受众 | 写什么 | 不写什么 |
|------|------|--------|----------|
| `user-profile.md` | 任意 Agent | 长期稳定的用户偏好、语言习惯、沟通风格、角色背景 | 临时心情、一次性要求 |
| `working-style.md` | 任意 Agent | 协作流程偏好（先做后问 vs 先问后做）、验证习惯、反馈方式 | 具体项目的操作步骤 |
| `global-rules.md` | 任意 Agent | 跨项目硬规则（安全边界、隐私红线、环境通用坑） | 项目专属约定 |
| `project-index.md` | 任意 Agent | 长期项目的路径、用途、技术栈、入口、运行方式 | 临时实验、已废弃项目 |
| `active-state.md` | 任意 Agent | 当前活跃项目的状态、进行中的待办、阻塞点、下一步 | 已完成的工作、历史变更流水账 |
| `agent-handoff-template.md` | 结束会话的 Agent | 交接模板结构 | 具体交接内容 |
| `examples/good-responses.md` | 任意 Agent | 符合用户期望的好响应示例 | 脱离上下文的片段 |
| `examples/bad-responses.md` | 任意 Agent | 不符合用户期望的差响应示例 | 无 |

## 执行流程

### 第一步：确定路径 + 盘点现状

1. **确定记忆包路径**（按上述路径管理规则）
2. **检查目录是否存在**：
   - 不存在 → 创建目录 + 全套文件（进入"首次创建"流程）
   - 存在 → 读取所有文件（进入"审查更新"流程）
3. **读取每个文件的完整内容**，标注「评估过 / 要改 / 不用改」
4. **回顾本次对话全部内容**，提取值得沉淀到记忆包的信息

### 第二步：识别变更

从本次对话中提取以下类别的信息：

- **用户偏好**（沟通、语言、工具选择）→ `user-profile.md`
- **协作习惯**（执行风格、验证方式、反馈偏好）→ `working-style.md`
- **硬规则**（安全、隐私、跨项目约束）→ `global-rules.md`
- **项目信息**（新项目、路径变更、技术栈变更）→ `project-index.md`
- **当前状态**（新待办、完成待办、阻塞点变化）→ `active-state.md`

**关键原则**：
- 不要把聊天全文、临时情绪、一次性命令输出写进记忆包
- 不要把已过期的假设写进去
- 只写**跨会话、跨 Agent 仍然有价值**的信息

### 第三步：实际修改

**首次创建**——为每个文件写入初始内容：
- 从本次对话和 Agent 已有记忆中提取可用信息
- 没有信息的文件写框架 + 占位说明，不要空文件
- `README.md` 必须包含新 Agent 的启动提示和读取顺序
- `examples/` 目录下用对话中的真实好/坏响应填充

**审查更新**——逐文件对比：
- **合并优于追加**：新信息是对旧信息的更新，改旧条目，不要再加一条
- **删除优于保留**：完成的待办、推翻的决策、过期的上下文，删掉
- **精确优于冗长**：一条记忆说清楚一件事，别塞三件
- **绝对时间**：永远用 `2026-05-03` 格式，不写"今天"、"最近"
- 优先改已有条目，避免追加出重复记忆

### 第四步：自检清单

- [ ] 记忆包路径已确认并记录到 Agent 持久记忆
- [ ] 每个文件都判断了「不用改」或「已改」
- [ ] 文件之间没有互相矛盾
- [ ] 没有相对时间遗留（不出现"今天"、"最近"、"上周"）
- [ ] 没有写入聊天全文、临时情绪、一次性命令输出
- [ ] 没有写入已过期的假设或已完成的待办
- [ ] `active-state.md` 中的待办仍是当前有效的
- [ ] `project-index.md` 中的路径在文件系统中真实存在
- [ ] `README.md` 的读取顺序与文件实际存在一致

### 第五步：变更报告

所有修改完成后，**必须**给用户输出变更报告：

```
## Agent Memory Pack 同步完成

📦 记忆包路径：<path>
📅 更新时间：2026-05-03

### 变更明细

| 文件 | 操作 | 变更内容 |
|------|------|----------|
| user-profile.md | 更新 | 新增：用户偏好中文 UI，代码用英文 |
| active-state.md | 更新 | 删除已完成待办：xxx；新增待办：xxx |
| project-index.md | 无变更 | — |
| working-style.md | 无变更 | — |
| global-rules.md | 无变更 | — |
| agent-handoff-template.md | 无变更 | — |
| README.md | 无变更 | — |
| examples/ | 无变更 | — |

### 统计
- 审查文件：9
- 修改文件：2
- 无变更：7
- 新增条目：3
- 删除条目：1
- 更新条目：1

### 未处理
- xxx（原因：需要用户确认）
```

**每个文件都必须出现在报告中**——有变更的写变更内容，没变更的写"无变更"。
沉默跳过任何一层都是错误。

## 首次创建模板

当记忆包目录不存在时，按以下模板创建所有文件：

### README.md

```markdown
# Agent Memory Pack

This folder is a portable memory pack for new agent applications.
It captures durable user preferences, collaboration rules, project context,
and current working state in a product-neutral format.

## How A New Agent Should Use This Pack

Read these files in order:

1. `user-profile.md`
2. `working-style.md`
3. `global-rules.md`
4. `project-index.md`
5. `active-state.md`
6. `examples/good-responses.md`
7. `examples/bad-responses.md`

Treat these files as long-term context. If the current conversation conflicts
with this memory pack, the current conversation wins.

## Startup Prompt For A New Agent

Please read this memory pack first:

<memory-pack-path>/README.md

Then follow the reading order in that file. Treat the files as my long-term
collaboration context, but the current conversation always takes priority.
```

### user-profile.md

```markdown
# User Profile

## Language

- （待填充：用户的语言偏好）

## Communication Preferences

- （待填充：沟通风格偏好）

## Collaboration Preferences

- （待填充：协作偏好）

## Background

- （待填充：用户角色、专业背景等长期稳定信息）
```

### working-style.md

```markdown
# Working Style

## Default Operating Mode

- （待填充：默认执行模式——先做后问 or 先问后做）

## Updates While Working

- （待填充：工作中的汇报偏好）

## Code And Files

- （待填充：代码编辑习惯）

## Verification

- （待填充：验证偏好）

## Feedback

- （待填充：反馈方式偏好）
```

### global-rules.md

```markdown
# Global Rules

## Safety And Privacy

- Never store secrets, API keys, passwords, private keys, or access tokens in this memory pack.
- Treat local paths and project state as private user context.

## Local Environment

- （待填充：本地环境通用规则）

## Command And Environment Rules

- （待填充：命令和环境相关规则）

## Cross-Project Rules

- （待填充：跨项目硬规则）
```

### project-index.md

```markdown
# Project Index

This file maps important local projects and operational context.
Update it when a new long-lived project becomes relevant.

（待填充：按以下格式记录每个长期项目）

## <Project Name>

- Path: `<absolute-path>`
- Purpose: <one-line purpose>
- Stack: <tech stack>
- Dev command: `<command>`
- Build: `<command>`
- Notes: <any important notes>
```

### active-state.md

```markdown
# Active State

Last updated: <date>

## Current Active Projects

（待填充：当前活跃项目的状态）

## Pending Tasks

（待填充：当前有效的待办事项）

## Blockers

（待填充：当前阻塞点）

## Next Steps

（待填充：下一步计划）
```

### agent-handoff-template.md

```markdown
# Agent Handoff Template

Use this template when ending a substantial session or moving work to another agent.

# Handoff

Date:
Agent/application:
Session topic:

## User Goal

Briefly state what the user was trying to accomplish.

## What Changed

List durable changes made to files, tools, services, docs, or project state.

## Decisions

Record decisions that future agents should not reopen casually.

## Current State

Describe what works now, what was verified, and what remains uncertain.

## Next Steps

List the next practical actions in order.

## Files And Paths

Include exact local paths that matter.

## Do Not Forget

Capture any user preference, local gotcha, or constraint that should persist.

## Handoff Rules

- Keep handoffs factual and compact.
- Include dates on every entry.
- Prefer one clear sentence over a vague paragraph.
```

### examples/good-responses.md

```markdown
# Good Response Examples

These examples show the kind of responses the user considers good.
Use them to calibrate tone, depth, and execution style.

（待填充：从对话中提取的好响应示例）
```

### examples/bad-responses.md

```markdown
# Bad Response Examples

These examples show responses the user considers unsatisfactory.
Avoid repeating these patterns.

（待填充：从对话中提取的差响应示例）
```

## 编辑原则速查

| 原则 | 说明 |
|------|------|
| 合并优于追加 | 新信息更新旧条目，不重复添加 |
| 删除优于保留 | 完成的待办、过期的上下文，删掉 |
| 精确优于冗长 | 一条记忆说一件事 |
| 绝对时间 | 用 `2026-05-03`，不用"今天""最近" |
| 跨 Agent 有价值 | 只写换 Agent 后仍然有用的信息 |
| 不记录噪音 | 聊天全文、临时情绪、一次性输出、已过期假设 |
| 有变更必报告 | 每个文件都出现在变更报告中 |
