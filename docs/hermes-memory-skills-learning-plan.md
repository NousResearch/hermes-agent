# Hermes 记忆、会话回忆与技能系统：8 天学习计划

本计划基于当前 `main` 的实现，目标是掌握 Hermes 的事实记忆、会话回忆、程序性技能、外部用户模型，以及 Skill 开放标准。按每天约 90 分钟设计，每天都有明确的阅读、实验、产出和验收标准。

先澄清一个关键点：README 中的 “FTS5 session search with LLM summarization” 需要结合当前代码理解。**现在的 `session_search` 工具本身不会调用 LLM**，而是返回数据库中的真实消息；主 Agent 可以根据检索结果生成总结。遗留的 `auxiliary.session_search` 配置现在会被忽略，见 `hermes_cli/config_defaults.py:760` 和 `tools/session_search_tool.py:1`。

## 一、先建立能力地图

| 能力 | 本质 | 主要存储 | 是否调用 LLM |
|---|---|---|---|
| Agent-curated memory | 保存少量、长期有效的事实和用户偏好 | `MEMORY.md`、`USER.md` | 前台 `memory` 工具、后台 Review |
| Periodic nudges | 按 turn/工具迭代计数，触发后台复盘 | 内存计数器 | 触发后启动 Review fork |
| Autonomous skill creation | 把可复用工作流程沉淀为 Skill | `SKILL.md` + 支持文件 | Background Review、`skill_manage` |
| Skills self-improve | 根据使用中的纠正和经验修改已有 Skill | 同上 | Background Review 优先 patch |
| FTS5 session search | 对历史消息做关键词检索和上下文浏览 | SQLite `state.db` | **检索本身不调用 LLM** |
| Honcho user modeling | 外部服务根据对话推导用户偏好、目标和模式 | Honcho 服务端 | Dialectic reasoning |
| agentskills.io | Skill 的格式和互操作标准 | `SKILL.md` 包 | 不是学习算法 |

可以记成一句话：

> **Memory 记“重要事实”，Session Search 找“历史经过”，Skill 保存“可复用做法”，Honcho 推导“用户是怎样的人”，agentskills.io 规定 Skill 如何打包和交换。**

## 二、实验前准备

使用独立的学习环境，不影响正式数据。

在仓库根目录的 PowerShell 中：

```powershell
$env:HERMES_HOME = Join-Path $HOME 'hermes-learning-data'
$env:HERMES_RUNTIME_DIR = Join-Path $env:HERMES_HOME 'tools'

. .\activate.ps1
hermes setup
hermes
```

注意：

1. 后续每个新 PowerShell 窗口都要重新设置这两个环境变量并激活环境。
2. 验证 Background Review 时使用交互式 `hermes`，不要用 `hermes chat -q`；一次性模式会禁用自动后台 Review。
3. 实验项目、记忆内容全部使用虚构信息。
4. Windows 下测试通过 Bash 运行：

```bash
scripts/run_tests.sh <测试文件> -q
```

## Day 1：理解内置 Memory 的写入和读取

### 目标

弄清楚：

- `MEMORY.md` 和 `USER.md` 分别存什么；
- 记忆怎样写入；
- 为什么当前会话写入后，要下个会话才能稳定使用；
- Memory 和普通聊天上下文有什么区别。

### 阅读，约 25 分钟

按顺序阅读：

1. `website/docs/user-guide/features/memory.md:11`
   - 两个存储文件；
   - 字符上限；
   - Frozen Snapshot。
2. `tools/memory_tool.py:222`
   - `memory_tool()`；
   - `add`、`patch`、`remove`；
   - `target="memory"` 和 `target="user"`。
3. `agent/agent_init.py:1298`
   - Memory Store 初始化；
   - 默认字符限制；
   - `nudge_interval`。
4. `agent/prompt_builder.py`
   - 只关注记忆如何进入 System Prompt，不需要通读。

### 实验，约 40 分钟

#### 实验 1：明确让 Agent 保存记忆

在新 CLI 会话输入：

```text
请使用 memory 工具保存下面两条信息，并分别选择合适的 target：

1. 我偏好中文、分步骤、带实验的解释方式。
2. 我的学习项目叫 Bluebird，学习笔记放在项目的 docs 目录。
```

观察它是否真正调用了 `memory` 工具，而不是只回答“我记住了”。

然后检查文件：

```powershell
Get-Content (Join-Path $env:HERMES_HOME 'memories\USER.md')
Get-Content (Join-Path $env:HERMES_HOME 'memories\MEMORY.md')
```

预期：

- 用户偏好进入 `USER.md`；
- 项目事实进入 `MEMORY.md`；
- 如果 Agent 分类不同，记录原因，不必强求完全一致。

#### 实验 2：验证跨会话读取

在当前会话执行：

```text
/new
```

然后问：

```text
我偏好什么讲解方式？Bluebird 项目的笔记放在哪里？
```

再回答一个关键问题：

> 新会话为什么能够知道这些信息：是因为旧对话还在上下文里，还是因为记忆文件被重新注入了？

### 产出与验收

写一段不超过 200 字的笔记，必须回答：

1. Memory 的写入入口是什么？
2. 两种 target 如何分工？
3. 为什么 System Prompt 中的记忆快照不能在会话中途刷新？
4. Memory 达到字符上限后会发生什么？

通过标准：能明确说出 **“当前会话写入磁盘，下个会话重新注入”**。

### 对照测试

```bash
scripts/run_tests.sh tests/tools/test_memory_tool.py -q
```

重点阅读这些测试类，不必逐行看完：

- `TestMemoryStoreAdd`
- `TestMemoryStorePersistence`
- `TestMemoryToolDispatcher`
- `TestMemoryBatch`

## Day 2：理解 Periodic Nudges 和 Background Review

### 目标

掌握这条完整链路：

```text
用户对话/工具调用
  → 计数器增加
  → 达到 nudge 阈值
  → 主回答完成
  → 启动后台 Review fork
  → 检查是否值得保存记忆或技能
  → memory / skill_manage 写入
```

### 阅读，约 30 分钟

1. `agent/turn_context.py:753`
   - `_tick_memory_nudge()`；
   - 默认按用户 turn 计数。
2. `agent/turn_iteration_prep.py:165`
   - Skill Review 的迭代计数。
3. `agent/tool_executor.py:746`
   - 调用 `memory` 或 `skill_manage` 后，相应计数器会重置。
4. `agent/turn_finalizer.py:762`
   - 主回答完成后才启动后台 Review。
5. `agent/background_review.py:383`
   - Memory Review Prompt。
6. `agent/background_review.py:1310`
   - Review fork 如何选择 Prompt、快照和 focus。

默认机制：

- Memory：默认每 **10 个用户 turn** 触发一次；
- Skill：默认每 **10 次工具调用迭代** 触发一次；
- `/refine`：不等计数器，立即手动触发；
- 自动 Review 可以通过 `auxiliary.background_review.enabled: false` 关闭。

### 实验，约 40 分钟

第一轮不要真的等待 10 个 turn，直接用 `/refine`。

为了清楚看到自动写入，可以在学习环境的 `config.yaml` 中临时设置：

```yaml
memory:
  write_approval: true

skills:
  write_approval: true
```

然后进行一次对话：

```text
以后你给我整理技术学习笔记时，先给一个简短大纲，再展开细节。
这是跨任务的长期偏好，请在本次复盘时考虑保存。
```

执行：

```text
/refine 重点检查刚才关于学习笔记格式的长期偏好，判断应该写入 USER.md 还是相关 Skill。
```

Review 完成后检查：

```text
/memory pending
/skills pending
```

如果有 Skill 变更：

```text
/skills diff <id>
```

确认内容合理后再批准：

```text
/memory approve all
/skills approve all
```

#### 可选：观察真实 Nudge

如果想验证自动触发，再临时缩小学习环境的阈值：

```yaml
memory:
  nudge_interval: 2

skills:
  creation_nudge_interval: 3
```

然后分别完成两个用户 turn，以及一个包含多次工具调用的小任务。

**不要要求每次都产生写入。** Review 可以判断 `Nothing to save`；要观察的是触发过程，不是强迫它制造记忆。

### 产出与验收

画一张时序图，标出：

- 计数器在哪里增加；
- 为什么 Review 在主回答之后运行；
- 为什么 Review 使用消息快照和独立 fork；
- 为什么它不会直接修改主会话的 Prompt Cache。

通过标准：能解释 **Nudge 是触发器，Background Review 才是实际学习和写入过程**。

### 对照测试

```bash
scripts/run_tests.sh tests/agent/test_background_review.py -q
scripts/run_tests.sh tests/agent/test_background_review_memory_scope.py -q
scripts/run_tests.sh tests/agent/test_skip_background_review.py -q
```

## Day 3：掌握 FTS5 Session Search 和跨会话回忆

### 目标

理解：

- 会话如何进入 SQLite；
- FTS5 如何找到历史消息；
- `session_search` 的四种调用形态；
- “搜索结果”和“LLM 总结”分别发生在哪一层。

### 阅读，约 30 分钟

1. `website/docs/user-guide/sessions.md:902`
2. `tools/session_search_tool.py:601`
3. `hermes_state_search.py:1063`
4. `hermes_state_common.py:864`
   - `messages_fts` 表；
   - 外部内容表和 Trigger。
5. 中文搜索再补读 `hermes_state_search.py:1192`

四种调用形态：

| 形态 | 参数 | 用途 |
|---|---|---|
| Discovery | `query` | 搜索相关历史会话 |
| Scroll | `session_id + around_message_id` | 查看命中消息前后内容 |
| Read | 只有 `session_id` | 阅读整个或有界会话 |
| Browse | 无参数 | 浏览最近会话 |

### 实验，约 40 分钟

#### 第一步：制造一个可搜索的旧会话

在新会话输入：

```text
请不要把下面的信息写入 memory，只保留在当前会话：

项目 Albatross 的验收端口是 4317。
它的部署约束是：先运行数据库迁移，再启动 Web 服务。
```

正常结束这个话题，然后执行：

```text
/new
```

#### 第二步：跨会话检索

问：

```text
上次 Albatross 项目的验收端口和部署顺序是什么？
请先使用 session_search 查找历史会话，再根据查到的真实消息回答。
```

重点观察：

1. Agent 使用的 query；
2. 返回的 `session_id`、`snippet`、`match_message_id`；
3. 是否需要第二次 Scroll；
4. 最终回答怎样总结真实消息。

#### 第三步：练习 FTS5 查询

继续让 Agent 分别尝试：

```text
Albatross
"验收端口"
Albatross OR Bluebird
Alba*
```

再制造一条中文历史消息，测试：

```text
大别山
部署约束
```

记录中文检索是否命中，以及可能走了哪条后备路径。

### 产出与验收

写出以下对比：

| 问题 | 答案 |
|---|---|
| 是向量语义搜索吗？ | 不是，核心是 FTS5/BM25，加 CJK/Trigram/LIKE 等路径 |
| 工具内部调用 LLM 吗？ | 不调用 |
| 返回的是生成的摘要吗？ | 不是，返回真实消息及上下文 |
| 最终回答可以有总结吗？ | 可以，由主 Agent 根据检索结果生成 |
| Memory 和 Session Search 如何分工？ | 少量关键事实 vs 大量历史经过 |

### 对照测试

```bash
scripts/run_tests.sh tests/tools/test_session_search.py -q
```

重点阅读：

- `TestDiscoveryShape`
- `TestScrollShape`
- `TestReadShape`
- `TestCronDemotion`
- `TestNewResetLineageDiscovery`

中文检索可补充阅读 `tests/hermes_state/test_hermes_state.py:986` 附近的 FTS5 和 CJK 测试。

## Day 4：理解 Skill 格式、渐进加载和 agentskills.io

### 目标

在接触自动创建之前，先能手写并验证一个 Skill。

### 阅读，约 25 分钟

1. `website/docs/user-guide/features/skills.md:172`
   - Progressive Disclosure。
2. `website/docs/user-guide/features/skills.md:194`
   - `SKILL.md` 格式。
3. `website/docs/developer-guide/creating-skills.md`
4. 任选一个现有 Skill：
   - `skills/software-development/hermes-agent-skill-authoring/SKILL.md`
5. agentskills.io specification：
   - <https://agentskills.io/specification>

重点理解三级加载：

```text
skills_list
  → 只看名称和描述
skill_view(name)
  → 读取 SKILL.md
skill_view(name, path)
  → 按需读取 references/ 中的材料
```

### 实验，约 45 分钟

使用 `/learn` 生成一个学习专用 Skill：

```text
/learn 创建一个名为 hermes-learning-notes 的技能：

把技术主题整理成中文学习笔记，固定包含：
1. 核心概念
2. 数据流或执行链
3. 一个可操作实验
4. 常见误区
5. 验收问题

SKILL.md 只保留长期适用的流程；详细格式说明放 references/。
```

如果仍开启 `skills.write_approval`，检查 diff 后批准。

然后在 Skills 目录中找到它，检查：

- Frontmatter 是否有 `name`、`description`；
- 描述是否足够让模型判断何时使用；
- `SKILL.md` 是否只放通用流程；
- `references/` 是否按需存放细节；
- 有没有把某次会话写成流水账。

开始新会话并调用：

```text
/hermes-learning-notes SQLite FTS5
```

### 产出与验收

手写一张表：

| 内容 | 应该放哪里 |
|---|---|
| 何时使用该 Skill | Frontmatter description + When to Use |
| 每次都适用的步骤 | `SKILL.md` |
| 只在特定主题需要的细节 | `references/` |
| 可重复运行的验证逻辑 | `scripts/` |
| 一次性任务经过 | 不应写入 Skill |

通过标准：能解释 **agentskills.io 管的是 Skill 的格式和互操作，并不管 Hermes 何时自动创建或改进 Skill**。

## Day 5：理解 Autonomous Skill Creation

### 目标

弄清“复杂任务后自动创建技能”的实际实现，而不是把它理解成一个复杂任务分类器。

### 阅读，约 30 分钟

1. `agent/background_review.py:448`
   - `_SKILL_REVIEW_PROMPT`。
2. `agent/background_review.py:519`
   - Memory + Skill 联合 Review。
3. `tools/skill_manager_tool.py:768`
   - `skill_manage`。
4. `agent/turn_iteration_prep.py:165`
   - Skill Nudge 计数。
5. `agent/turn_finalizer.py:762`
   - Turn 结束后的触发条件。

重点看 Review 的选择优先级：

1. 优先修改本次已经加载的 Skill；
2. 其次修改已有的 class-level umbrella Skill；
3. 或添加支持文件；
4. 只有没有合适 Skill 时，才创建新的 class-level Skill。

### 实验，约 40 分钟

完成一个需要多次工具调用、但可以复用的工作流，例如：

```text
在临时目录里创建三份 Markdown 学习笔记。

要求：
1. 每份都有标题、摘要和验收问题；
2. 创建后逐个检查；
3. 最后汇总检查结果。
```

完成后观察是否触发 Background Review。必要时执行：

```text
/refine 检查刚才的工作是否形成了可复用的方法。只有真正可复用时才创建或修改 Skill，不要记录一次性任务流水。
```

检查：

- 新 Skill 是否真的是“一类任务”，而不是 `create-three-files-today`；
- 是否优先更新了已有 Skill；
- 是否包含可操作的 Procedure 和 Pitfalls；
- 是否出现日期、对话原文、一次性错误日志等不应持久化的内容。

再查看：

- `skills/.usage.json` 中的 `created_by`；
- `skills/.curator_ledger.jsonl` 中的变更记录。

注意：前台 `/learn` 创建的技能通常标记为 `created_by: learn`；只有特定后台来源才标记为 `created_by: agent`。这会影响 Curator 是否自动管理它。

### 产出与验收

回答三个问题：

1. “复杂任务”在实现上主要由什么信号近似？
2. 自动 Review 为什么不应该每次都创建 Skill？
3. `created_by` 为什么更像管理策略标记，而不是可靠的作者证明？

### 对照测试

```bash
scripts/run_tests.sh tests/tools/test_skill_manager_tool.py -q
scripts/run_tests.sh tests/agent/test_background_review_toolset_restriction.py -q
scripts/run_tests.sh tests/tools/test_skill_provenance.py -q
```

## Day 6：理解 Skill Self-Improve 和 Curator 的区别

这是最容易混淆的一天：**Background Review 和 Curator 不是同一个机制。**

### 阅读，约 30 分钟

1. `agent/background_review.py:456`
   - 用户纠正、工作流纠正、过时 Skill 都是学习信号。
2. `tools/skill_manager_tool.py:508`
   - Targeted Patch。
3. `website/docs/user-guide/features/curator.md:17`
4. `agent/curator.py:915`

对比：

| Background Review | Curator |
|---|---|
| 由对话和工具使用量触发 | 由时间和空闲状态触发 |
| 学习本次会话的新经验 | 维护整个 Skill 生命周期 |
| 创建或 patch Skill | stale、archive、可选 consolidate |
| 默认自动运行 | 默认只做确定性 prune |
| 关注“学到了什么” | 关注“哪些 Skill 还有用” |

Curator 默认配置：

- 运行间隔：7 天；
- 最少空闲：2 小时；
- 14 天未使用变为 stale；
- 30 天未使用归档；
- LLM consolidation 默认关闭；
- 归档可恢复，不等于删除。

### 实验，约 40 分钟

#### 实验 1：让已有 Skill 自我改进

先使用 Day 4 的 Skill：

```text
/hermes-learning-notes Prompt Caching
```

然后纠正：

```text
这个技能的输出太长了。以后先用不超过 6 行列出大纲，再展开细节；不要直接输出长篇正文。
```

执行：

```text
/refine 把刚才的纠正更新到实际负责学习笔记的 Skill 中，不要只写入 USER.md。
```

检查：

```text
/skills pending
/skills diff <id>
```

批准后开新会话，再次使用 Skill，验证行为是否改变。

#### 实验 2：安全观察 Curator

```bash
hermes curator status
hermes curator run --dry-run
hermes curator ledger
```

先不要运行 `--consolidate`。如果以后要在隔离环境测试，先执行：

```bash
hermes curator backup
```

### 产出与验收

必须能解释：

- Background Review 可以改进本次使用过的 Skill；
- Curator 默认只自动管理符合来源条件的 Skill；
- Pin、归档、备份和 Ledger 分别提供什么保护；
- 为什么“自动学习”还需要人工审批和回滚机制。

### 对照测试

```bash
scripts/run_tests.sh tests/agent/test_curator.py -q
scripts/run_tests.sh tests/hermes_cli/test_curator_run.py -q
```

## Day 7：理解 Memory Provider 和 Honcho Dialectic Modeling

Honcho 是**可选的 catalog 插件**，不是内置 Memory 的同义词。

### 阅读，约 30 分钟

1. `agent/memory_provider.py:84`
   - `MemoryProvider` ABC。
2. 重点方法：
   - `initialize`
   - `prefetch`
   - `sync_turn`
   - `get_tool_schemas`
   - `on_session_end`
3. `agent/memory_manager.py:391`
   - Provider 如何统一管理。
4. `website/docs/user-guide/features/honcho.md`
5. `plugin-catalog/honcho.yaml`

重点理解：

- `sync_turn`：把本轮对话同步给 Provider；
- `prefetch`：为后续对话取得相关上下文；
- `contextCadence`：基础用户/会话上下文多久刷新；
- `dialecticCadence`：多久做一次推理；
- `dialecticDepth`：推理深度；
- Peer：Honcho 如何区分不同用户和 Agent 身份。

### 无 Honcho API Key 时

做一个 45 分钟的纸面追踪即可：

1. 从 `MemoryManager.sync_all()` 找到 Provider 的 `sync_turn()`；
2. 从 `prefetch_all()` 找到上下文如何返回；
3. 对照 `agent/turn_context.py:1186`，观察外部 Memory 在 turn start 如何参与上下文组装；
4. 画出：

```text
Turn 开始
  → Prefetch
  → 取得用户模型/会话上下文
  → 注入当前请求
  → 模型生成回答
  → Turn 完成
  → sync_turn
  → Honcho 更新用户表示和结论
```

运行：

```bash
scripts/run_tests.sh tests/agent/test_memory_provider.py -q
scripts/run_tests.sh tests/agent/test_memory_async_sync.py -q
```

### 有 Honcho API Key 或自托管服务时

只使用虚构对话：

```bash
hermes plugins install honcho
hermes memory setup honcho
hermes honcho status
```

建议先使用 `tools` recall mode，手动观察 `honcho_context`、`honcho_search`、`honcho_reasoning`，理解后再试 `hybrid`。

**隐私注意：**Catalog 明确披露，对话 turn、推导结论、peer/session ID 会发送到所配置的 Honcho API。不要用真实私人对话做第一轮实验。

### 产出与验收

完成对比表：

| 维度 | Built-in Memory | Session Search | Honcho |
|---|---|---|---|
| 数据内容 | 精炼事实和偏好 | 原始历史消息 | 用户表示、结论、会话上下文 |
| 存储位置 | 本地 Markdown | 本地 SQLite | Honcho 服务端或自托管服务 |
| 检索方式 | 会话开始注入 | FTS5 | 语义搜索和 dialectic reasoning |
| 是否产生推导 | 由 Agent 审核写入 | 检索工具本身不推导 | 是 |
| 数据是否离开机器 | 默认否 | 默认否 | 使用云服务时会 |

## Day 8：综合实验和最终验收

### 综合场景

用三个连续会话完成一个完整学习闭环。

#### Session A：产生事实和经验

1. 保存一条用户偏好；
2. 讨论一个带唯一关键词的虚构项目；
3. 完成一个多步骤任务；
4. 创建或更新一个 Skill。

#### Session B：验证跨会话记忆

执行 `/new`，分别询问：

- 一个应该来自 `USER.md` 的偏好；
- 一个只存在于旧会话、没有保存到 Memory 的细节；
- 一个应该通过 Skill 完成的工作流。

记录 Agent 分别使用了：

```text
System Prompt 中的 Memory
session_search
Skill
```

#### Session C：验证自我改进

1. 使用之前的 Skill；
2. 给出明确纠正；
3. 执行 `/refine`；
4. 检查并批准 diff；
5. 在新会话验证 Skill 行为改变；
6. 运行 `hermes curator run --dry-run`。

### 最终产出

写一篇 1～2 页笔记，建议保存为：

`docs/hermes-learning-system-notes.md`

必须包含：

1. 一张完整数据流图；
2. Memory、Session Search、Skill、Honcho 的区别；
3. Nudge 和 Background Review 的触发过程；
4. Background Review 和 Curator 的区别；
5. 哪些地方调用 LLM，哪些地方不调用；
6. Prompt Cache 为什么要求 Memory 快照冻结；
7. 写入审批、归档、备份、Ledger 如何降低自动学习风险。

### 最终自测

不查资料回答以下问题：

1. `MEMORY.md` 和 `USER.md` 如何分工？
2. 为什么当前会话新写入的 Memory 不会马上进入 System Prompt？
3. Memory Nudge 和 Skill Nudge 分别按什么计数？
4. Background Review 为什么在独立 fork 中运行？
5. `session_search` 当前是否使用辅助 LLM 生成摘要？
6. FTS5 检索和主 Agent 最终总结分别发生在哪一层？
7. 自动创建 Skill 前为什么优先考虑修改已有 Skill？
8. Background Review 和 Curator 的触发方式有什么不同？
9. agentskills.io 标准不负责哪部分行为？
10. 使用 Honcho 云服务时，哪些数据可能离开本机？

如果这 10 个问题能答清楚，并且亲自完成过一次 **“保存记忆 → 新会话读取 → 搜索旧会话 → 创建 Skill → 纠正并改进 Skill”** 的闭环，就已经掌握了这组能力的核心。
