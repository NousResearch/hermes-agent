# Gateway Skills — 配置机制说明

Gateway skills 是 skill-graph 生态中的基础架构技能。它们构成 skill 发现与路由管道的核心层，
通过 system prompt 中的「预安装」声明实现零延迟加载。

## 什么是 Gateway Skill

Gateway skill 是一类**始终可用、不依赖 skill-graph 搜索即可加载**的技能。它们的特征：

- **零延迟访问**：通过 `skill_view()` / `skill_load()` 可直接加载，无需 graph 搜索
- **管道基础层**：参与 skill 的发现、路由、执行和验证流程
- **驻留于 `~/.hermes/skills/`**：作为 symlink 指向实际 skill 目录，实现扁平化索引

当前 Gateway Skills 列表：

| Skill | 职责 | 加载时机 |
|-------|------|---------|
| `skill-graph` | 意图分类 + 路由表 + skill 发现 | **每轮对话第一步**（硬编码在 system prompt 中） |
| `intent-router` | 意图分类、消歧、多意图拆分 | 每轮对话第一步（由 protocol 加载） |
| `post-flight` | 后置收尾：完整性检查、待办扫描、commit 管理 | 每次响应结束前 |
| `quality-gate` | 双轮质量验证（风险筛查 + 定向检查） | commit 前 / 任务完成后 |
| `service-manager` | 服务启停、注册、监控 | 需要操作服务时 |
| `troupe-lookup` | 角色/场景/工作流查找 | Pipeline 执行中需要查角色/场景时 |
| `skill-editor` | Skill 创建、编辑、关系管理 | 需要修改 skill 时 |
| `rules-edit-workflow` | Rules 文件编辑的双确认流程 | 修改 rules 时 |

> 注：`project-directories` 也在计划加入此列表（高频使用：修改任何项目代码前的必加载 skill）。

## System Prompt 中的呈现

当 `config.yaml` 中 `agent.skill_graph_mode: true` 时，system prompt 的 **identity tier**（与 SOUL.md 同级权重）会注入以下内容：

### 1. Operating Protocol（SKILL_GRAPH_IDENTITY）

```text
## Operating Protocol

Skills are discovered at runtime through the skill-graph plugin —
search, don't guess.  Follow this protocol
for EVERY user input, in order:

1. Load the skill-graph companion:
   skill_load("skill-graph")
2. From the loaded content, read the Phase 1 classification table
   and classify the user's intent.
3. Read the Phase 4 routing table and find the matching entry.
4. Call skill_graph_search() with the query from step 3.
5. skill_load("returned-skill-name") for full instructions.
6. Follow its instructions to complete the task.
7. After completing the main work, run your post-response checks.
```

相关代码：`agent/prompt_builder.py` → `SKILL_GRAPH_IDENTITY`

### 2. Skill Discovery Guidance（SKILL_GRAPH_GUIDANCE）

```text
Skill discovery: This profile uses a knowledge graph for dynamic skill
discovery. Call skill_graph_search(query) to find the right skill
by describing what you need in natural language, then load it with
skill_load(name). skills_list() can be used as a fallback for profile-local
skills (zero-latency, limited scope) when graph search returns poor results.
```

相关代码：`agent/prompt_builder.py` → `SKILL_GRAPH_GUIDANCE`

### 3. Available Skills 段

```text
Available Skills
  skill-graph — Skill knowledge graph — discover and load skills by intent
```

这是 system prompt 中**唯一**显式列出的 skill。`skill-graph` 本身是一个 gateway skill，
它作为所有其他 skill 的入口点。其他 gateway skills（intent-router、post-flight 等）
通过 `skill-graph` 的 Protocol 步骤间接引用。

相关代码：`agent/system_prompt.py` → `build_system_prompt_parts()` (line 202-226)

## 配置方式

### 启用 skill-graph 模式

在 profile 的 `config.yaml` 中：

```yaml
agent:
  skill_graph_mode: true
```

### 将 skill 添加为 Gateway Skill（symlink 方式）

Gateway skills 通过 symlink 机制实现零延迟。将 skill 目录链接到 `~/.hermes/skills/` 下：

```bash
# 将 intent-router 从 personal-suite 链接到 ~/.hermes/skills/
ln -s ~/studio/hermes/projects/hermes-personal-suite/skills/intent-router \
      ~/.hermes/skills/hermes/intent-router

# 将 post-flight 链接进来
ln -s ~/studio/hermes/projects/hermes-personal-suite/skills/post-flight \
      ~/.hermes/skills/hermes/post-flight

# 将 quality-gate 链接进来
ln -s ~/studio/hermes/projects/hermes-personal-suite/skills/quality-gate \
      ~/.hermes/skills/software-development/quality-gate
```

Symlink 建立后：
- `skill_view("intent-router")` 可以零延迟加载（不需要 graph 搜索）
- `skills_list()` 可以直接列出（flat index）
- 不会与 skill-graph 的 `source_dirs` 造成重复索引（graph 扫描时会 deduplicate）

### 通过 source_dirs 实现（推荐备选）

如果 skill 的源目录已在 `skill-graph.source_dirs` 中，**不需要额外 symlink**。skill-graph 的 FTS5 索引已经覆盖了该目录下的所有 skill：

```yaml
skills:
  config:
    skill-graph:
      source_dirs:
        - ~/studio/hermes/projects/hermes-personal-suite/skills
```

在此配置下，`intent-router`、`post-flight` 等 skill 通过 `skill_graph_search()` 即可找到，
无需 symlink。但首次搜索有 ~10-50ms 的 DB 查询开销。

### 两种方式的对比

| 特性 | Symlink → `~/.hermes/skills/` | `source_dirs` |
|------|------------------------------|---------------|
| 加载延迟 | 零延迟（直接文件系统命中） | ~10-50ms（SQLite FTS5 查询） |
| `skills_list()` 可见 | ✅ | ❌（仅在 graph 中） |
| 管理复杂度 | 需手动维护 symlink | 自动扫描，零维护 |
| 适用场景 | 高频 gateway skill（每轮必加载） | 一般 domain skill |

**推荐策略**：最核心的 gateway skills（skill-graph、intent-router、post-flight）用 symlink，
其余的通过 source_dirs 管理。

## 以 intent-router 为例

`intent-router` 是用户意图分类的核心 gateway skill。它在 skill-graph 管道中的位置：

```
用户输入
    │
    ▼
intent-router (Phase 1-3 分类)
    │
    ├── 1A Task mgmt  → skill_graph_search("task workflow")
    ├── 1B Execute     → Phase 2 消歧 → Phase 4 路由
    ├── 1C Design      → 只讨论，不执行
    ├── 1D Info query  → skill_graph_search(领域关键词)
    └── 1E Meta        → 直接处理
    │
    ▼
skill_graph_search() → 加载 domain skill → 执行
    │
    ▼
post-flight → quality-gate
```

intent-router 有两个铁律：
1. **必须永远第一个加载**：即使输入看起来直接映射到某个 skill，也必须先走分类流程
2. **每轮重新判断意图**：不能缓存上一轮的分类结果

## 以 post-flight 为例

`post-flight` 是后置收尾 gateway skill，确保每次响应结束前执行完整性检查：

**触发条件**（不可跳过）：
- 调用了任何工具
- 做了 commit / push / pipeline 阶段切换
- 修改了 task 目录、skill、rules、config
- 创建了新文件或改了已有文件

**唯一跳过情况**：用户说"停"/"stop"/"等我"

**检查流程**：
1. **Integrity Check** — "我说过要做的事真的做了吗？"
2. **Pending Action Scan** — 操作后的后续（更新索引、处理 commit 等）
3. **Notification Decision** — 是否需要通知用户
4. **Attention Routing** — 路由到正确的关注点

## 注意事项

- **Symlink 与 source_dirs 不要重叠**：如果一个 skill 的源目录已在 `source_dirs` 中且同时有 symlink，graph 会检测到并去重，但增加不必要的复杂度。只选一种方式。
- **重启生效**：symlink 变更后需要重启 Hermes（`/reset` 或退出重开），因为 skill index 在 session 启动时构建。
- **category 目录**：symlink 目标路径应按 skill 的 `category` 字段放在对应子目录下（如 `hermes/`、`software-development/`）。
