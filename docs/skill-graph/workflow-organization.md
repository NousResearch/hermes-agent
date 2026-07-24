# Workflow Organization — Gateway Skills + Post-Flight 联动指南

Gateway skills 构成了一条完整的 **意图 → 发现 → 执行 → 验证** 管道。
本文档说明如何利用这些 gateway skills 组织可靠的工作流。

## 管道概览

```
                         ┌─────────────────────────────────────┐
                         │         Gateway Skills Pipeline       │
                         └─────────────────────────────────────┘

  用户输入
     │
     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ intent-router│───▶│ skill-graph  │───▶│ domain skill │───▶│ post-flight  │
│ 意图分类     │    │ 路由 + 发现  │    │ 领域执行     │    │ 后置收尾     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
                                                             ┌──────────────┐
                                                             │ quality-gate │
                                                             │ 质量验证     │
                                                             └──────────────┘
```

### 各层职责

| 层 | Gateway Skill | 输入 | 输出 |
|----|--------------|------|------|
| 意图分类 | `intent-router` | 用户原始输入 + 对话上下文 | 意图类型 (1A~1E) |
| 路由发现 | `skill-graph` | 意图类型 | 匹配的 domain skill 名称 |
| 领域执行 | `domain skill` | 具体指令 | 执行结果 + 产物 |
| 后置收尾 | `post-flight` | 本轮操作记录 | 完整性报告 + commit 摘要 |
| 质量验证 | `quality-gate` | 产物 + checklist | pass/fail + 修正建议 |

## 标准工作流：以「修 bug」为例

### 用户输入

> "这个 relay proxy 的 timeout 硬编码成 30 秒了，改成能从 query 参数配置，默认 30 秒"

### 第 1 步：intent-router 分类

```
Phase 1: 输入分类
  用户说「改成」 → 这是执行指令 → 1B Execute

Phase 2: 意图消歧
  不是"设计讨论"、不是"信息查询" → 确认 1B

Phase 3: 预检
  哪个项目？→ project-directories 定位
  当前分支正确吗？→ git branch --show-current
  有 TASK_MEMORY.md 吗？→ 没有就创建

Phase 4: 路由
  意图："修改 relay proxy 配置"
  → skill_graph_search("relay proxy configuration timeout parameter")
```

### 第 2 步：skill-graph 发现

```python
skill_graph_search("relay proxy configuration timeout parameter")
# → 返回 relay-proxy-config skill
skill_load("relay-proxy-config")
```

### 第 3 步：domain skill 执行

根据 `relay-proxy-config` skill 的指引：
1. 读取 `config.py` 中的 timeout 定义
2. 修改为从 query 参数读取，默认 30 秒
3. 更新相关测试
4. 运行测试验证

### 第 4 步：post-flight 收尾

```
Integrity Check:
  ✅ "我看看配置" → 真的 read_file 了
  ✅ "我改一下" → 真的 patch 了
  ✅ "跑测试" → 真的 terminal 执行了

Pending Action Scan:
  - 改了代码 → 运行 test suite
  - 测试通过 → 考虑是否需要 commit
  - 变更了源文件 → 检查服务是否需要重启

Notification Decision:
  - 测试全部通过 → 报告结果
  - 输出 commit 摘要（不 push）
```

### 第 5 步：quality-gate 验证

```
Pass 1: 风险筛查
  ✅ 修改是否在正确的分支上？
  ✅ 是否有遗留的 print/console.log？
  ✅ Import 是否完整？

Pass 2: 定向检查
  ✅ 默认值是否保持 30 秒不变？
  ✅ query 参数解析是否处理了非法输入？
  ✅ 相关测试是否覆盖了新路径？
```

## 多意图拆分工作流

当用户在一句话中包含多个独立意图时，intent-router 会先拆分为独立子任务：

### 用户输入

> "把 timeout 改成可配置的，顺便看看最近的 CI 为什么挂了"

### intent-router 拆分

```
Multi-Intent Split:
  ├── 子任务 A: "把 timeout 改成可配置的" → 1B Execute
  │   → skill_graph_search("relay proxy configuration")
  │   → relay-proxy-config → 执行 → post-flight
  │
  └── 子任务 B: "看看最近的 CI 为什么挂了" → 1D Info query
      → skill_graph_search("CI failure debug pipeline")
      → ci-debug → 分析 → post-flight
```

每个子任务独立走完整的 pipeline：分类 → 路由 → 执行 → post-flight。

## 设计讨论工作流（1C）

当用户意图是「设计讨论」时，pipeline 有不同路径：

### 用户输入

> "你觉得这个 API 应该用 REST 还是 GraphQL？"

### 流程

```
Phase 1: 1C Design discussion
  → 不执行任何操作
  → 回答时不调用修改性工具（write_file、patch、terminal 写入命令等）

Phase 4: 路由（讨论也需要 skill 指导）
  → skill_graph_search("API design REST GraphQL comparison")
  → 加载相关 design skill 获取框架
  → 基于框架讨论，不执行

post-flight: 
  - 检测到是设计讨论 → 不执行操作 → 不需要后续
```

## Pipeline 执行工作流

对于更复杂的任务（如 lantern-tower 论文 pipeline），intent-router 会路由到 `pipeline-manager`：

```
用户输入: "开始润色第二章"
     │
     ▼
intent-router → 检测到 pipeline 场景
     │
     ▼
pipeline-manager → 读取 PIPELINE.md → 按阶段编排
     │
     ├── Phase 1: 加载 censoris → 审查
     ├── Phase 2: 加载 domain reviewer → 评审
     ├── Phase 3: 聚合反馈 → 加载 theoris → 修订
     └── Phase N: post-flight（pipeline-manager 保证）
     │
     ▼
quality-gate → 最终验证
```

> pipeline-manager 自身保证每一步的 post-flight，不需要在 pipeline 执行中手动调用。

## Gateway Skills 加载策略

### 每轮必加载

这些 gateway skill 的加载开销接近于零，应在每轮对话中都走一遍：

| Skill | 原因 |
|-------|------|
| `intent-router` | 用户意图可能随时改变 |
| `post-flight` | 任何操作后都需要收尾 |

### 按需加载

| Skill | 触发条件 |
|-------|---------|
| `quality-gate` | commit 前 / 任务完成 / Phase 切换 |
| `service-manager` | 需要操作服务 |
| `troupe-lookup` | Pipeline 中需要查角色/场景 |
| `skill-editor` | 需要创建/编辑 skill |
| `rules-edit-workflow` | 需要修改 rules 文件 |

## 自定义 Gateway Skill

当你有一个高频使用的 skill 需要加入 gateway 列表时：

### 1. 确认资格

Gateway skill 应该满足：
- **高频使用**：几乎每轮或每天都会被加载
- **低延迟需求**：需要零延迟访问（不能等待 graph 搜索）
- **管道基础层**：参与 skill 发现/执行/验证流程

### 2. 建立 symlink

```bash
ln -s <source-skill-dir> ~/.hermes/skills/<category>/<skill-name>
```

### 3. 验证

```bash
# 重启 Hermes 后验证
hermes skills list | grep <skill-name>
# 或在新 session 中
skill_view("<skill-name>")
```

### 4. 在 intent-router 路由表中注册（如需要）

如果新 gateway skill 需要被 intent-router 识别为特定意图的处理者，
在 intent-router 的 SKILL.md 中补充路由规则。

## 反模式与陷阱

### ❌ 跳过 intent-router 直接加载 domain skill

```
用户: "set-dialog-pane .1"
❌ 直接 skill_load("tmux-dual-pane")   ← 跳过了分类
✅ intent-router → 分类 → 路由 → tmux-dual-pane
```

**为什么不能跳过**：即使输入看起来明确，也可能漏掉 Phase 2 消歧（执行 vs 讨论）
和 Phase 3 预检（项目边界、规则检查）。

### ❌ 缓存 intent 分类结果

```
第 1 句: "这个 timeout 是多少？"    → 1D Info query ✅
第 2 句: "改成 60 秒"               → 仍用 1D？❌
```

**正确做法**：每轮都重新判断意图。第 2 句是 1B Execute，不是 1D。

### ❌ 执行完操作后跳过 post-flight

```
✅ write_file 了 → 没有跑 post-flight → 没检查是否需要 commit
```

**正确做法**：任何工具调用后都在响应结束前跑 post-flight。

### ❌ 在 post-flight 中 push

```
❌ git commit && git push   ← 链式执行，禁止
✅ git commit → post-flight → 等用户说 push → git push
```

## 参考

- [Gateway Skills 配置说明](gateway-skills.md) — 详细的配置机制
- `agent/prompt_builder.py` — `SKILL_GRAPH_IDENTITY` 和 `SKILL_GRAPH_GUIDANCE` 定义
- `agent/system_prompt.py` — System prompt 组装逻辑
- `hermes_cli/commands.py` — `_collect_gateway_skill_entries()` 用于 gateway 平台 skill 注册
