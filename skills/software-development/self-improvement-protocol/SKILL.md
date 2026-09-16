---
name: self-improvement-protocol
description: |
  持续自我改进协议 — 任务后反思、纠错提取、skill 自动修补的系统化循环.
  触发词: "复盘 / 反思 / 学到了什么 / 哪里做错了 / 改进 / 提取教训 / post-mortem / lesson learned".
  也会在复杂任务 (5+ tool calls) 结束时自动触发检查.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [self-improvement, reflection, continual-learning, meta-cognition]
    related_skills: [archived-memory-recall, hermes-agent-skill-authoring, systematic-debugging]
---

# Self-Improvement Protocol (持续自我改进协议)

## SkillOpt 的 6 阶段 ReflACT 流水线

论文 "SkillOpt: Executive Strategy for Self-Evolving Agent Skills" (Yang 2026) 提出了一个更系统化的方法：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Rollout → Reflect → Aggregate → Select → Update → Evaluate → Accept/Reject │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 阶段 1: Rollout (执行)
- 用当前 skill 执行一批任务
- 收集成功/失败的轨迹

### 阶段 2: Reflect (反思)
- 分析失败轨迹，提取 common failure patterns
- 生成 patch 建议 (append, insert_after, replace, delete)

### 阶段 3: Aggregate (聚合)
- 合并多个 batch 的 patch 建议
- 去重、去冲突

### 阶段 4: Select (选择)
- 按影响力排序 patch
- 选择 top-K edits

### 阶段 5: Update (更新)
- 应用 edits 到 skill 文档
- 保护 SLOW_UPDATE 区域

### 阶段 6: Evaluate (评估)
- 用新 skill 执行验证集
- Hard gate: exact-match accuracy
- Soft gate: partial credit
- Accept/Reject 决策

### SkillOpt vs 我的 Self-Improvement Protocol

| 维度 | SkillOpt | 我的协议 |
|------|----------|---------|
| **触发** | 每个 epoch | 每个任务/纠错 |
| **反思** | 批量分析轨迹 | 单次纠错提取 |
| **更新** | patch + rewrite | memory/skill patch |
| **验证** | 硬/软 gate | 用户确认 |
| **元学习** | meta-skill | 暂无 |

### 可借鉴的改进

**1. Meta-Skill 概念**

SkillOpt 有一个 "meta-skill" 层，记录"如何优化 skill"的经验：
- 哪些类型的 edit 有帮助
- 哪些类型的 edit 太模糊/冗余/有害
- 什么抽象层级的规则最有效
- 什么失败修复模式应该优先

**对我的启示**：可以创建一个 meta-skill，记录"如何改进 Hermes Agent 的 skill"的经验。

**2. 验证 Gate**

SkillOpt 用硬/软指标验证 skill 改进：
- Hard: exact-match accuracy
- Soft: partial credit
- Mixed: weighted average

**对我的启示**：skill patch 后，可以用用户反馈 (是否纠正) 作为验证信号。

**3. Protected Regions**

SkillOpt 用 `<!-- SLOW_UPDATE_START -->` 和 `<!-- SLOW_UPDATE_END -->` 标记保护区域，防止被快速更新覆盖。

**对我的启示**：skill 中的 "硬约束" 区域应该被保护，不被临时 patch 覆盖。

## 理论基础

基于论文 "Never Stop Learning: A Survey of Continual Learning and Self-Iteration in LLMs" 的核心洞察：

1. **自我改进必须有 verifier** (§4.1) — 没有验证信号的自我改进会退化 (model collapse)
2. **CL + SI 不可能同时最优** (Proposition 2) — 需要 LoRA 式低秩更新 (小改动) 而非全量重训
3. **混合策略是趋势** (§5.7) — RAG(易变事实) + 编辑(纠错) + 合并(能力整合) + 重放(防遗忘)

对我 (Hermes Agent) 来说：
- **Verifier** = 用户纠正、工具失败、任务回退、空响应
- **低秩更新** = memory add / skill patch (小改动) 而非重写整个系统
- **RAG** = MEMORY.md + archive/ (事实性知识不动参数)
- **Replay** = 纠错信号的结构化重放

## 核心循环：OODA-Reflect

```
┌─────────────────────────────────────────────────────┐
│  Observe → Orient → Decide → Act → Reflect → Store  │
│                                        ↑        │    │
│                                        └────────┘    │
└─────────────────────────────────────────────────────┘
```

### 阶段 1: Observe (观察) — 每回合开头

每个回合开始时，扫描：
- MEMORY 百分比 (≥85% → 归档)
- 用户消息中的纠正信号
- 用户消息中的归档/召回触发词

### 阶段 2: Orient (定向) — 任务执行中

实时检测异常信号：
- 工具返回空/失败 → 立即停下根因分析
- 用户打断 ("不对 / 不是这样 / 停") → 记录纠错
- 需要回退/重试 → 记录失败模式

### 阶段 3: Reflect (反思) — 任务结束后

**触发条件** (满足任一)：
1. 复杂任务完成 (5+ tool calls)
2. 用户明确纠正了行为
3. 工具失败后成功恢复
4. 发现了新的 edge case
5. 用户说"复盘 / 反思 / 学到了什么"

**反思清单** (快速自检)：

```markdown
## 反思 #<N> — <日期> — <任务摘要>

### Verifier 信号
- [ ] 用户纠正了？ → 记录纠正内容
- [ ] 工具失败了？ → 记录失败原因
- [ ] 需要回退？ → 记录回退路径
- [ ] 空响应？ → 记录触发条件

### 提取的知识
- 类型: [事实/偏好/流程/陷阱/架构]
- 稳定性: [永不过期/7天有效/会过期]
- 归属: [MEMORY/Skill/Archive/丢弃]

### 行动
- [ ] memory add/replace (高频引用)
- [ ] memory-archive add (稳定但低频)
- [ ] skill_manage patch (流程/陷阱)
- [ ] 丢弃 (一次性)
```

### 阶段 4: Store (存储) — 分类路由

根据知识类型选择存储目标：

| 知识类型 | 信号 | 存储位置 | 示例 |
|---------|------|---------|------|
| **硬约束** | "永远不要 / 绝对不行 / 📌" | MEMORY.md (锁定) | "禁用 hermes update" |
| **用户偏好** | "我更喜欢 / 老规矩" | USER.md | "中文交流" |
| **流程陷阱** | "踩坑 / 会出错 / 别忘了" | Skill patch | "matplotlib 不在 venv" |
| **稳定事实** | 设备配置 / API 端点 | Archive | "BobAPI 4 keys 分组" |
| **临时状态** | PR 号 / 任务结果 | 不存储 (session_search) | "修复了 bug #123" |

## 纠错信号的结构化捕获

当用户纠正我时，不是简单记一条 memory，而是提取结构化模式：

```yaml
correction_pattern:
  trigger: "<什么情况下会犯这个错>"
  wrong_behavior: "<我做了什么>"
  correct_behavior: "<应该怎么做>"
  verifier: "<怎么判断做对了>"
  related_skills: [<涉及的 skill>]
```

### 示例

用户说："你又直接改回去了，我切 provider 是有原因的"

```yaml
correction_pattern:
  trigger: "检测到配置被用户修改后"
  wrong_behavior: "直接撤销用户的修改"
  correct_behavior: "先诊断为什么用户改了，再决定"
  verifier: "操作前先说明推理链路"
  related_skills: [hermes-bobapi-config]
```

这条纠错应该：
1. → memory add (USER.md: "不要擅自撤销配置")
2. → skill patch (相关 skill 加 pitfall)

## Skill 自动修补机制

论文 §5.3 知识编辑的核心洞见：**单次编辑可以精准，但累积编辑会退化**。

对我而言：skill 是"参数"，每次 patch 是一次"编辑"。累积 patch 过多时应该重写。

**修补触发** (满足任一)：
1. 使用 skill 时踩到文档里没写的坑
2. 用户纠正了 skill 里的流程
3. 发现 skill 里的命令/路径已过时
4. 外部依赖变更 (API 端点、包名)

**修补流程**：
```
发现问题
  ↓
skill_manage(action='patch') ← 立即修补，不要等
  ↓
验证修补 (重读 skill 确认)
  ↓
memory add: "Skill <name> 已修补: <改了什么>"
```

**重写触发**：
- 单个 skill 的 patch 次数 ≥ 5
- skill 文件行数 > 原始的 150%
- 用户说"这个 skill 太乱了"

## 主动重放 (Replay) 机制

论文 §3.3: replay buffer 防止遗忘。我的 replay buffer = 纠错记录。

### 具体工作示例

**场景**：用户让我配置 Bob API provider，我之前犯过"直接撤销用户配置"的错误。

```bash
# 1. 加载 BobAPI skill 时，主动检查历史纠错
skill_view(name='hermes-bobapi-config')
# → 触发：检查该 skill 的历史教训

# 2. 搜索该 skill 的纠错记录
session_search("hermes-bobapi-config 纠错", limit=3)
# → 返回：用户说"你又直接改回去了，我切 provider 是有原因的"

# 3. 提取纠错模式
correction_pattern:
  trigger: "检测到配置被用户修改后"
  wrong_behavior: "直接撤销用户的修改"
  correct_behavior: "先诊断为什么用户改了，再决定"
  verifier: "操作前先说明推理链路"

# 4. 应用到当前任务
# → 这次我不会直接改回去，而是先问："我注意到 provider 被改成了 X，
#    是你有意改的吗？还是需要我帮你检查？"
```

**关键区别**：
- **没有 replay**：每次遇到类似场景都可能重复犯错
- **有 replay**：加载 skill 时自动回忆纠错，行为自动修正

### Replay 触发时机

| 时机 | 动作 | 示例 |
|------|------|------|
| 加载某 skill 时 | memory-search "纠错 <skill名>" | 加载 bobapi-config → 检查历史纠错 |
| 用户说"老规矩" | memory-search "老规矩 流程" | 召回标准 5 步运维流程 |
| 开始相关任务 | 主动检查类似场景教训 | 配置新 provider → 检查旧 provider 纠错 |

## 跨 Skill 知识传播

论文 §5.4 模型合并：独立训练的 specialist 可以免训练合并。

我的 126 个 skills 是独立的 "specialist"。当一个 skill 里学到的教训适用于另一个 skill 时，应该主动传播。

**传播触发**：
- 修补 skill A 时发现 pitfall 也适用于 skill B
- 用户在 skill A 的任务中纠正的行为，在 skill B 中也会发生

**传播方式**：
```bash
# 修补源 skill
skill_manage(action='patch', name='skill-a', ...)

# 检查是否有同类 skill 需要同样的修补
skills_list()  # 扫描 related_skills

# 修补目标 skill
skill_manage(action='patch', name='skill-b', ...)
```

### 具体示例：跨 Skill 传播

**场景**：修补 `hermes-bobapi-config` 时发现"不要擅自撤销用户配置"的 pitfall。

```bash
# 1. 修补源 skill
skill_manage(action='patch', name='hermes-bobapi-config', 
  old_string="...",
  new_string="## Pitfall: 不要擅自撤销用户配置\n...")

# 2. 检查相关 skills
skills_list()  
# → 发现 nginx-multi-domain-proxy, xray-proxy-server 也有配置管理

# 3. 传播到相关 skills
skill_manage(action='patch', name='nginx-multi-domain-proxy',
  old_string="...",
  new_string="## Pitfall: 不要擅自撤销用户配置\n...")
```

> **🚧 Future Work**: 目前跨 skill 传播是手动的。论文中的 TIES-Merging/DARE 
> 方法可以自动化合并多个 skill 的 pitfall 列表，但需要：
> 1. 结构化 pitfall 格式（当前是自然语言）
> 2. 自动检测"相同 pitfall"的语义匹配
> 3. 合并冲突解决（当两个 skill 对同一 pitfall 有不同表述）
> 
> 这是下一阶段的改进方向。

## 反模式

❌ **只反思不行动**: 写了反思清单但不执行 memory add / skill patch
❌ **过度反思**: 每个小任务都走完整反思流程 → 只在触发条件满足时反思
❌ **反思后不验证**: 归档完不跑 memory-search 验证召回
❌ **patch 后不重读**: skill_manage patch 后不确认改动生效
❌ **把流水账当反思**: "今天做了 X" 不是反思，"做 X 时发现 Y 应该改为 Z" 才是

## 与现有系统的关系

```
Self-Improvement Protocol
  │
  ├─ 观察层 ─── MEMORY.md 百分比 / 用户纠正信号
  │
  ├─ 反思层 ─── 任务后自检 / 纠错提取
  │
  ├─ 存储层 ─┬─ MEMORY.md (硬约束/偏好)
  │          ├─ Archive (稳定事实)
  │          ├─ Skills (流程/陷阱) ← skill_manage patch
  │          └─ 丢弃 (临时状态)
  │
  └─ 重放层 ─── session_search + memory-search (防遗忘)
```

## 维护

- 这个 skill 本身也应该被持续改进
- 每次使用后检查是否有新的触发条件需要添加
- 反思清单根据实际使用频率调整
