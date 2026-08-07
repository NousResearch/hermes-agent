# SkillOpt 洞察 — Agent Skill 优化的工程化方法

> 来源: "SkillOpt: Executive Strategy for Self-Evolving Agent Skills" (Yang 2026)
> GitHub: https://github.com/microsoft/SkillOpt
> 阅读日期: 2026-05-31

## 核心思想

**把 skill 优化当成神经网络训练来对待**：
- Skill = 模型参数 (可编辑的文档)
- Epoch = 一轮完整的训练循环
- Batch = 一组任务执行
- Learning Rate = 更新幅度控制
- Validation Gate = 接受/拒绝改进

## ReflACT 6 阶段流水线

```
Rollout → Reflect → Aggregate → Select → Update → Evaluate
   ↓          ↓          ↓         ↓        ↓         ↓
 执行任务   分析轨迹   合并建议   选top-K   应用edit   验证gate
```

### 1. Rollout (执行)

```python
# 用当前 skill 执行一批任务
results = adapter.rollout(env, current_skill, output_dir)
# 收集成功/失败轨迹
```

### 2. Reflect (反思)

**失败分析 Prompt**:
```
You are an expert failure-analysis agent for AI agent tasks.
You will be given MULTIPLE failed agent trajectories from a single minibatch
and the current skill document.
Your job is to identify the most important COMMON failure patterns across
the batch and propose a concise set of skill edits.
```

**成功分析 Prompt**:
```
You will be given several successful agent trajectories from one minibatch
and the current skill document.
Summarize any useful lessons from these trajectories into one complete
replacement skill document.
```

### 3. Aggregate (聚合)

合并多个 batch 的 patch 建议：
- 去重 (相同 pattern)
- 去冲突 (矛盾的建议)
- 分层合并 (failure + success)

### 4. Select (选择)

按影响力排序 edits：
- 支持度 (support_count): 多少 batch 触发了这个建议
- 类型 (source_type): failure vs success
- 选择 top-K edits

### 5. Update (更新)

**Edits 操作类型**:
```json
{
  "op": "append",
  "content": "<markdown to add at end of skill>"
}
{
  "op": "insert_after",
  "target": "<exact heading/text to insert after>",
  "content": "<markdown>"
}
{
  "op": "replace",
  "target": "<exact text to replace>",
  "content": "<replacement>"
}
{
  "op": "delete",
  "target": "<exact text to remove>"
}
```

**Protected Regions**:
```markdown
<!-- SLOW_UPDATE_START -->
... protected content ...
<!-- SLOW_UPDATE_END -->
```

### 6. Evaluate (评估)

**Validation Gate**:
```python
def evaluate_gate(candidate_skill, cand_score, current_skill, current_score):
    if cand_score > current_score:
        return "accept"
    else:
        return "reject"
```

**Gate Metrics**:
- Hard: exact-match accuracy
- Soft: partial credit (F1, etc.)
- Mixed: weighted average

## Meta-Skill 概念

**Optimizer-side memory** that captures lessons about how to optimize skills better:

```markdown
# Meta Skill

## What Works
- Specific, actionable rules > vague principles
- Rules grounded in evidence > generic advice
- Concise rules > long documents

## What Doesn't Work
- Rules that are too specific to one task
- Rules that conflict with existing rules
- Rules that are too vague to be actionable

## Editing Strategy
- Prioritize failure patterns with high support count
- Avoid redundant edits
- Keep skill documents concise
```

## 学习率控制

### 固定调度
```yaml
lr_schedule:
  - epoch: 0: lr = 1.0
  - epoch: 1: lr = 0.8
  - epoch: 2: lr = 0.6
  - epoch: 3: lr = 0.4
```

### 自主学习率
Optimizer 自己决定更新幅度：
```
Given the current skill, the proposed edit, and the validation score,
decide whether to apply the edit with full strength, reduced strength, or reject it.
```

## 对 Hermes Agent 的启示

### 1. 批量反思 > 单次反思

**SkillOpt**: 分析一批任务的 common patterns
**我**: 每次纠错单独处理

**改进**: 积累一批纠错后，一起分析 common patterns

### 2. Meta-Skill 层

**SkillOpt**: 有 optimizer-side memory
**我**: 没有

**改进**: 创建 meta-skill，记录"如何改进 skill"的经验

### 3. 验证 Gate

**SkillOpt**: 用硬/软指标验证
**我**: 用用户确认

**改进**: skill patch 后，观察用户是否再次纠正同一问题

### 4. Protected Regions

**SkillOpt**: 用 SLOW_UPDATE markers 保护关键内容
**我**: 没有

**改进**: skill 中的 "硬约束" 区域应该被保护

### 5. Edits 操作类型

**SkillOpt**: append, insert_after, replace, delete
**我**: 只有 patch (find-and-replace)

**改进**: 支持更多操作类型

## 与 Self-Improvement Protocol 的整合

```
Self-Improvement Protocol (OODA-Reflect)
  │
  ├─ Observe ─── 用户纠正 / 工具失败
  │
  ├─ Orient ──── 识别 failure pattern
  │
  ├─ Decide ──── 生成 patch 建议
  │                ↓
  │         [SkillOpt Select] ← 按影响力排序
  │
  ├─ Act ─────── 应用 edit
  │                ↓
  │         [SkillOpt Evaluate] ← 验证 gate
  │
  └─ Reflect ─── 更新 meta-skill
```
