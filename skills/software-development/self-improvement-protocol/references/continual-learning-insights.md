# 持续学习论文 (Chen 2026) — 可操作洞察

> 来源: "Never Stop Learning: A Survey of Continual Learning and Self-Iteration in LLMs"
> 阅读日期: 2026-05-30
> 全文: /tmp/cl_survey_full.txt (140K chars, 47 pages)

## 核心框架: 三维分类法

| 维度 | 选项 | 对应我的系统 |
|------|------|-------------|
| What | 知识/技能/对齐 | Memory/Skills/USER.md |
| How | 外部信号/自生成/架构 | 用户输入/反思/参数隔离 |
| When | 离线/在线/测试时 | 批量/实时/按需 |

## 五大方法族 × 我的映射

| 方法族 | 论文方法 | 我的实现 | 状态 |
|--------|---------|---------|------|
| 参数隔离 | LoRA, PackNet | 126 个独立 skills | ✅ 良好 |
| 正则化 | EWC, SI | USER.md 偏好约束 | ⚠️ 静态 |
| 重放 | Experience Replay | session_search | ⚠️ 被动 |
| 知识编辑 | ROME, MEMIT | memory replace | ✅ 可用 |
| 流式适应 | TTT, Online DPO | 技能按需加载 | ✅ 良好 |

## 六个关键定理/洞察

### 1. CL+SI 不可能同时最优 (Proposition 2)
- 条件: FCL(ε) ∩ FSI(δ) = ∅ 除非 ε+δ ≥ γ(T-1)/√(d/r)
- 对我: 增大容量(d) → 更多 skills; 降低更新秩(r) → 小 patch; 降低多样性(γ) → 课程设计

### 2. 自我改进必须有 verifier (§4.1)
- 无 verifier 的自我改进会退化 (model collapse)
- 对我: verifier = 用户纠正 + 工具失败 + 任务回退

### 3. LoRA 是最佳折中 (§3.1)
- 遗忘更少, 计算开销小, 可按领域堆叠
- 对我: skill patch (小改动) 优于 skill rewrite (全量)

### 4. 混合策略是趋势 (§5.7)
- RAG(易变事实) + 编辑(纠错) + 合并(能力整合) + 重放(防遗忘)
- 对我: Memory(事实) + Skill patch(流程) + Archive(长期) + session_search(重放)

### 5. 模型合并免训练组合能力 (§5.4)
- TIES-Merging: 85-92% 联合训练性能
- 对我: 跨 skill 知识传播 (一个 skill 的教训 → 相关 skills)

### 6. Online DPO > 离线 RLHF (§5.1)
- 迭代在线 DPO 一致性优于单轮离线
- 对我: 持续小改进 > 一次性大改

## 已落地的改进

1. **Self-Improvement Protocol** — 新 skill: `hermes-agent/self-improvement-protocol`
   - OODA-Reflect 循环: Observe → Orient → Decide → Act → Reflect → Store
   - 纠错结构化捕获模板
   - Skill 自动修补机制
   - 跨 skill 知识传播

2. **纠错模式模板** — references/correction-patterns.md
   - 30 秒快速纠错模板
   - 知识存储决策树

3. **反思频率指南** — 根据任务复杂度决定是否反思
