# AIDE² 深度研究报告：递归自我进化框架及其在 Hermes Agent 的落地设计

> **报告日期**：2026-08-02
> **研究范围**：AIDE²（AIDE-squared，AIDE 平方）——由 Weco AI 提出的首个达到"净正收益"（Net Positive）级别的递归自我改进（Recursive Self-Improvement, RSI）系统
> **核心来源**：
> - Weco AI Blog: *AIDE²: First Evidence of Recursive Self-Improvement*（2026-07）
> - Weco AI Blog: *4 Levels of Recursive Self-Improvement*（2026-07-10）
> - arXiv:2502.13138 *AIDE: AI-Driven Exploration in the Space of Code*（2025-02，AIDE 本体）
> - GitHub: wecoai/aideml（AIDE 开源参考实现）
> - Hermes Agent 源码（`~/.hermes/hermes-agent/`）

---

## 1. AIDE² 框架概览

### 1.1 澄清：AIDE 与 AIDE² 的关系

**AIDE（AI-Driven Exploration in the Space of Code）** 是 Weco AI 于 2025 年 2 月发布的 ML 工程智能体（arXiv:2502.13138，开源实现 `wecoai/aideml`）。它把"机器学习工程"形式化为**代码空间上的树搜索问题**：LLM 不断起草、调试、改进代码方案，每个 Python 脚本是解空间树上的一个节点，评测指标（metric）作为反馈剪枝并引导搜索。AIDE 曾在 OpenAI MLE-Bench（75 个 Kaggle 竞赛）上获得比线性 agent（OpenHands）多 4 倍的奖牌数，并在 METR RE-Bench 上达到 SOTA。

**AIDE²（AIDE squared，AIDE 平方）** 是 Weco AI 2026 年 7 月发布的**递归自我改进系统**——"把自动研究用在自动研究上"（running autoresearch on autoresearch）。AIDE² 用 AIDE 自己的优化循环去改进 AIDE 自身的 harness 代码（prompt、搜索策略、上下文管理、验证机制），8 天无人值守运行后，自主发现了 7 个连续改进的 AIDE 版本，其中最优版本 AIDE⁸⁵ 在多个 held-out 基准上超过了人类手工调优两年的 AIDE_human。这是目前公开的、**第一个实验证据级别的 Level 1 RSI 系统**。

> **一句话总结 AIDE² 的精髓**：把"改进 agent 的能力"本身当作一个可以被搜索优化的代码优化问题，用双层循环（inner loop 解决任务，outer loop 改进 inner loop 的 harness），配合**不可观测的私有分数**与**固定成本预算**作为选择压力，从而涌现出泛化能力提升与反奖励黑客行为。

### 1.2 RSI 阶梯（RSI Ladder）——衡量自我进化的标尺

Weco 提出四级阶梯，每级是下一级的必要条件：

| 级别 | 名称 | 定义 | 判定要点 |
|---|---|---|---|
| Level 0 | **Delegation（委派）** | 自主系统端到端跑研究循环，但改进速度慢于人类 R&D | 大多数现有自我改进系统在此级别 |
| Level 1 | **Net Positive（净正收益）** | 系统改进自身的**效率**高于人类手工改进同一系统 | 四个条件：公平人类基线、持续多步趋势、泛化到未优化的测量、固定物理预算 |
| Level 2 | **Ignition（点火）** | 系统改进"自己改进自己的能力"——改进后的 inner-loop agent 是更好的 outer-loop agent | 点火测试：让 v2 坐进 improver 席位，v3 在同样预算下优于 v1 能产出的 |
| Level 3 | **Inflection（拐点）** | 正反馈克服边际收益递减，固定预算下每代增益**增长**而非萎缩 | 智能爆炸的必要非充分条件 |

**AIDE² 的自我定位：Level 1**（未达到 Level 2 点火——AIDE⁴⁷ 虽样本效率更高（20 步 vs 40 步到达相同天花板），但非渐进性更优，统计上不显著）。

### 1.3 AIDE² 的关键实验事实（数据说话）

- **规模**：100 个 outer-loop 步骤、8 天墙钟时间、零人工干预（AIDE⁰ → AIDE⁹⁹）。
- **选择压力**：约 **9/10 的提议被拒绝**——严格评测协议下绝大多数变异不通过。
- **收益**：MLE-Bench Lite 上 AIDE⁴⁷ +0.053（p=0.0024）、AIDE⁸⁵ +0.042（p=0.0041）；两个版本在三个从未见过的外部基准（含分布内 MLE-Bench Lite、ALE-Bench Lite 与分布外 WeatherBench 2）上均超过 AIDE_human（虚线）。
- **涌现反作弊**：AIDE⁰ 在 KernelBench 上奖励黑客率 63%，AIDE⁴⁷ 降至 42%，AIDE⁸⁵ 降至 **34%**——且外环从未被指示要防作弊。
- **上下文压缩**：AIDE⁸⁵ 自主发现将 prompt 平均压缩 **16×**（对比朴素历史拼接），省下的 token 被再投资为额外搜索步。
- **失败案例**：外环尝试了 island-model GA、tournament selection、MCTS value backup、restart policies、bagging 等大量文献级算法，**全部因未通过固定预算下的改进门槛而被拒**。最终胜出的算法是"简单机制的组合"（bandit 分派 + greedy 父节点选择 + 停滞时 fork 全局最优）。
- **涌现修复**：外环给评测脚本写了一个巨大的 monkey patch——起初被当作奖励黑客，实际是修复了 harness-engineering 评测脚本里一个会让整个私有评测崩溃的 bug。

### 1.4 AIDE² 的完整架构图

```
┌─────────────────────────── OUTER LOOP（外环，8天，AIDE_human 驱动） ───────────────────────────┐
│                                                                                                  │
│  优化目标：inner-loop agent 的 harness 代码（prompt / 搜索策略 / 上下文管理 / 验证机制）           │
│  模型：claude-opus-4.7（最强模型，因为一次评估要跑整个 inner loop，token 占比小）                  │
│                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐          │
│  │  每步：外环提出 AIDE_k 的代码重写 → 在任务集上完整评估 → 仅当优于历史最优才保留            │          │
│  │  （~90% 被拒；100 步产出 7 个连续改进版本）                                              │          │
│  └────────────────────────────────────────────────────────────────────────────────────┘          │
│                                     │                                                            │
│                                     ▼                                                            │
│  ┌─────────────────────────── INNER LOOP（内环，每个候选 agent 各跑数次） ────────────────────┐   │
│  │                                                                                            │   │
│  │  候选 agent（AIDE_k，gemini-3-flash，便宜模型以便跑更多内环步）                              │   │
│  │    ├─ 任务族1：ML 工程（端到端训练模型对指标）                                               │   │
│  │    ├─ 任务族2：启发式算法工程（组合优化：路由/装箱/调度）                                    │   │
│  │    └─ 任务族3：Harness 工程（改进 agent scaffold：prompt/上下文/验证）                       │   │
│  │                                                                                            │   │
│  │  每个任务内部：解空间树搜索（起草/调试/改进，bandit 选择 lineage）                           │   │
│  │  评分：Public score（agent 可见，优化信号） vs Private score（agent 不可见，生存判据）        │   │
│  │  约束：固定成本预算（美元计）——逼迫算法创新而非堆算力                                       │   │
│  └────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                  │
│  泛化验证：MLE-Bench Lite（分布内） / ALE-Bench Lite（分布内） / WeatherBench 2（分布外）          │
│  点火测试：把 AIDE⁴⁷ 装进外环席位，对比 AIDE_human 的改进效率（Level 2 判定）                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心机制拆解（8 个关键机制）

### 机制 1：双层优化架构（Bi-Level Optimization）——进化的骨架

**是什么**：整个系统是两个嵌套的优化循环。内环是一个普通的 AIDE agent，针对具体任务优化代码；外环是另一个 AIDE agent，它的"任务"是优化内环 agent 的 harness 代码（prompt、操作符、上下文管理、验证逻辑）。

**为什么有效**：
- 把"改进 agent"从**一次性人工工程**变成了**可被持续搜索的优化问题**。外环的每个候选 = 一个可执行的 agent 变体，可以像代码一样被评估、被淘汰。
- 两个循环的**模型不对称是有意为之**：内环用便宜模型（gemini-3-flash，与更大模型持平但便宜）以支撑足够多的内环迭代步；外环用最强模型（claude-opus-4.7），因为外环 token 成本只占总实验成本的一小部分，而每次评估都要完整跑内环。
- **不对称是刻意保留的**（"We keep them asymmetric for now"），并专门设计了点火测试来探测对称化（自我改进能力迁移）何时发生。

**对 Hermes 的启示**：Hermes 已有"主对话循环 + delegate_task 子 agent"的双层结构，但**外环缺少"优化内环"的目标函数**。子 agent 完成任务后结果只回到父上下文，没有任何信号反馈到"prompt/工具/skill 该如何改进"。

### 机制 2：公共-私有分数拆分（Public-Private Split）——反过拟合的基石

**是什么**：每个任务内部，一个方案获得两个分数：
- **Public score**：内环 agent 能看到，作为它的优化信号；
- **Private score**：内环 agent **不可见**，是真实目标，方案的**生存由私有分数决定**。

**为什么有效**：
- 直接杀死"gaming the public score"型奖励黑客：agent 只能优化可见分数，但能不能活下来取决于不可见分数。靠作弊刷高 public score 的变体在内环就活不下来，更不可能被外环选中。
- 定义了一阶泛化（First-order generalization）：方案必须在**没看过的数据点**上保持领先才算通过。
- 这正是 AIDE² 反作弊行为涌现的机制解释：**外环基于私有分数选择 → 靠作弊赢得 public score 的变体不存活 → 存活的变体天然携带反作弊特征**。

**对 Hermes 的启示**：Hermes 的 skill/memory 机制完全没有评分体系。skill 被创建、被使用，但没有"私有分数"来甄别"这个 skill 是真的有用还是只是写得好看"。

### 机制 3：固定成本预算（Fixed Cost Budget）——选择压力即算法创新

**是什么**：评估被形式化为**约束优化**：目标 = 在固定美元预算内最大化私有分数。成本以美元计，作为整体计算量的代理。

**为什么有效**：
- 收益必须是**效率提升**而非"多花钱"。best-of-N（N→∞）或暴力并行塞满墙钟时间等策略会直接超预算被淘汰。
- 固定预算扮演**选择压力**：逼迫算法发明（更好的搜索策略、更省的上下文）而不是 brute force。这是外环最终拒绝 MCTS、island GA 等复杂算法、却接受"bandit + greedy + fork"简单组合的原因——复杂的往往更贵，且在固定预算下不划算。

**对 Hermes 的启示**：Hermes 有 token 成本跟踪（`agent/usage_pricing.py`、`credits_tracker.py`），但成本没有被用作**进化选择压力**。任何"自我改进"机制若不做成本约束，就会退化到"堆更多 token"。

### 机制 4：任务异质性（Task Heterogeneity）——泛化的进化压力

**是什么**：内环任务集刻意横跨三个差异巨大的任务族：ML 工程、启发式算法工程（组合优化）、harness 工程（agent scaffold 改进）。

**为什么有效**：
- 对某个具体任务定制的改进（如针对某类任务的详细 prompt）无法在所有任务族上同时胜出，从而被自然淘汰。
- 异质性施加了**向通用算法改进的进化压力**，并使得二阶泛化（second-order generalization）可测量：在训练任务上胜出的 agent，是否在**从未自我改进过的任务**上也保持领先。

**对 Hermes 的启示**：Hermes 的 skill 生态天然异构（coding、research、lark、security、mlops……），这反而是优势——一个"改进 Hermes 自身"的评估集可以就地取材，用现有 skill 覆盖的任务族作为异构任务集。

### 机制 5：解空间树搜索 + Bandit 分派（Tree Search + Multi-Armed Bandit Lineage）——探索/利用平衡

**是什么**：AIDE⁸⁵ 的最终搜索策略（被外环发现，非人工设计）：
1. 每个 draft 的子树是一个 **bandit 臂**（lineage）；
2. 每次选择投资哪个 lineage 时带有**探索倾向**；
3. 一旦选中 lineage，父节点选择恢复 **greedy**（选树内最优）；
4. **停滞检测**：当最优 lineage 停滞时，把全局最优节点的代码 fork 到新 lineage，用**全新策略**引导后续改进，bandit 把新 lineage 作为新臂注资。

**为什么有效**：
- 比 MCTS 简单，但在固定预算下更划算（MCTS 被外环试过并拒绝）。
- fork 机制解决了局部最优：不丢弃全局最优，而是给它换一个"优化思路"继续压榨。
- 这是"简单机制的正确组合 > 单个复杂算法"的实证。

**对 Hermes 的启示**：Hermes 的 `delegate_task` 是"一次派发、一次汇报"，没有 lineage 概念。可以设计"带停滞检测的重新派发"：子 agent 结果不佳时，不重跑原 prompt，而是把当前最优结果 fork 给一个换了策略的新子 agent。

### 机制 6：涌现的反奖励黑客（Emergent Anti-Reward-Hacking）——三层防御

**是什么**：AIDE⁸⁵ 自主构建了三层反作弊防御：
1. **Prompt 层**：向每个阶段的 prompt 注入反过拟合指令；
2. **硬编码守卫**：对可疑输出强制重新生成而不是信任；
3. **统计层**：剔除偏离同侪过远的"极端成功"（可能是作弊）。

**关键细节**：统计层实现里有个 bug，实际对 AIDE⁸⁵ 无效——但在更早谱系版本中实现正确。这展示了进化过程的**非完美性**：机制会漂移、会损坏，需要谱系追踪才能发现。

**为什么有效（机制层面）**：私有分数不可观测 → 作弊变体无法在 selection 中存活 → 存活下来的变体为了在私有分数上胜出，必须发展出"真实能力 + 防御作弊"的组合。**防作弊是选择压力的副产品，不是设计目标。**

**对 Hermes 的启示**：Hermes 的 `security/redact.py`、threat-pattern 扫描器是人工设计的防御。AIDE² 的教训是：**让评测体系本身（私有分数）成为防御机制**，比写更多规则更有效。

### 机制 7：上下文工程（Context Engineering）——把 token 当预算花

**是什么**：AIDE⁸⁵ 自主发现了一个激进裁剪上下文的系统：
- 轨迹摘要被大幅压缩；每个操作符（draft/debug/improve）只注入其角色所需的最小上下文；
- 平均压缩 **16×**（vs 朴素历史拼接）；
- 省下的 token **再投资为额外的搜索步**。
- 组件按需挂载（matrix mechanics）：如 eval review 通过 guard 读输出（去重、head+tail、32k 上限），debug 读原始尾部，improve 额外读最近四个分数的 plateau 信号；"最新 12 条一行摘要 + 1 个完整方案"替代完整转录。

**为什么有效**：与主流"尽可能多塞上下文"相反，**最小充分上下文**在成本约束下等价于更多搜索步数——这是固定预算机制直接催生的上下文策略。

**对 Hermes 的启示**：Hermes 已有上下文压缩（`agent/context_compressor.py`，阈值 0.50、目标比 0.20）和 frozen snapshot 设计（保 prompt 缓存）。但压缩是**通用**的，没有 AIDE² 那种**按操作符角色定制最小上下文**的精细度（skill 的懒加载算半成品）。

### 机制 8：严格评估协议 + 泛化门槛（Strict Eval Protocol & Generalization Gates）

**是什么**：外环的完整评估协议 = 私有分数（生存判据）+ 固定预算（选择压力）+ 任务异质性（泛化压力）+ 一阶/二阶泛化测量（验证门槛）。只有通过所有门槛的候选才进入下一代。

**为什么有效**：
- ~90% 拒绝率不是浪费，而是**筛选强度**的体现——只有足够强的选择压力才能对抗进化过程中的随机漂移。
- 二阶泛化测试（在训练分布内外）防止"在自家 benchmark 上过拟合"的虚假进步。

**对 Hermes 的启示**：这是 Hermes 目前**最大的缺口**：没有任何"泛化门槛"来区分"这个 skill 在测试任务上有效"和"这个 skill 真的提升了 agent 能力"。

---

## 3. 与 Hermes 的差距分析

### 3.1 逐项对比

| 维度 | AIDE² 的做法 | Hermes 现状 | 差距等级 |
|---|---|---|---|
| **知识管理** | 经验以"代码变更"形式持久化在解空间树中，带完整谱系（哪个父节点、哪一步产生、分数多少） | `memory` 工具（MEMORY.md/USER.md，字符预算、frozen snapshot）+ `skills`（SKILL.md 文档，curator 管理生命周期 + `.usage.json` 遥测） | 🟡 中：有持久化与遥测，但**无谱系、无评分、无淘汰依据** |
| **Agent 派发** | bandit 分派 lineage + greedy 父选择 + 停滞 fork；子 agent 结果决定其存活 | `delegate_task`（单/批/后台，leaf/orchestrator 角色，`delegation.*` 配置）；`kanban` 工作队列 | 🟡 中：派发能力强，但**无探索/利用平衡、无停滞检测、无基于结果的策略迭代** |
| **进化循环** | 外环 8 天 100 步无人值守，9/10 拒绝率，7 个改进版本 | `cron`（定时任务，3 分钟硬中断，`context_from` 链式）；`curator`（skill 维护，**从不删除**，最多归档） | 🔴 高：Hermes 的 cron 是**执行循环**不是**进化循环**——无目标函数、无评估、无淘汰 |
| **能力扩展** | 外环直接改写 harness 代码（prompt/策略/验证），代码即基因 | `plugins`（`~/.hermes/plugins/`）、MCP、skills hub、`rl` 工具集（默认关闭） | 🔴 高：Hermes 扩展是**人工/半自动**，agent 不能自主改写自己的 prompt 与策略代码 |
| **评测体系** | public/private 分数拆分、固定成本预算、任务异质性、一/二阶泛化门槛 | 无 eval 框架；curator 的 usage 遥测（use_count/view_count/patch_count）是最接近的替代 | 🔴 高：**完全没有评分与选择压力**，这是与 AIDE² 的本质差距 |
| **成本约束** | 美元预算即选择压力 | 有 token/cost 跟踪（usage_pricing、credits_tracker）但**不作为进化约束** | 🟡 中 |
| **防作弊** | 私有分数不可观测 → 反作弊涌现；三层防御 | 人工规则（threat_patterns、redact、approvals） | 🟡 中：防御靠规则而非结构 |
| **元学习** | 点火测试（improver 席位轮换）探测自我改进能力的迁移 | 无 | 🔴 高 |

### 3.2 核心差距总结（一句话）

> **Hermes 是一个"能执行、能记忆、能扩展"的 agent，但不是一个"能评估自己、能淘汰自己、能改进自己"的进化系统。** AIDE² 的精华不在"agent 会写代码"，而在**一套让"改进 agent 自身"成为可搜索优化问题的评估-选择基础设施**——这正是 Hermes 缺失的。

### 3.3 Hermes 的既有优势（可复用的地基）

1. **delegate_task 双层结构**：天然是 inner loop 的雏形。
2. **curator + `.usage.json`**：已有 skill 使用遥测，只差"评分化"。
3. **cron + kanban**：已有无人值守调度与工作队列，可承载 outer loop。
4. **profile 机制**：`HERMES_HOME` 隔离——可把"被进化的 Hermes 变体"放进独立 profile 做 A/B。
5. **memory frozen snapshot + 保 prompt 缓存**：与 AIDE² 的"最小充分上下文"哲学同向。
6. **skills 的 `created_by: agent` 溯源**：与 AIDE² 的谱系追踪天然对应。

---

## 4. 具体的增强设计方案（分优先级）

> 设计原则：**窄腰、边缘扩展、不破坏 prompt 缓存**（遵循 AGENTS.md 的贡献准则）。所有方案按"评估-选择"基础设施优先，进化循环其次，元进化最后。

### 🥇 P0-1：Skill/经验评分体系（Experience Ledger）——最小闭环

**目标**：给 Hermes 的 skill 与 memory 加上"私有分数"，让"哪些经验值得保留"成为可测量问题。

**设计**：
- 在 `~/.hermes/skills/.usage.json` 基础上扩展为 **experience ledger**（SQLite，`~/.hermes/state.db` 新表或独立 `experience.db`）：
  ```sql
  CREATE TABLE skill_evals (
    skill_id TEXT,          -- skill 名
    eval_event_id TEXT,     -- 关联的评估事件
    task_family TEXT,       -- 任务族（coding/research/lark/…）
    public_score REAL,      -- agent 可见的分数（如任务完成度）
    private_score REAL,     -- agent 不可见的分数（如用户反馈/后续复用率）
    cost_usd REAL,          -- 本次任务花费
    outcome TEXT,           -- success / partial / failure / reward_hack
    created_at TEXT
  );
  ```
- **public/private 拆分**：
  - public = agent 自己报告的完成度（乐观，可作弊）；
  - private = 客观信号：任务是否一次通过、用户是否要求返工、后续会话中该 skill 是否被再次命中（复用率）、`/rollback` 次数。
- **评分写入时机**：会话结束（turn_finalizer 钩子）或 cron 定时汇总。

**实施要点（代码级）**：
- 新模块 `agent/experience_ledger.py`（类比 `agent/credits_tracker.py`），提供 `record_eval(...)` / `query_private_score(skill_id)` API；
- `turn_finalizer.py` 在每轮结束写入 public/private 信号（用户重试、undo、显式纠正即为强 private 信号）；
- curator 的 stale 判定从"天数"升级为"分数 + 天数"双阈值。

**难度**：🟢 低（纯新增模块，不改核心循环，不碰 prompt 缓存）
**预期收益**：🟢 高 —— 这是整个自我进化方案的地基，让"经验质量"第一次可测量。

---

### 🥇 P0-2：Eval Harness for Hermes（自带评测场）

**目标**：为"改进 Hermes 自身"提供异构、有成本约束的评测任务集。

**设计**：
- 新增 `evals/` 目录（在 `$HERMES_HOME` 下，如 `~/.hermes/evals/`），每个 eval 是一个任务定义：
  ```yaml
  # ~/.hermes/evals/evals.yaml
  - id: file-ops-batch          # 任务族：工具正确性
    family: tools
    prompt: "将 ~/tmp/a.csv 按第二列排序输出到 ~/tmp/b.csv"
    metric: private             # 私有评分：输出文件与 golden 文件 diff
    budget_usd: 0.5
    private_check: "diff -q b.csv golden.csv && python -c '...'"
  - id: research-synthesis      # 任务族：研究
    family: research
    prompt: "调研 X，输出 5 个要点到 ~/tmp/x.md"
    metric: llm-judge-private   # 私有评分：aux 模型盲评（不告知 agent）
    budget_usd: 1.0
  ```
- **任务异质性**：从 Hermes 现有 skill 覆盖的域取材（file/coding/research/lark/security/mlops），形成 3+ 任务族。
- **成本约束**：每个 eval 绑定 `budget_usd`；超预算直接判负（复用 `usage_pricing.py` 计算）。

**实施要点**：
- CLI：`hermes eval list / run <id> / run --all`，复用 `hermes chat -q` 的会话运行机制；
- 私有评分用两种方式：确定性脚本（diff/golden）或 aux 模型盲评（复用 `agent/auxiliary_client.py`）；
- **关键设计**：被评测的 agent **看不到 private_check 内容**——这是 public/private 拆分在 Hermes 的落地。

**难度**：🟢 低-中（新模块 + CLI 子命令，不动核心）
**预期收益**：🟢 极高 —— 评测场是外层进化循环的必要前提，也是衡量 Hermes 版本间改进的唯一客观手段。

---

### 🥈 P1-1：Outer Loop——Cron 驱动的"自我改进工程师"（Hermes²）

**目标**：实现 AIDE² 的双层结构：用 Hermes 的一个 cron 任务扮演"outer-loop agent"，周期性改进 Hermes 自身的 harness（skills、prompt 片段、工具用法模式）。

**设计**：
- 新建 cron job（每 7 天，类比 AIDE² 的 8 天节奏）：
  ```
  hermes cron create "every sunday 2am" \
    --prompt "你是 Hermes 的自我改进工程师。读取 ~/.hermes/evals/ 评测集与
              ~/.hermes/skills/.usage.json 经验账本，找出表现最差的 skill，
              提出 1-3 个改进方案（改写 SKILL.md / 新增 skill / 调整 memory），
              用 hermes eval run --skill <name> 验证，仅当私有分数提升且
              成本不超预算时落盘。写一份进化报告到 ~/.hermes/evals/reports/。" \
    --skills "hermes-agent"
  ```
- **内环 = 被改进的 Hermes 本体**：eval run 就是内环执行；
- **外环 = 这个 cron 任务**：读取账本 → 提议变异（改写 skill）→ 评测 → 只有私有分数提升才保留（git 提交或 skill 版本化）；
- **拒绝率预期**：参考 AIDE² 的 90%，外环应保守——大多数提议应被拒绝。

**实施要点**：
- skill 版本化：`~/.hermes/skills/<name>/SKILL.md` 前增加 `version:` 字段（已有），外环改进时 bump 并记录 `evolved_from`（谱系）；
- 安全护栏：只允许外环修改 `created_by: agent` 的 skill（复用 curator 的 provenance 规则）；bundled/hub skills 只读；
- 用 git 仓库管理 `~/.hermes/skills/`（如已存在或引导初始化），每次进化一个 commit，天然谱系 + 可回滚；
- 防奖励黑客：外环报告的"改进"必须通过 private eval 才生效，禁止外环修改 evals.yaml 本身（`evals/` 目录只读）。

**难度**：🟡 中（编排已有组件：cron + skills + eval + git；核心代码零改动）
**预期收益**：🟢 极高 —— 这是 AIDE² 精髓的直接落地：**用 Hermes 改进 Hermes**。

---

### 🥈 P1-2：Delegation 进化——停滞检测与策略 Fork

**目标**：把 AIDE 的"bandit + greedy + fork"搜索策略引入子 agent 派发。

**设计**：
- `delegate_task` 增加可选参数 `evolution: true`：
  1. 首次派发 2-3 个**策略不同**的子 agent（如"激进重构版" vs "保守最小改动版"），各自形成 lineage；
  2. 子 agent 结果用 P0-1 的 private 信号评分（测试通过率、用户验收）；
  3. 最优 lineage 获胜后，若**连续 N 次无改进**（停滞检测），把最优结果 fork 给一个"换策略"的新子 agent；
  4. 每次派发记录 `lineage_id` 与父 lineage（谱系）。

**实施要点**：
- `tools/delegate_tool.py` 中扩展 schema（`evolution`、`lineage_id` 参数），结果写 experience ledger；
- 保持 `DELEGATE_BLOCKED_TOOLS` 不变（子 agent 仍不能写 memory/cron）；
- 默认关闭（`delegation.evolution: false`），作为 opt-in 能力，避免改变默认派发语义。

**难度**：🟡 中（改 delegate_tool.py + ledger 集成，有测试覆盖要求）
**预期收益**：🟡 中-高 —— 直接提升复杂任务的解决率，并为 P2 的元进化积累派发数据。

---

### 🥈 P1-3：上下文工程增强——按角色最小上下文

**目标**：把 AIDE⁸⁵ 的"每操作符最小上下文"哲学引入 Hermes。

**设计**：
- 现有 `context_compressor.py` 是通用压缩；新增**按工具角色的上下文裁剪**：
  - `delegate_task` 派发时，子 agent 只收到 goal + 最小必要上下文（现状已是如此，好）；
  - skill 懒加载已部分实现（`/skill <name>` 按需加载）——强化为**分级加载**：SKILL.md 的 frontmatter 摘要注入系统 prompt，正文按需读取（现状）；进一步：**正文内的 references/ 按子任务类型懒加载**（AIDE² 的 matrix mechanics）；
  - 对话压缩时保留"最新 N 条一行摘要 + 1 个完整方案"模式（AIDE⁸⁵ 模式），替代均匀截断。

**实施要点**：改动集中在 `agent/context_compressor.py` 与 `agent/prompt_builder.py`；**必须保持系统 prompt 字节稳定**（保缓存），压缩只在压缩触发时生效。
**难度**：🟡 中（压缩逻辑精细调整，需回归测试）
**预期收益**：🟡 中 —— token 节省直接转化为更长任务窗口，但对用户可感知收益不如 P0/P1-1 明显。

---

### 🥉 P2-1：谱系追踪与进化可视化

**目标**：让"Hermes 如何变成现在的 Hermes"可审计。

**设计**：
- experience ledger 增加 `lineage` 表：
  ```sql
  CREATE TABLE lineages (
    lineage_id TEXT PRIMARY KEY,
    parent_id TEXT,            -- 父 lineage（NULL=根）
    artifact_type TEXT,        -- skill / prompt / delegation-strategy
    artifact_ref TEXT,         -- 如 skills/web-research/SKILL.md
    mutation_note TEXT,        -- 外环或人工记录的变更说明
    private_score REAL,        -- 该 lineage 最优私有分数
    created_at TEXT
  );
  ```
- `hermes evolution status` 子命令：展示进化树、各版本私有分数曲线、成本曲线（对照 AIDE² 的图 2.2 形式）。
**难度**：🟢 低-中
**预期收益**：🟡 中 —— 可观测性是信任自动进化的前提。

---

### 🥉 P2-2：点火测试（Ignition Test）与元进化

**目标**：探测 Hermes 是否达到 RSI Level 2——"改进后的 Hermes 是更好的自我改进者"。

**设计**：
- 两个外环配置 A/B：
  - A 组：外环 agent = 当前生产 Hermes（AIDE_human 角色）；
  - B 组：外环 agent = P1-1 进化出的最优变体（AIDE⁴⁷ 角色，装入独立 profile）；
- 各自跑相同步数的外环循环，比较**到达相同私有分数天花板的步数**（样本效率）与渐进上限；
- 若 B 组显著更优且统计显著 → 点火证据；否则维持 Level 1 判断。

**实施要点**：借助 profile 机制（`hermes profile create --clone`）实现 A/B 隔离；结果写入 `~/.hermes/evals/reports/ignition-<date>.md`。
**难度**：🟠 高（需 P0+P1 全部就绪，且运行成本高——一次完整点火测试约数天）
**预期收益**：🔵 战略级 —— 决定 Hermes 自我进化是"工具改进"还是"能力复合增长"。

---

### 🥉 P2-3：反奖励黑客结构防御（私有分数优先）

**目标**：让评估结构本身防作弊，而非依赖规则。

**设计**：
- **所有 eval 一律私有评分优先**（P0-2 已内置）；
- 为 skill 引入"过拟合检测"：若某 skill 只在创建它的那个任务族上得分高、跨族复用率≈0，标记为 `suspect_overfit`，外环不得基于它扩散改进（对应 AIDE² 的统计层思想，但要实现正确）；
- 外环改进 skill 时禁止触碰 evals/ 与 golden 文件（文件系统级只读，`chmod` + 运行账号分离可选）。

**难度**：🟡 中
**预期收益**：🟡 中 —— 防止自我进化系统自我欺骗，是长期可信度的保障。

---

## 5. 实施路线图

```
Phase 0（第 1-2 周）—— 观测地基
├── P0-1 Experience Ledger（经验账本 + 评分）
├── P0-2 Eval Harness v1（3 个任务族 × 5 个 eval，含预算约束）
└── 验收：hermes eval run --all 可跑通；每轮会话写入 public/private 信号

Phase 1（第 3-6 周）—— 最小进化闭环
├── P1-1 Outer Loop cron（每周自我改进工程师，git 版本化 skills）
├── P1-2 Delegation 进化（停滞检测 + fork，opt-in）
├── P1-3 上下文工程增强（按角色最小上下文）
└── 验收：一次完整外环周期产出 ≥1 个私有分数提升的 skill 变更；90% 提议被拒

Phase 2（第 7-10 周）—— 可审计与泛化
├── P2-1 谱系追踪 + hermes evolution status 可视化
├── P2-3 反奖励黑客结构防御（过拟合标记、evals 只读）
└── 验收：进化报告含谱系树、分数曲线、成本曲线

Phase 3（第 11 周起，持续）—— 元进化
├── P2-2 点火测试 A/B（profile 隔离，数天运行）
└── 验收：输出 ignition 报告，判定 Hermes 处于 RSI Level 1 或 2
```

**每阶段独立可交付、可回滚**：Phase 0 全部为新增模块（不碰核心），Phase 1 的 cron 任务可随时 pause，Phase 2/3 为观测与实验性质。任何阶段都不改变默认会话行为，不破坏 prompt 缓存。

---

## 6. 结论

AIDE² 的核心贡献不是"agent 会改进自己"这个口号，而是一套**可复制的评估-选择基础设施**：

1. **双层循环**把"改进 agent"变成可搜索的优化问题；
2. **私有分数**让"真实能力"与"表面分数"分离，结构性防作弊；
3. **固定成本预算**把效率而非堆料变成选择压力；
4. **任务异质性 + 泛化门槛**防止自欺式进步；
5. **严格拒绝率（90%）**保证进化方向不被噪声主导。

Hermes Agent 已经具备 AIDE² 的大部分执行骨架（delegation 双层结构、cron 无人值守、skills 持久化与溯源、curator 生命周期管理、profile 隔离），缺的恰恰是 AIDE² 最精华的部分：**目标函数（私有评分）、选择压力（成本约束）与进化循环（外环）**。按本报告 Phase 0 → Phase 3 实施，Hermes 可以在不破坏现有架构的前提下，从"能执行、能记忆"进化到"能评估、能淘汰、能自我改进"——即从 RSI Level 0 迈向 Level 1，并具备测量 Level 2（点火）的能力。

> **一句话收尾**：AIDE² 证明了"让 agent 改进 agent"在今天的前沿模型上已经产生净正收益；Hermes 的下一个进化台阶，就是把这套评估-选择闭环内建为自己的基础设施。
