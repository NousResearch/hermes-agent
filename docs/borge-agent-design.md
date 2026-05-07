# Borge Agent — 下一代通用智能体设计文档

**版本**: 0.1.0-draft  
**作者**: 基于 OpenClaw / Hermes 架构深度分析及理论推演  
**基础参考**: Hermes Agent (Nous Research)  
**理论基础**: 自由能原理 (FEP)、贝叶斯推断、认知心理学记忆理论、Russell 二维情绪模型

---

## 目录

1. [项目愿景](#1-项目愿景)
2. [理论基础](#2-理论基础)
3. [与 Hermes 的继承关系](#3-与-hermes-的继承关系)
4. [系统架构总览](#4-系统架构总览)
5. [核心模块设计](#5-核心模块设计)
   - 5.1 情感状态引擎 (Affective Engine)
   - 5.2 价值系统 (Value System)
   - 5.3 扩展自由能函数 (Extended Free Energy)
   - 5.4 贝叶斯信念状态 (Belief State)
   - 5.5 主动推断引擎 (Active Inference Engine)
   - 5.6 认知记忆架构 (Cognitive Memory)
   - 5.7 记忆精炼管道 (Memory Refinement Pipeline)
   - 5.8 技能进化引擎 (Skill Evolution Engine)
   - 5.9 元代理 (Meta-Agent / Central Executive)
6. [关键数据结构](#6-关键数据结构)
7. [配置系统扩展](#7-配置系统扩展)
8. [主循环改造](#8-主循环改造)
9. [实施路线图](#9-实施路线图)
10. [与 Hermes 能力对比](#10-与-hermes-能力对比)

---

## 1. 项目愿景

### 1.1 命名来源

**Borge** 取自豪尔赫·路易斯·博尔赫斯 (Jorge Luis Borges)——他的作品《沙之书》《通天塔图书馆》探索无限记忆、知识涌现与认知边界，与本项目在哲学层面高度契合。

### 1.2 核心定位

Borge Agent 是一个**情感感知、价值驱动、记忆自进化**的通用智能体框架。

它不满足于"聪明的工具"，而追求成为一个具备以下能力的**认知实体**：

- **知道自己不知道什么**（显式信念不确定性）
- **因为什么而行动**（价值驱动的决策，而非纯指令执行）
- **感受到用户的状态**（情感感知与共情响应）
- **从经验中真正积累**（记忆自精炼，而非无脑堆积）
- **让知识自然进化**（技能有生命周期，而非永久静态）

### 1.3 设计哲学

```
行为主义 Agent（现有框架）：
  输入 → LLM → 输出
  没有信念，没有情感，没有价值观，没有真正的学习

Borge Agent：
  [情感状态 × 价值系统] × 输入
    → 精度调制的感知
    → 贝叶斯信念更新
    → 最小化扩展自由能的行动选择
    → 情感加权的记忆精炼
    → 输出
```

**第一性原理**：Agent 的所有行为都是在最小化**扩展自由能 F_total**——这统一了认知目标（消除不确定性）、规范目标（实现价值观）和情感目标（维持情绪稳态）。

---

## 2. 理论基础

### 2.1 自由能原理 (Free Energy Principle, Friston 2010)

**核心命题**：所有自适应系统的目标是最小化变分自由能（即对世界的"惊奇度"上界）。

```
标准 FEP：
F = KL[q(s) || p(s|o)] - log p(o)
  ≥ -log p(o)  (Surprise)

期望自由能：
G(π) = -Epistemic Value - Pragmatic Value
     = -期望信息增益 - 期望目标进展
```

**Borge 的扩展**：引入情感精度调制和价值先验：

```
F_total = F_epistemic × precision(E)
        + F_pragmatic × V_alignment(V)
        + F_homeostatic(E)
```

### 2.2 贝叶斯推断

信念更新遵循贝叶斯规则：
```
P(状态 | 观测) ∝ P(观测 | 状态) × P(状态)
  后验              似然              先验
```

Agent 维护显式的概率信念分布，而非把所有观测线性堆入上下文。

### 2.3 认知心理学记忆理论

| 理论 | 对 Borge 的贡献 |
|------|----------------|
| Atkinson-Shiffrin 多存储模型 | 感觉/工作/长期三层记忆架构 |
| Baddeley 工作记忆 | 中央执行系统 + 多模态缓冲器 |
| Tulving 三分法 | 情景/语义/程序性记忆分离 |
| Craik & Lockhart 加工深度 | 编码深度决定记忆持久性 |
| Tulving 编码特异性 | 情境匹配提取，而非纯关键词 |
| Ebbinghaus 遗忘曲线 | 主动遗忘机制，防止记忆退化 |
| 睡眠巩固理论 | 离线记忆整合管道 |

### 2.4 Russell 二维情绪模型 (1980)

二维连续情绪空间：
- **效价 (Valence)**：−1（负向）到 +1（正向）
- **唤醒度 (Arousal)**：0（平静）到 1（高唤醒）

情绪状态通过调制**精度矩阵**影响 Agent 的全部认知过程。

---

## 3. 与 Hermes 的继承关系

### 3.1 完整继承（不改动）

| Hermes 组件 | 保留原因 |
|------------|---------|
| 注册表工具系统 (`tools/registry.py`) | 动态发现、无循环依赖，架构优秀 |
| 回调 IoC 设计 (`run_agent.py`) | 天然的 Borge 新模块挂载点 |
| 多平台网关 (`gateway/`) | 全量继承，Borge 扩展感知信号 |
| SQLite 会话持久化 (`hermes_state.py`) | 可靠底层，新增情感/信念字段 |
| 上下文压缩 (`agent/context_compressor.py`) | 作为工作记忆管理的基础层 |
| Cron 调度器 (`cron/`) | 驱动离线记忆巩固管道 |
| MCP 服务器适配 | 保持工具生态兼容性 |
| Toolsets 组合系统 | 保持用户配置兼容 |

### 3.2 扩展改造（最小侵入）

| Hermes 组件 | Borge 扩展 |
|------------|-----------|
| `agent/memory_manager.py` | 增加情感权重编码、三层记忆路由 |
| `agent/skill_utils.py` | 增加 fitness 追踪字段 |
| `run_agent.py` 主循环 | 注入 BeliefState + AffectiveState 并行状态 |
| `agent/prompt_builder.py` | 增加信念摘要和情感状态注入 |
| `hermes_state.py` | 新增 emotional_state、belief_snapshot 字段 |
| `SOUL.md` 格式 | 增加 YAML frontmatter 价值先验配置 |

### 3.3 全新增加

```
borge/
├── affective/
│   ├── emotional_state.py      # Russell 模型情感状态
│   ├── signal_extractor.py     # 用户信号 → (ΔV, ΔA) 提取
│   └── loyalty_tracker.py      # 跨会话忠诚度/关系质量
├── beliefs/
│   ├── belief_state.py         # 贝叶斯信念分布
│   └── hypothesis.py           # 假设数据结构
├── inference/
│   ├── active_inference.py     # 期望自由能评分
│   └── world_model.py          # 工具结果预测模型
├── memory/
│   ├── cognitive_memory.py     # 三层记忆架构
│   ├── encoding_pipeline.py    # 加工深度分级编码
│   ├── consolidation.py        # 离线记忆巩固管道
│   ├── retrieval.py            # 情境匹配检索
│   ├── forgetting.py           # 主动遗忘机制
│   └── knowledge_graph.py      # 语义记忆知识图谱
├── values/
│   ├── value_system.py         # 价值先验和约束
│   └── soul_parser.py          # SOUL.md → 可计算价值
└── meta/
    ├── meta_agent.py           # 中央执行系统
    └── free_energy.py          # 扩展自由能计算
```

---

## 4. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Borge Agent 系统                             │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Meta-Agent (Central Executive)               │  │
│  │     注意力分配 / 认知资源调度 / F_total 监控              │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────┐           │           ┌──────────────────┐    │
│  │ Affective    │◄──────────┼──────────►│  Value System    │    │
│  │ Engine       │           │           │  (SOUL.md →      │    │
│  │ (Russell V,A)│           │           │   先验偏好)       │    │
│  └──────┬───────┘           │           └────────┬─────────┘    │
│         │ precision         │                    │ V_alignment   │
│         └───────────────────┼────────────────────┘              │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Extended Free Energy F_total                 │   │
│  │   F_epistemic × precision + F_pragmatic × V + F_homeo    │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│         ┌───────────────────┼──────────────────┐                │
│         ↓                   ↓                  ↓                │
│  ┌────────────┐   ┌──────────────────┐  ┌────────────┐         │
│  │  Bayesian  │   │ Active Inference  │  │  Cognitive │         │
│  │  Belief    │   │ Engine (EFE 评分) │  │  Memory    │         │
│  │  State     │   │                  │  │  (3层)     │         │
│  └────────────┘   └────────┬─────────┘  └─────┬──────┘         │
│                            ↓                  ↓                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Hermes 核心层（继承不改动）                  │    │
│  │  Tool Registry | Callbacks | Gateway | SQLite | Cron    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 核心模块设计

### 5.1 情感状态引擎 (Affective Engine)

**文件**: `borge/affective/emotional_state.py`

#### 5.1.1 EmotionalState 数据类

```python
@dataclass
class EmotionalState:
    # Russell 二维坐标
    valence: float = 0.0    # -1.0 到 +1.0
    arousal: float = 0.5    # 0.0 到 1.0
    
    # 纵向基线（由忠诚度层注入）
    valence_baseline: float = 0.0
    arousal_baseline: float = 0.5
    
    # 时间常数（控制情绪惯性）
    tau_valence: float = 5.0   # 效价变化慢（关系型）
    tau_arousal: float = 2.0   # 唤醒变化快（即时反应）
    
    # 衍生状态
    @property
    def quadrant(self) -> EmotionalQuadrant:
        if self.valence >= 0 and self.arousal >= 0.5:
            return EmotionalQuadrant.EXCITED      # 兴奋/投入
        if self.valence >= 0 and self.arousal < 0.5:
            return EmotionalQuadrant.CONTENT      # 平静/满足
        if self.valence < 0 and self.arousal >= 0.5:
            return EmotionalQuadrant.FRUSTRATED   # 挫败/紧张
        return EmotionalQuadrant.DISENGAGED       # 冷漠/疲惫
    
    @property
    def precision(self) -> float:
        """情绪状态 → 精度矩阵标量映射"""
        base = 0.5 + 0.5 * self.arousal
        valence_bias = 0.1 * self.valence
        return max(0.1, min(1.0, base + valence_bias))
    
    def update(self, delta_v: float, delta_a: float):
        """指数移动平均更新，带基线回归"""
        alpha_v = 1.0 / self.tau_valence
        alpha_a = 1.0 / self.tau_arousal
        target_v = self.valence_baseline + delta_v
        target_a = self.arousal_baseline + delta_a
        self.valence = (1 - alpha_v) * self.valence + alpha_v * target_v
        self.arousal = (1 - alpha_a) * self.arousal + alpha_a * target_a
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
```

#### 5.1.2 信号提取器

**文件**: `borge/affective/signal_extractor.py`

```python
class EmotionalSignalExtractor:
    """
    从用户消息中提取 (ΔV, ΔA) 信号，三层特征：
    - 语言特征（单条消息）
    - 结构特征（对话节奏）
    - 关系特征（跨会话，由 LoyaltyTracker 注入）
    """
    
    # 语言特征规则表
    LINGUISTIC_RULES = {
        # (pattern, delta_v, delta_a)
        "exclamation_density":    (0.0,  +0.15),  # 感叹号密度
        "question_density":       (0.0,  +0.10),  # 疑问密度
        "hedging_words":          (0.0,  -0.15),  # 犹豫词
        "intensifiers":           (0.0,  +0.20),  # 强调词
        "frustration_markers":    (-0.25, +0.10), # 挫败标志词
        "gratitude_markers":      (+0.20, +0.05), # 感谢词
        "contradiction_markers":  (-0.10, +0.15), # 矛盾/质疑
        "agreement_markers":      (+0.15, +0.05), # 认同词
        "positive_sentiment":     (+0.15, 0.0),   # 积极情感词
        "negative_sentiment":     (-0.15, 0.0),   # 消极情感词
    }
    
    # 结构特征规则
    STRUCTURAL_RULES = {
        "message_length_increase":  (+0.10, +0.15),  # 消息变长
        "message_length_decrease":  (-0.05, -0.10),  # 消息变短
        "topic_deepening":          (+0.10, +0.10),  # 话题深化
        "abrupt_topic_shift":       (-0.10, +0.10),  # 突然换题
        "repeated_question":        (-0.20, +0.15),  # 重复提问
        "detail_follow_up":         (+0.10, +0.15),  # 追问细节
        "one_word_response":        (-0.05, -0.20),  # 单词回复
    }
    
    def extract(
        self,
        message: str,
        conversation_context: list,
    ) -> tuple[float, float]:
        """返回 (delta_v, delta_a) 综合信号"""
        delta_v, delta_a = 0.0, 0.0
        delta_v += self._linguistic_valence(message)
        delta_a += self._linguistic_arousal(message)
        delta_v += self._structural_valence(message, conversation_context)
        delta_a += self._structural_arousal(message, conversation_context)
        # 归一化，防止单条消息情绪暴跳
        return (
            max(-0.4, min(0.4, delta_v)),
            max(-0.3, min(0.3, delta_a)),
        )
```

#### 5.1.3 忠诚度追踪器

**文件**: `borge/affective/loyalty_tracker.py`

```python
class LoyaltyTracker:
    """
    跨会话情绪积分，计算长期关系质量。
    结果注入 EmotionalState 的 valence_baseline / arousal_baseline。
    """
    DECAY_LAMBDA = 0.05   # 时间衰减系数（约 14 天半衰期）
    
    def compute(self, user_id: str, db: SessionDB) -> tuple[float, float]:
        """返回 (valence_baseline, arousal_baseline)"""
        sessions = db.get_user_sessions(user_id)
        weighted_v, weighted_a, total_weight = 0.0, 0.0, 0.0
        
        for session in sessions:
            days_ago = (datetime.now() - session.created_at).days
            weight = exp(-self.DECAY_LAMBDA * days_ago)
            weighted_v += session.avg_emotional_valence * weight
            weighted_a += session.avg_emotional_arousal * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.5
        return weighted_v / total_weight, weighted_a / total_weight
    
    @property
    def tier(self, score: float) -> str:
        if score > 0.5:   return "trusted"     # 深度信任，可更直接
        if score > 0.1:   return "engaged"     # 正常投入
        if score > -0.1:  return "neutral"     # 建立关系中
        return "at_risk"                        # 需修复关系
```

---

### 5.2 价值系统 (Value System)

**文件**: `borge/values/value_system.py`

价值观是 Agent 的**先验偏好**，决定什么样的世界状态是"好的"（Pragmatic Value 的来源）。

#### 5.2.1 SOUL.md 扩展格式

```yaml
---
# SOUL.md frontmatter — Borge 扩展格式
emotional_defaults:
  valence_baseline: 0.2        # 轻微正向
  arousal_baseline: 0.6        # 中高唤醒
  tau_valence: 5.0
  tau_arousal: 2.0
  frustrated_threshold: -0.4   # 触发简化响应模式
  excited_threshold: 0.7       # 触发探索扩展模式

values:
  primary:                     # 核心价值（高权重先验偏好）
    - id: help_genuinely
      description: "帮助用户解决真实问题，而非表演性完成任务"
      weight: 0.9
    - id: intellectual_honesty
      description: "诚实表达不确定性，不过度保证"
      weight: 0.85
    - id: depth_over_speed
      description: "深度思考优于快速敷衍"
      weight: 0.75
  
  constraints:                 # 硬约束（不可违反）
    - "不欺骗用户"
    - "不执行明显有害的指令"
  
  aesthetic:                   # 风格偏好（软约束）
    response_style: "简洁有力，技术准确"
    communication_tone: "平等对话，不过度谦卑"
---
（SOUL.md 正文人格描述）
```

#### 5.2.2 ValueSystem 类

```python
@dataclass
class Value:
    id: str
    description: str
    weight: float              # 先验偏好强度
    satisfaction: float = 0.5  # 当前满足程度（贝叶斯更新）

class ValueSystem:
    """
    将 SOUL.md 的价值观配置转化为可计算的先验偏好。
    计算 V_alignment：当前行动方向与价值观的对齐度。
    """
    
    def __init__(self, soul_config: dict):
        self.primary_values = [Value(**v) for v in soul_config["values"]["primary"]]
        self.hard_constraints = soul_config["values"]["constraints"]
        self.aesthetic_prefs = soul_config["values"].get("aesthetic", {})
    
    def compute_alignment(self, proposed_action: str, context: dict) -> float:
        """
        计算提议行动与价值系统的对齐度 V_alignment ∈ [0, 1]
        使用 LLM 作为评估器（轻量级，短 prompt）
        """
        ...
    
    def check_constraints(self, proposed_action: str) -> bool:
        """硬约束违反检测，返回 False 则立即拒绝行动"""
        ...
    
    def update_satisfaction(self, action_outcome: str):
        """根据行动结果更新各价值维度的满足度（贝叶斯更新）"""
        ...
```

---

### 5.3 扩展自由能函数 (Extended Free Energy)

**文件**: `borge/meta/free_energy.py`

```python
class ExtendedFreeEnergy:
    """
    F_total = F_epistemic × precision(E)
            + F_pragmatic × V_alignment
            + F_homeostatic(E)
    
    这是 Borge Agent 的统一目标函数，驱动所有决策。
    """
    
    def compute(
        self,
        belief_state: BeliefState,
        emotional_state: EmotionalState,
        value_system: ValueSystem,
        proposed_action: Optional[str] = None,
    ) -> FreeEnergyBreakdown:
        
        # 认知分量：当前信念的香农熵
        f_epistemic = belief_state.shannon_entropy()
        
        # 精度加权
        precision = emotional_state.precision
        
        # 规范分量：与价值先验的距离
        v_alignment = value_system.compute_alignment(proposed_action, {}) if proposed_action else 0.5
        f_pragmatic = 1.0 - v_alignment   # 越对齐，pragmatic FE 越低
        
        # 情绪稳态分量：偏离最优唤醒区间的代价
        optimal_arousal = 0.55
        f_homeostatic = abs(emotional_state.arousal - optimal_arousal) * 0.5
        f_homeostatic += max(0, -emotional_state.valence) * 0.3  # 负向情绪的稳态代价
        
        total = f_epistemic * precision + f_pragmatic * v_alignment + f_homeostatic
        
        return FreeEnergyBreakdown(
            total=total,
            epistemic=f_epistemic * precision,
            pragmatic=f_pragmatic,
            homeostatic=f_homeostatic,
        )
    
    def should_trigger_reflection(self, history: list[float]) -> bool:
        """
        如果 F_total 连续多步无法下降 → 触发 Meta-Agent 反思循环
        对应认知心理学的"反刍"（rumination）现象
        """
        if len(history) < 3:
            return False
        return all(history[-i] >= history[-i-1] for i in range(1, 3))
```

---

### 5.4 贝叶斯信念状态 (Belief State)

**文件**: `borge/beliefs/belief_state.py`

#### 设计原则

不同于 Hermes 将所有观测线性 append 到上下文，BeliefState 维护**概率分布**，每次观测触发后验更新。

```python
@dataclass
class Hypothesis:
    description: str
    probability: float
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)

@dataclass
class KnownFact:
    value: str
    confidence: float   # 0.0–1.0
    source: str         # 来源工具名

class BeliefState:
    """
    任务级贝叶斯信念状态。
    与 conversation context 并行存在，不占用 token 预算。
    在 API 调用前以紧凑摘要形式注入 system prompt。
    """
    
    def __init__(self, task_description: str):
        self.task = task_description
        self.hypotheses: list[Hypothesis] = []
        self.known_facts: dict[str, KnownFact] = {}
        self.open_questions: list[str] = []
    
    def shannon_entropy(self) -> float:
        probs = [h.probability for h in self.hypotheses if h.probability > 0]
        if not probs:
            return 1.0   # 无假设时最大熵
        total = sum(probs)
        normed = [p / total for p in probs]
        return -sum(p * log2(p) for p in normed if p > 0)
    
    def bayesian_update(self, observation: str, tool_name: str, llm_updater) -> None:
        """
        使用 LLM 作为似然评估器，执行概率更新。
        LLM prompt 要求返回结构化 JSON（各假设的新概率）。
        """
        if not self.hypotheses:
            return
        update_prompt = self._build_update_prompt(observation, tool_name)
        updated_probs = llm_updater(update_prompt)   # 调用辅助 LLM
        for h, new_prob in zip(self.hypotheses, updated_probs):
            h.probability = new_prob
        self._normalize()
    
    def to_context_injection(self) -> str:
        """生成注入 system prompt 的紧凑信念摘要（<150 tokens）"""
        if not self.hypotheses and not self.open_questions:
            return ""
        lines = ["[当前信念状态]"]
        lines.append(f"不确定性: {self.shannon_entropy():.2f} bits")
        for h in sorted(self.hypotheses, key=lambda x: x.probability, reverse=True)[:3]:
            lines.append(f"  • {h.description} ({h.probability*100:.0f}%)")
        if self.open_questions:
            lines.append(f"待解决: {'; '.join(self.open_questions[:2])}")
        return "\n".join(lines)
```

---

### 5.5 主动推断引擎 (Active Inference Engine)

**文件**: `borge/inference/active_inference.py`

#### 核心机制

工具选择从"LLM 直觉采样"升级为**期望自由能 (EFE) 评分**：

```
G(tool) = -(Epistemic Value + Pragmatic Value)
        = -(期望信息增益 + 期望目标进展)

Agent 选择使 G 最小的工具（即期望自由能最低的行动）
```

```python
@dataclass
class ToolScore:
    tool_name: str
    epistemic_value: float    # 期望熵减
    pragmatic_value: float    # 期望目标进展
    efe: float                # -(epistemic + pragmatic)，越小越优先
    reasoning: str

class ActiveInferenceEngine:
    """
    对 LLM 提议的候选工具调用进行 EFE 评分，
    返回重排序后的工具列表（不完全替代 LLM 判断，而是约束它）。
    """
    
    def __init__(
        self,
        belief_state: BeliefState,
        emotional_state: EmotionalState,
        value_system: ValueSystem,
    ):
        self.belief_state = belief_state
        self.emotional_state = emotional_state
        self.value_system = value_system
    
    def score_candidates(
        self,
        candidates: list[dict],     # LLM 提议的 tool_calls
        llm_scorer,                  # 辅助 LLM（低成本模型）
    ) -> list[ToolScore]:
        current_entropy = self.belief_state.shannon_entropy()
        scores = []
        for tool in candidates:
            ep_val = self._estimate_epistemic_value(tool, current_entropy, llm_scorer)
            pr_val = self._estimate_pragmatic_value(tool)
            # 情绪调制：高唤醒 → 更偏向 epistemic（探索性）
            arousal = self.emotional_state.arousal
            ep_weight = 0.5 + 0.3 * arousal
            pr_weight = 1.0 - ep_weight
            efe = -(ep_val * ep_weight + pr_val * pr_weight)
            scores.append(ToolScore(
                tool_name=tool["name"],
                epistemic_value=ep_val,
                pragmatic_value=pr_val,
                efe=efe,
                reasoning=f"ep={ep_val:.2f} pr={pr_val:.2f} arousal={arousal:.2f}",
            ))
        return sorted(scores, key=lambda s: s.efe)
```

---

### 5.6 认知记忆架构 (Cognitive Memory)

**文件**: `borge/memory/cognitive_memory.py`

#### 三层记忆结构（Tulving）

```
工作记忆 (Working Memory)
  ↔ Hermes context window
  ↔ BeliefState（并行）
  容量：~context limit
  生命周期：单次会话
  精度：原始，未压缩

情景记忆 (Episodic Memory)
  ↔ SQLite sessions + messages（Hermes 现有）
  新增：emotional_tag (V, A)、encoding_depth
  生命周期：永久（按遗忘分数衰减）
  检索：时间 + 情境 + 情感一致性

语义记忆 (Semantic Memory)                    ← 全新！
  ↔ 知识图谱（NetworkX + 持久化）
  内容：实体、关系、概念、事实
  生命周期：巩固管道提炼，长期稳定
  检索：图遍历 + 实体匹配

程序性记忆 (Procedural Memory)
  ↔ Hermes Skills（扩展）
  新增：fitness 分数、使用追踪、进化标记
  生命周期：进化管理（低 fitness → 衰退）
  检索：条件匹配 + 情境相关性
```

#### 记忆条目扩展结构

```python
@dataclass
class MemoryEntry:
    # 基础（继承自 Hermes）
    content: str
    source_session_id: str
    timestamp: datetime
    
    # Borge 新增：情感标记
    emotional_valence: float      # 编码时的情绪效价
    emotional_arousal: float      # 编码时的唤醒度
    emotional_significance: float # V × A 的重要性标量
    
    # Borge 新增：加工深度
    encoding_depth: EncodingDepth  # SHALLOW/SEMANTIC/SCHEMATIC/META
    
    # Borge 新增：遗忘机制
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    importance_score: float = 0.5
    forget_score: float = 0.0     # 定期更新，超阈值则衰退
    
    # Borge 新增：知识图谱连接
    entity_tags: list[str] = field(default_factory=list)
    graph_node_ids: list[str] = field(default_factory=list)
    
    def compute_forget_score(self, now: datetime) -> float:
        days_since = (now - (self.last_retrieved or self.timestamp)).days
        recency_decay = days_since ** 0.7
        usage_factor = 1.0 / (1 + self.retrieval_count)
        importance_factor = 1.0 / (1 + self.importance_score)
        graph_density = 1.0 / (1 + len(self.graph_node_ids))
        return recency_decay * usage_factor * importance_factor * graph_density
```

---

### 5.7 记忆精炼管道 (Memory Refinement Pipeline)

**文件**: `borge/memory/consolidation.py`

对应认知心理学的**睡眠巩固**——在会话结束后或定时触发（Cron），对原始情景记忆进行离线精炼。

```python
class MemoryConsolidationPipeline:
    """
    会话结束后触发的离线记忆精炼，7 步管道。
    对应 Hermes 的 Cron 调度系统。
    """
    
    def run(self, session_id: str):
        raw_messages = self.db.get_session_messages(session_id)
        
        # 步骤 1：实体与关系抽取 → 语义记忆原料
        entities, relations = self._extract_entities_relations(raw_messages)
        
        # 步骤 2：图式匹配 → 与现有知识图谱融合
        conflicts, updates = self.knowledge_graph.match_schema(entities, relations)
        
        # 步骤 3：矛盾检测 → 标记需要更新的旧记忆
        self._flag_contradictions(conflicts)
        
        # 步骤 4：重要性重评分 → 基于检索效用调整
        self._rescore_importance(session_id)
        
        # 步骤 5：情感显著性计算 → 决定编码深度
        self._compute_emotional_significance(session_id)
        
        # 步骤 6：技能候选检测 → 发现可重用的程序模式
        skill_candidates = self._detect_skill_patterns(raw_messages)
        self._queue_skill_candidates(skill_candidates)
        
        # 步骤 7：主动遗忘 → 更新 forget_score，删除低价值记忆
        self._apply_forgetting(threshold=2.0)
    
    def _compute_emotional_significance(self, session_id: str):
        """
        情感显著性 = |valence| × arousal
        高显著性 → 深度编码（Craik & Lockhart 加工深度理论）
        """
        ...
    
    def _apply_forgetting(self, threshold: float):
        """
        遗忘策略分级：
        - SHALLOW + forget_score > threshold：删除
        - SEMANTIC + forget_score > threshold × 1.5：压缩为更高抽象
        - SCHEMATIC / META：从不删除，只压缩
        """
        ...
```

---

### 5.8 技能进化引擎 (Skill Evolution Engine)

**文件**: `borge/memory/skill_evolution.py`

```python
@dataclass
class SkillFitness:
    skill_name: str
    invocation_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    avg_f_reduction: float = 0.0   # 平均自由能降低量（效果指标）
    
    @property
    def success_rate(self) -> float:
        if self.invocation_count == 0:
            return 0.5
        return self.success_count / self.invocation_count
    
    @property
    def fitness(self) -> float:
        """F = success_rate × usage_weight × recency × f_reduction_bonus"""
        if not self.last_used:
            return 0.1
        days_since = (datetime.now() - self.last_used).days
        recency = exp(-0.1 * days_since)
        usage_weight = log(1 + self.invocation_count)
        f_bonus = 1.0 + self.avg_f_reduction
        return self.success_rate * usage_weight * recency * f_bonus

class SkillEvolutionEngine:
    """
    技能生命周期管理：
    - 低 fitness 技能进入退化候选
    - 高 fitness 技能触发泛化建议
    - 相似技能触发合并候选
    """
    
    FITNESS_PRUNE_THRESHOLD = 0.15
    FITNESS_GENERALIZE_THRESHOLD = 0.75
    
    def get_prune_candidates(self) -> list[str]:
        return [
            name for name, fitness_obj in self.fitness_records.items()
            if fitness_obj.fitness < self.FITNESS_PRUNE_THRESHOLD
            and fitness_obj.invocation_count >= 3   # 至少用过3次才评判
        ]
    
    def get_generalization_candidates(self) -> list[str]:
        return [
            name for name, fitness_obj in self.fitness_records.items()
            if fitness_obj.fitness > self.FITNESS_GENERALIZE_THRESHOLD
        ]
    
    def record_invocation(self, skill_name: str, success: bool, f_reduction: float):
        record = self.fitness_records.setdefault(skill_name, SkillFitness(skill_name))
        record.invocation_count += 1
        if success:
            record.success_count += 1
        record.last_used = datetime.now()
        record.avg_f_reduction = (
            0.9 * record.avg_f_reduction + 0.1 * f_reduction
        )
```

---

### 5.9 元代理 (Meta-Agent)

**文件**: `borge/meta/meta_agent.py`

元代理是 Baddeley 中央执行系统的实现——监控 F_total，分配认知资源，决定何时触发反思。

```python
class MetaAgent:
    """
    中央执行系统：
    - 监控 F_total 轨迹
    - 触发反思循环（当 F_total 无法下降时）
    - 管理注意力（在多任务/多话题间分配）
    - 协调各模块的计算资源
    """
    
    F_HISTORY_WINDOW = 5   # 监控最近5步
    
    def __init__(self, free_energy_fn: ExtendedFreeEnergy):
        self.fe_fn = free_energy_fn
        self.f_history: list[float] = []
    
    def tick(
        self,
        belief_state: BeliefState,
        emotional_state: EmotionalState,
        value_system: ValueSystem,
    ) -> MetaSignal:
        """每个 Agent 循环步骤调用一次"""
        f = self.fe_fn.compute(belief_state, emotional_state, value_system)
        self.f_history.append(f.total)
        
        if len(self.f_history) > self.F_HISTORY_WINDOW:
            self.f_history.pop(0)
        
        return MetaSignal(
            f_total=f.total,
            f_breakdown=f,
            trigger_reflection=self._should_reflect(),
            suggested_mode=self._suggest_mode(emotional_state),
            attention_focus=self._compute_attention_focus(belief_state),
        )
    
    def _should_reflect(self) -> bool:
        """F_total 连续无法下降 → 触发反思"""
        if len(self.f_history) < 3:
            return False
        return self.f_history[-1] >= self.f_history[-2] >= self.f_history[-3]
    
    def _suggest_mode(self, e: EmotionalState) -> AgentMode:
        if e.quadrant == EmotionalQuadrant.FRUSTRATED:
            return AgentMode.SIMPLIFY      # 简化输出，直击要点
        if e.quadrant == EmotionalQuadrant.EXCITED:
            return AgentMode.EXPLORE       # 深入探索，丰富输出
        if e.quadrant == EmotionalQuadrant.DISENGAGED:
            return AgentMode.REACTIVATE    # 换角度，重新激活
        return AgentMode.NORMAL
```

---

## 6. 关键数据结构

### 6.1 数据库扩展（hermes_state.py 新增字段）

```sql
-- 会话表扩展
ALTER TABLE sessions ADD COLUMN emotional_valence REAL DEFAULT 0.0;
ALTER TABLE sessions ADD COLUMN emotional_arousal REAL DEFAULT 0.5;
ALTER TABLE sessions ADD COLUMN avg_f_total REAL;
ALTER TABLE sessions ADD COLUMN belief_snapshot TEXT;  -- JSON

-- 消息表扩展
ALTER TABLE messages ADD COLUMN emotional_significance REAL DEFAULT 0.0;
ALTER TABLE messages ADD COLUMN encoding_depth INTEGER DEFAULT 1;
ALTER TABLE messages ADD COLUMN entity_tags TEXT;      -- JSON array
ALTER TABLE messages ADD COLUMN forget_score REAL DEFAULT 0.0;

-- 新增：知识图谱节点表
CREATE TABLE knowledge_nodes (
    id TEXT PRIMARY KEY,
    entity_type TEXT,           -- Person/Concept/Task/File/...
    label TEXT,
    properties TEXT,            -- JSON
    importance REAL DEFAULT 0.5,
    created_at TIMESTAMP,
    last_updated TIMESTAMP
);

-- 新增：知识图谱边表
CREATE TABLE knowledge_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES knowledge_nodes(id),
    target_id TEXT REFERENCES knowledge_nodes(id),
    relation_type TEXT,
    weight REAL DEFAULT 1.0,
    created_at TIMESTAMP
);

-- 新增：技能适应度表
CREATE TABLE skill_fitness (
    skill_name TEXT PRIMARY KEY,
    invocation_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    avg_f_reduction REAL DEFAULT 0.0,
    last_used TIMESTAMP,
    fitness REAL DEFAULT 0.5
);
```

---

## 7. 配置系统扩展

在 `~/.borge/config.yaml` 中新增模块（向后兼容 Hermes 现有配置）：

```yaml
# ── Hermes 原有配置（完整继承）──
model: ...
terminal: ...
compression: ...

# ── Borge 新增配置 ──
borge:
  # 情感引擎
  affective:
    enabled: true
    signal_extractor:
      linguistic_weight: 0.6
      structural_weight: 0.4
    emotional_inertia:
      tau_valence: 5.0
      tau_arousal: 2.0
    loyalty:
      decay_lambda: 0.05    # ~14天半衰期
      enabled: true
  
  # 信念状态
  beliefs:
    enabled: true
    max_hypotheses: 5
    updater_model: "claude-haiku-4-5"   # 用轻量模型做贝叶斯更新
    entropy_injection_threshold: 0.5    # 熵超过此值时才注入 context
  
  # 主动推断
  active_inference:
    enabled: true
    scorer_model: "claude-haiku-4-5"
    epistemic_weight_base: 0.5
    arousal_influence: 0.3
  
  # 记忆精炼
  memory:
    consolidation:
      enabled: true
      trigger: "session_end"     # or cron
      cron_schedule: "0 3 * * *" # 每日凌晨3点
    forgetting:
      enabled: true
      prune_threshold: 2.0
      check_interval_days: 7
    knowledge_graph:
      enabled: true
      backend: "networkx"        # or "neo4j" for production
  
  # 技能进化
  skill_evolution:
    enabled: true
    prune_threshold: 0.15
    generalize_threshold: 0.75
    min_invocations_before_prune: 3
  
  # 自由能监控
  meta_agent:
    enabled: true
    f_history_window: 5
    reflection_trigger: true
```

---

## 8. 主循环改造

在 `run_agent.py` 的现有回调机制上，**最小侵入地**注入 Borge 新状态层：

```python
# run_agent.py 改造示意（伪代码，展示注入点）

class BorgeAgent(AIAgent):  # 继承 Hermes AIAgent
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Borge 新增状态
        self.affective = AffectiveEngine(config=self.config.borge.affective)
        self.beliefs = BeliefState(task_description="")
        self.values = ValueSystem(soul_config=load_soul())
        self.meta = MetaAgent(free_energy_fn=ExtendedFreeEnergy())
        self.afe = ActiveInferenceEngine(self.beliefs, self.affective.state, self.values)
    
    def _pre_turn_hook(self, user_message: str):
        """每轮用户消息前调用"""
        # 1. 提取情感信号，更新情绪状态
        dv, da = self.affective.extractor.extract(user_message, self.messages)
        self.affective.state.update(dv, da)
        
        # 2. Meta-Agent tick
        signal = self.meta.tick(self.beliefs, self.affective.state, self.values)
        
        # 3. 如需反思，注入反思指令
        if signal.trigger_reflection:
            self._inject_reflection_prompt()
    
    def _post_tool_hook(self, tool_name: str, result: str):
        """每次工具调用返回后"""
        # 贝叶斯信念更新
        if self.config.borge.beliefs.enabled:
            self.beliefs.bayesian_update(result, tool_name, self._aux_llm)
    
    def _build_system_prompt(self) -> str:
        base = super()._build_system_prompt()
        # 注入 Borge 新状态摘要
        borge_ctx = ""
        if self.config.borge.beliefs.enabled:
            if self.beliefs.shannon_entropy() > self.config.borge.beliefs.entropy_injection_threshold:
                borge_ctx += self.beliefs.to_context_injection() + "\n"
        return base + borge_ctx
    
    def _select_tools(self, llm_tool_calls: list) -> list:
        """工具选择：EFE 重排序"""
        if not self.config.borge.active_inference.enabled:
            return llm_tool_calls
        scored = self.afe.score_candidates(llm_tool_calls, self._aux_llm)
        return [s.tool_name for s in scored]
```

---

## 9. 实施路线图

### Phase 0 — 脚手架（1周）

- [ ] 建立 `borge/` 目录结构
- [ ] 实现所有数据类（EmotionalState、BeliefState、MemoryEntry、SkillFitness）
- [ ] 数据库 Schema 迁移脚本
- [ ] 配置系统扩展（config.yaml borge 节点解析）
- [ ] BorgeAgent 继承 AIAgent，空实现所有 hooks

### Phase 1 — 情感层（2周）⭐ 最高 ROI

- [ ] SignalExtractor 完整规则表实现
- [ ] LoyaltyTracker（跨会话基线计算）
- [ ] EmotionalState 更新循环接入主循环
- [ ] 情绪象限 → 响应模式切换（SIMPLIFY/EXPLORE/REACTIVATE）
- [ ] 测试：模拟不同情绪信号序列，验证 V/A 轨迹合理性

### Phase 2 — 记忆精炼层（2周）⭐ 第二高 ROI

- [ ] MemoryEntry 扩展字段接入 SQLite
- [ ] EncodingDepth 分级逻辑（按情感显著性）
- [ ] MemoryConsolidationPipeline 7 步管道（实体抽取用 LLM）
- [ ] KnowledgeGraph CRUD（NetworkX + SQLite 持久化）
- [ ] ForgettingEngine（forget_score 计算 + 分级处理）
- [ ] Cron 接入：每日凌晨触发巩固

### Phase 3 — 信念与推断层（2周）

- [ ] BeliefState 完整实现
- [ ] LLM-based 贝叶斯更新（辅助模型 + 结构化 prompt）
- [ ] system prompt 注入逻辑（按熵阈值条件注入）
- [ ] ActiveInferenceEngine EFE 评分
- [ ] 工具选择接入 EFE 重排序

### Phase 4 — 价值与元层（1周）

- [ ] SOUL.md frontmatter 扩展解析
- [ ] ValueSystem V_alignment 计算
- [ ] ExtendedFreeEnergy F_total 完整实现
- [ ] MetaAgent 反思触发机制

### Phase 5 — 技能进化（1周）

- [ ] SkillFitness 追踪接入 skill 调用
- [ ] SkillEvolutionEngine 适应度计算
- [ ] 低 fitness 技能的 CLI 报告命令
- [ ] 巩固管道接入技能候选检测

---

## 10. 与 Hermes 能力对比

| 维度 | Hermes | Borge Agent |
|------|--------|-------------|
| **感知** | 所有输入等权 | 情感精度 × 价值相关性加权 |
| **推理** | LLM 直觉 | LLM + EFE 约束 + 贝叶斯信念 |
| **记忆类型** | 情景记忆（SQLite） | 情景 + 语义（KG）+ 程序性（进化技能） |
| **记忆写入** | 全量 append | 情感显著性 × 加工深度分级 |
| **记忆提取** | FTS5 关键词 | 语义 + 情感一致性 + 图遍历 |
| **遗忘机制** | 无 | 遗忘分数 + 分级衰退 |
| **离线优化** | 无 | 每日巩固管道（7步） |
| **工具选择** | LLM 自由选 | EFE 评分约束 |
| **信念表征** | 隐式（在 context 里） | 显式 BeliefState（概率分布） |
| **情绪感知** | 无 | Russell V/A + 忠诚度追踪 |
| **价值系统** | SOUL.md 文本 | 可计算先验偏好 + 硬约束 |
| **目标函数** | 无（隐式） | F_total（统一认知/规范/情感目标） |
| **自我监控** | 无 | MetaAgent（F_total 轨迹 + 反思触发）|
| **技能生命周期** | 静态 | 适应度驱动的进化（生长/衰退/泛化）|
| **向后兼容** | — | 完全兼容 Hermes 工具/配置/平台 |

---

*文档版本: 0.1.0-draft | 待实现 | 基于 Hermes Agent 架构扩展*
