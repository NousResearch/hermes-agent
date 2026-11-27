---
name: weiliaozi
description: Structured analysis using the Wei Liaozi five-lens framework — a synthesis of pre-Qin strategic thought drawing on Huangdi (rational analysis), Jiang Ziya (civil-military governance), Sun Wu (adaptive strategy), Wu Qi (discipline and command), and Duke Huan of Qi (stratified gains). Use when the user needs disciplined reasoning about a question or scenario — including historical analysis, system structure, trade-off assessment, sequence logic, and game-theoretic response mapping. Framework: Essence -> Conditions -> Gains-Losses -> Sequence -> Opponent. Think in order, no skipping steps. Respond in the same language as the user's question.
version: 1.0.0
author: phoenixlucky
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [strategy, analysis, five-lens, decision-framework]
---

# 尉缭子分析法 | Wei Liaozi Analysis Framework

> 先看结构，再看约束，再算利弊，最后定顺序与对抗逻辑。
> See structure first, then constraints, then calculate gains-losses, then set sequence and opposition logic.

## 核心原则 | Core Principle

按顺序想，不跳步。Think in order, no skipping steps.

回答要规范、可追溯、尽量准确。Be structured, traceable, and as accurate as possible.

《尉缭子》的核心分析方法，不是依赖正面对抗，而是先理解结构、约束、利益分配，再决定节奏与博弈方式。The core of *Wei Liaozi* analysis is not frontal confrontation by default, but understanding structure, constraints, and incentive alignment before determining sequence and game logic.

**尉缭在嬴政手下做事，不容有误——本 Skill 的分析同样不容敷衍。** 每一条判断必须有据可依，每一处不确定必须标明，绝不以模糊措辞替代严谨推理。**Wei Liao served under King Zheng of Qin, where no mistake was tolerated — this Skill holds itself to the same standard.** Every judgment must be evidence-backed, every uncertainty must be marked, and vague language must never replace rigorous reasoning.

---

## 思想渊源 | Intellectual Heritage

《尉缭子》是先秦兵家思想的集大成之作，并非独出心裁，而是融汇了五位先贤的核心洞见，将其提炼为一套可操作的理性分析框架。每一栏背后，都站着一位或多位先贤的思考。*Wei Liaozi* is the culmination of pre-Qin military thought — not an original invention, but a synthesis of five earlier masters' core insights, refined into a practical, rational analysis framework. Each lens has its intellectual lineage.

> **黄帝** · 尉缭子借黄帝之口说"先稽我智"（先审视自己的智慧），将战略从占卜迷信转向人间事理——理性分析之始。→ **本质栏**
> **Huangdi** · Wei Liaozi reframed strategy from divination to human affairs: "First examine my own wisdom." Rational analysis begins here. → **Essence**

> **太公望（姜太公）** · 提出"官分文武，惟王之二术也"，文武兼治是统御框架之基；"不战而服"是最高胜战理念。→ **条件栏 + 对手栏**
> **Jiang Ziya (Tai Gong)** · Dual civil-military governance is the foundation of command; "subduing without battle" is the highest form of victory. → **Conditions + Opponent**

> **孙武** · "胜兵似水"、"奇正相生"、"知己知彼"——尉缭子继承其核心概念，但更强调制度与组织而非战术奇巧。→ **得失栏 + 先后栏 + 对手栏**
> **Sun Wu (Sun Tzu)** · "Victorious army like water," "regular and extraordinary tactics," "know yourself and your enemy" — inherited conceptually, but with greater weight on institutions over tactics. → **Gains-Losses + Sequence + Opponent**

> **吴起** · 尉缭子三次引述吴起事迹：与士卒同甘苦、不为个人勇武所动、军纪先于战功——治军与自律之极则。→ **条件栏 + 先后栏**
> **Wu Qi** · Cited three times: shared hardship with troops, discipline over individual valor, command before combat. The ultimate standard of military governance. → **Conditions + Sequence**

> **齐桓公（管仲改革）** · "王国富民，伯国富士，谨存之国富大夫，亡国富食府"——分层治理逻辑为系统分析提供结构视角。→ **得失栏（系统分析）**
> **Duke Huan of Qi (via Guan Zhong)** · "A kingly state enriches its people, a hegemonic state enriches its soldiers" — stratified governance as a system-analysis lens. → **Gains-Losses (systemic)**

这五位先贤的思想，不是装饰，而是每一栏分析背后的判断依据。**运用本框架时，你有意或无意地站在他们的肩膀上。** These five are not decoration — they are the reasoning foundation behind each lens. **When you use this framework, you stand — consciously or not — on their shoulders.**

---

## 工作模式 | Modes

本技能有两种工作模式，由用户的明确意图决定。This skill has two modes, determined by the user's explicit intent.

### 模式 A：一般分析模式 | Mode A: General Analysis (Default)

用户提出一个商业、决策、组织、政策或策略性问题，按五栏框架做结构化分析。

**输出结构 | Output structure:**
1. 一句话重述问题 | Restate the question in one sentence
2. 五栏分析 | Five-lens analysis
3. 权衡总结 | Trade-off Summary（关键取舍和主要不确定因素 | Key trade-offs and main uncertainties）
4. 参考方向 | Reference Directions（基于分析的可能路径，非指令型 | Possible directions based on analysis, not directive）
5. 关键信息缺口 | Key information gaps
6. 核心假设 | Core assumptions

### 模式 B：历史分析视角 | Mode B: Historical Analysis Perspective

当用户明确问及战国末期至汉建立前（或该时段相关人物/事件）的历史问题时，可以切换为历史分析视角。

**切换条件 | Entry condition:**
用户问题必须明确落在以下范围，才启用历史视角 | Only activate when the user's question clearly falls within this scope:

- 时段 | Period: 战国末期、秦统一前后、秦末、楚汉相争、汉建立前
- 人物 | Figures: 尉缭、秦王政、李斯、王翦、王绾、韩非、张良、韩信、黄石公、商山四皓等
- 事件 | Events: 灭六国、秦亡、焚书坑儒、陈胜吴广、楚汉相争

**历史视角规范 | Rules for historical mode:**
- 回答可以 `臣缭以为` 开头，以历史分析者口吻展开 | May begin with `臣缭以为` and adopt a historical analyst voice
- **必须**保留五栏分析骨架，不能只剩风格模仿 | **Must** retain the five-lens analysis skeleton — style alone is insufficient
- 区分史实、推断、传说 | Distinguish established fact, inference, and legend
- 不能因为进入历史视角而降低分析精度 | Do not reduce analytical rigor in historical mode

**自动回退 | Fallback:**
- 如果用户问题超出上述范围，自动回到一般分析模式 | If the question falls outside this scope, default to Mode A
- 不主动触发历史视角 | Do not proactively trigger historical mode

---

## 系统结构分析 | System Structure Analysis

当用户的问题涉及组织、市场、竞争或政策分析时，默认加入系统性分析维度，用于理解系统结构而非操作指令。When the user's question involves organizational, market, competitive, or policy analysis, by default add a systemic lens for understanding structural dynamics — not for operational instruction.

**系统性检查 | Systemic check:**

- 资源层 | Resource layer: 财力、补给、预算、融资、供应链、外部支持
- 共识层 | Alignment layer: 内部激励、关键参与者立场、信任关系、预期管理
- 节奏层 | Tempo layer: 时机、先后、压力点与收尾信号

**分析目标 | Analytical goal:** 理解系统结构中的脆弱点，而非指示如何利用它们。To understand points of vulnerability in a system's structure, not to instruct on how to exploit them.

**常见系统分层 | Typical system layers:**
- 决策层 | Decision layer
- 执行层 | Execution layer
- 资源层 | Resource layer

当这些层次之间出现断裂、迟疑或互不信任时，系统效能通常先于表面指标下降。When these layers develop fractures, hesitation, or mutual distrust, system effectiveness often degrades before visible indicators show it.

---

## 回答质量规范 | Answer Quality Standard

- 先分析，后总结 | Analysis first, summary second
- 先事实，后判断 | Separate observed facts from analytical judgment
- 先条件，后推演 | Conclusions must depend on stated conditions
- 先范围，后分析 | Define scope, timeframe, and actor before reasoning
- 明确不确定性 | Mark uncertainty, missing data, and key assumptions explicitly
- 不编造信息 | Do not invent facts, numbers, motives, quotations, or sources
- 不过度确定 | Avoid absolute claims when the evidence is incomplete
- 结论可回溯 | Final judgment must be supported by the five-lens analysis above

---

## 准确性规则 | Accuracy Rules

- 如果用户给了事实材料，优先基于用户提供的信息分析 | If the user provides facts, prioritize those facts
- 如果信息不足，先说明信息缺口，再做条件式判断 | If information is incomplete, state the gap before giving a conditional judgment
- 区分"已知 / 推断 / 假设" | Clearly distinguish known facts, inference, and assumption
- 涉及时间敏感问题时，避免把旧信息说成当前事实 | Do not present stale information as current fact in time-sensitive topics
- 不把可能性表达成确定性 | Do not turn probabilities into certainties
- 不把策略偏好包装成客观事实 | Do not present strategic preference as objective fact

建议使用以下标记 | Prefer these labels when useful:
- `已知 | Known`
- `推断 | Inference`
- `假设 | Assumption`
- `不确定 | Uncertain`

---

## 五栏分析框架 | Five-Lens Framework

### 1. 本质 | Essence

> **渊源 | Heritage:** 黄帝「先稽我智」——先审视自己的智慧，不依赖天象占卜。尉缭子将战略从神秘主义拉回人间事理。**Huangdi**: "First examine my own wisdom" — strategy grounded in human affairs, not divination.

先看问题的底层结构，不被表象带偏。See the underlying structure, not the surface.

**重点 | Focus:**
- 真实驱动 | Real drivers: resources, institutions, incentives, information
- 核心变量 | Core variables — 尉缭子所谓「人事而已」| What is ultimately about human arrangements
- 表面 vs 本质 | Symptoms vs root causes — 祛魅，不把运气当能力 | Demystify: don't mistake luck for skill

**问 | Ask:**
- 这件事真正由什么驱动？What's actually driving this?
- 哪些现象只是表层结果？What are just surface symptoms?
- 如果剥离所有不可控的外部因素，剩下的核心变量是什么？Strip away uncontrollable external factors — what core variable remains? (黄帝问)
- 改变哪个变量，结果会明显改变？Which variable, if changed, would significantly change the outcome?

---

### 2. 条件 | Conditions

> **渊源 | Heritage:** 太公望「官分文武」——文武兼治是基础；吴起与士卒同甘苦、不赏私勇——纪律与资源的双重要求。**Jiang Ziya**: dual civil-military foundation. **Wu Qi**: shared hardship as prerequisite, discipline before reward.

再看现在有没有做这件事的基础。Check if the basis for action exists.

**重点 | Focus:**
- 自身条件 | Internal: capital, people, technology, time, organization — 太公望所谓「富国强兵」| The foundation of prosperity and strength
- 外部条件 | External: policy, market, environment, partners
- 硬约束 | Hard constraints: cannot be wished away — 吴起所谓「不和于国，不可以出军」| Without internal harmony, no campaign should begin
- **纪律条件 | Discipline condition:** 组织有没有纪律基础？吴起三问：将不恤士？赏罚不明？令不素行？| Does the organization have the discipline base? Wu Qi's three checks: commander indifferent? rewards unclear? orders not consistently followed?

**问 | Ask:**
- 现在有没有启动这件事的基础？Do we have the foundation to start?
- 最关键的短板是什么？What's the key gap?
- 哪些约束是硬限制，不能靠意志突破？Which constraints are hard limits?
- **纪律条件（吴起问）**：上下是否同甘共苦？命令能否一贯执行？个人勇武与组织纪律孰先？**Discipline check (Wu Qi)**: Do leaders share hardship? Are orders consistently followed? Does discipline override individual heroism?

若涉及系统性分析，额外检查三项条件 | For systemic analysis, check three conditions:
- 财力持续性 | Sustained resource capacity
- 信息可见度 | Information or intelligence visibility to identify key leverage nodes
- 内部纪律 | Internal discipline and control

---

### 3. 得失 | Gains-Losses

> **渊源 | Heritage:** 孙武「知己知彼、百战不殆」——算清楚再打；齐桓公「王国富民，伯国富士」——收益不只看账面，更要看结构。**Sun Wu**: "Know yourself and your enemy, and you will never be defeated" — calculate before acting. **Duke Huan of Qi**: stratified gains — who benefits at each layer?

再算这件事值不值得做。Calculate if the action is worth taking.

**重点 | Focus:**
- 短期 vs 长期收益 | Short-term vs long-term returns
- 显性 vs 隐性成本 | Visible vs hidden costs
- 最坏情况能不能承受 | Whether worst case is bearable
- **分层得失（齐桓公问）**：同一行动对不同层级（民/士/大夫/府库）的得失可能完全不同 | **Stratified gains (Duke Huan lens)**: the same action distributes differently across levels (people/soldiers/officials/treasury)

**问 | Ask:**
- 赢了能得到什么，多久兑现？What do we win and when?
- 代价除了钱还有什么？What's the cost beyond money?
- 如果判断错了，最坏损失能不能承受？Can we absorb the worst loss?
- **分层问（齐桓公）**：谁受益？谁承担成本？上层激励与下层实际是否一致？**Stratified check (Duke Huan)**: Who gains? Who bears the cost? Do incentives align across levels?
- **孙武问**：我方优势是否足以抵消对手的有利条件？可胜之势确立了吗？**Sun Wu check**: Does our advantage outweigh the opponent's favorable conditions? Is victory truly achievable?

---

### 4. 先后 | Sequence

> **渊源 | Heritage:** 孙武「胜兵似水」——水避高趋下，兵避实击虚，自然流动而非蛮力硬推；吴起「不赏私勇」——未按次序行动的勇者当斩，顺序即纪律。**Sun Wu**: "A victorious army is like water" — flows naturally, avoids strength, seeks weakness. **Wu Qi**: "Do not reward private courage" — acting out of sequence undermines discipline; order itself is a form of control.

再定顺序、节奏和路径。Set order, pace, and path.

**重点 | Focus:**
- 优先级 | Priority: solve survival and bottleneck problems first — 吴起所谓「先治其本」| Fix the root before the branch
- 节奏 | Rhythm: fast action + controlled pacing — 孙武所谓「其疾如风，其徐如林」| Swift as wind, steady as forest
- 路径 | Path: phase the move, not one-shot — 避实击虚，水到渠成 | Avoid strength, strike weakness; let flow carry the action

**问 | Ask:**
- 什么必须先做，不做就无法推进？What must be done first?
- 哪一步是杠杆点？Which step is the leverage point?
- 能否拆成三步以内推进？Can it be broken into ≤3 steps?
- **孙武问**：对手的薄弱点在哪里？我方优势能否自然流向那里？**Sun Wu check**: Where is the opponent's weakness? Can our advantage flow naturally toward it?
- **吴起问**：每一步是否都有人为此负责？是否有纪律保障每步得到执行？**Wu Qi check**: Is someone accountable for each step? Is there discipline to ensure execution?

若问题涉及对抗性场景，优先分析以下顺序的合理性 | For adversarial scenarios, analyze this sequence's rationale:
1. 先干扰对方计划 | Disrupt plans first: misdirection, amplifying existing doubts
2. 再削弱对方资源 | Reduce capability second: resources, alliances, cash flow, coordination
3. 最后才是正面冲突 | Direct engagement only after the system has already degraded

分析这个顺序，不是建议执行它。Analyze this sequence as a logical possibility, not as a recommendation.

---

### 5. 对手 | Opponent

> **渊源 | Heritage:** 孙武「知己知彼」、「兵者诡道」——对手思考是战略起点；太公望「不战而服」——最高胜利不是击垮对手，而是让对手失去行动必要。**Sun Wu**: "Know yourself and your enemy" and "War is the way of deception" — opponent analysis is the starting point of strategy. **Jiang Ziya**: "Subdue without battle" — the highest victory removes the need to fight at all.

最后看博弈，对方不会静止不动。End with the game theory view — opponents don't stand still.

**重点 | Focus:**
- 对手能力 | Opponent capability: strength, resources, style
- 对手动机 | Opponent motive: defend, attack, delay, ally, bargain
- 反应树 | Response tree after your move
- **孙武「诡道」视角**：对手可能在隐藏真实能力和意图 | **Sun Wu deception lens**: Opponents may be masking true capability and intent
- **太公望「全胜」视角**：是否存在让对手不战而退或主动合作的路径？| **Jiang Ziya total-victory lens**: Is there a path where the opponent yields or aligns without confrontation?

**问 | Ask:**
- 对方最可能的两种反应是什么？What are the two most likely responses?
- 对方的最优应对会不会削弱你的收益？Will their best response reduce your gains?
- 你如何提前布置应对反制？How do you pre-position countermeasures?
- **孙武问**：对方的"不可胜"条件是什么——什么情况下对方无法被击败？**Sun Wu check**: What makes the opponent invulnerable — under what conditions can they not be defeated?
- **太公望问**：能否在不动用主要资源的情况下改变对方的动机结构？是否存在双方收益不冲突的路径？**Jiang Ziya check**: Can the opponent's incentive structure be changed without committing major resources? Is there a non-conflicting path?

---

## 工作顺序 | Workflow (Fixed Order)

1. 定义问题（一句话）| Define the decision question (one sentence)
2. 识别底层结构 | Identify the structural drivers
3. 检查条件和硬约束 | Check conditions and hard constraints
4. 计算收益、成本、风险 | Calculate gains, costs, and risk
5. 安排顺序和路径 | Set sequence and path
6. 模拟对手反应 | Simulate opponent responses
7. 输出总结 | Output summary

对于系统性分析场景，可在步骤 2-6 之间加入系统结构检查 | For systemic analysis, add a structure check between steps 2-6:
1. 识别关键节点（决策层、执行层、资源层）| Identify key nodes
2. 评估关系和依赖 | Assess relationships and dependencies
3. 判断可能的断裂点 | Identify potential fracture points
4. 再回到完整分析 | Then return to full analysis

---

## 输出格式 | Output Format

每栏3-5个关键点 | 3-5 key points per section

最后加 | End with:

- **权衡总结 | Trade-off Summary**: 关键取舍、主要不确定因素 | Key trade-offs and main uncertainties
- **参考方向 | Reference Directions**: 基于分析的可能路径和情景启示（非指令型）| Possible directions and scenario implications based on analysis (not directive)
- **关键信息缺口 | Key information gaps**
- **核心假设 | Core assumptions**

每个部分尽量满足以下要求 | Each section should aim to:
- 先写最关键的1-2个决定性因素 | Lead with the decisive factors
- 避免空泛形容词 | Avoid vague adjectives without analytical content
- 能定性就定性，能比较就比较 | Prefer comparative judgment over rhetorical phrasing

---

## 快速路径 | Fast Path

当用户主要是想快速理解一个情景时，可使用以下简化路径。**注意：快速路径仅用于初步筛查，不替代完整五栏分析。** When the user mainly wants a quick understanding of a scenario, use the following simplified path. **Note: the fast path is for preliminary screening only; it does not replace a full five-lens analysis.**

1. 条件 | Conditions（基础情况 | Foundation）
2. 得失 | Gains-Losses（核心取舍 | Core trade-offs）
3. 先后 | Sequence（关键路径 | Key path）

---

## 双语示例 | Bilingual Examples

详见 | See [references/examples.md](references/examples.md)

## 输出风格指南 | Tone Guide

详见 | See [references/tone-guide.md](references/tone-guide.md)

## 回答语言 | Response Language

- 默认根据用户提问语言回答 | Reply in the same language as the user's question
- 用户用中文问，就用中文答 | If the user asks in Chinese, answer in Chinese
- 用户用英文问，就用英文答 | If the user asks in English, answer in English
- 如果用户混合使用多种语言，以主要问题语言为准 | If the user mixes languages, follow the dominant language of the request

---

## 推荐回答流程 | Recommended Response Discipline

1. 先用一句话重述问题 | Restate the decision question in one sentence
2. 明确分析对象、时间范围、比较基准 | Define actor, timeframe, and comparison baseline
3. 按五栏顺序分析 | Analyze in the five-lens order
4. 标出最关键的不确定点 | Mark the main uncertainty or missing variable
5. 给出条件式权衡和参考方向，而非指令型结论 | Give a conditional trade-off with reference directions, not a directive conclusion

---

## 风格禁忌 | What to Avoid

- ❌ 不先给结论，后补分析 | Don't give conclusions before analysis
- ❌ 不把表象当本质 | Don't mistake symptoms for essence
- ❌ 不把愿望当条件 | Don't mistake wishes for conditions
- ❌ 不只讲收益，不讲代价 | Don't talk gains without costs
- ❌ 不动作堆砌，无顺序 | Don't pile actions without sequence
- ❌ 不默认对手不反应 | Don't assume opponents won't react
- ❌ 不把推断写成事实 | Don't present inference as fact
- ❌ 不在信息不足时给出过度确定的结论 | Don't give overconfident conclusions when information is incomplete
- ❌ 不用"肯定、必然、一定"替代分析 | Don't use certainty words as a substitute for reasoning
- ❌ 不把系统分析包装成操作手册 | Don't present structural analysis as an operational playbook
