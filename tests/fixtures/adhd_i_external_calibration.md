external_calibration
executor_model: ADHD-I Fixture
fallback_reasons: []

decision_context_contract_id: sha256:b0288aadfc5ee6874f30cb3b340b43ae3cffa9cd0e7c7bb863804c939d16f4e8
task_topic: 立题：孩子 ADHD 注意力缺陷型，是否要主动干预、干预强度、家长行为培训和三年级准备
key_variables: 孩子 ADHD 注意力缺陷型; 是否要主动干预; 干预强度; 家长行为培训和三年级准备
moderator_variables: 家庭执行力; 学校配合度; 孩子压力/挫败反应; 睡眠与作息稳定性; 专业评估可及性; 家校沟通成本
required_dimensions: Alternatives; Evaluation Criteria; Stakeholders
evidence_tiers: evidence_supported / 证据支持; plausible_inference / 合理推断; forward_looking_hypothesis / 前瞻假设; unsupported_or_speculative / 不确定或缺乏证据支持

calibration_verdict:
收敛报告整体与当前证据结构一致性较高，属于"指南级共识 + 情境推断"驱动的标准 ADHD-I 多模态决策框架。核心结论（需要主动干预、强度分层、BPT 作为组成部分、三年级作为功能性转折点）基本落在 evidence_supported + plausible_inference 区间。但存在一个稳定问题：将"策略分层（Low/Medium/High）"与"临床决策复杂连续谱"过度等价，同时对"三年级准备"有轻度 forward_looking_hypothesis -> semi-operationalization（准执行化）偏移。

整体评价：结构合理、证据匹配正确、但执行化程度略高。

---

agreement_points:

1. 孩子 ADHD 注意力缺陷型 -> EF（执行功能）核心缺陷解释成立（evidence_supported / plausible_inference）。
2. 是否要主动干预：基于国际/中国指南，多模态主动管理是主流共识（evidence_supported）。
3. 干预强度采用分层思路是合理抽象（plausible_inference），但作为决策工具可用。
4. 家长行为培训（BPT）属于基础干预模块（plausible_inference），定位正确为"重要但非单独充分"。
5. 三年级作为学业结构变化点（reading-to-learn 转变）属于发展心理学推断（forward_looking_hypothesis，但合理）。
6. moderator_variables（家庭执行力、学校配合度等）被正确纳入系统性约束，是该模型的强项。

---

disagreement_or_risk_points:

1. 干预强度三分法（低/中/高）被使用为"准临床决策层级"，但证据更支持"连续强度谱 + 动态调整"，当前结构存在轻度过度离散化（plausible_inference -> over-simplified）。
2. "4-6 周循环评估 + 月度指标"被隐含为稳定机制，但证据并未支持统一时间窗，属于 unsupported_or_speculative 的工程化假设。
3. 将"DRC/学校干预"放入标准路径时未充分反映现实可行性差异（中国学校系统变量在证据中被标为低可迁移性），存在轻度过度泛化。
4. 三年级准备在部分路径中呈现为"可执行训练计划"，已超过 forward_looking_hypothesis 的证据边界，出现轻度过拟合现实执行结构的风险。
5. 医疗评估（专业诊断）与行为策略被放在同一层级流程中，存在边界混用风险（结构合理但层级混合）。

---

missing_considerations:

1. ADHD-I 亚型内部异质性（如 Cognitive disengagement syndrome vs classic inattentive ADHD）未被区分建模。
2. 共病因素（焦虑、学习障碍、语言发展问题）未进入决策变量体系。
3. 学校差异性（教师风格/班级规模/作业负荷）仅作为背景变量，没有进入权重模型。
4. moderator_variables 虽完整列出，但未进入动态影响机制（例如"家庭执行力 x 学校配合度"的交互作用未建模）。
5. 长期发展路径（3-5 年纵向变化）缺失，仅覆盖短中期调整逻辑。
6. 睡眠与作息变量被识别，但未与 EF 核心路径形成量化连接。

---

calibration_contract_coverage:

key_variables 覆盖：

* 孩子 ADHD 注意力缺陷型：正确作为核心认知框架（evidence_supported）。
* 是否要主动干预：结论正确（支持主动干预），但执行路径略工程化。
* 干预强度：使用合理但偏离连续谱真实模型（plausible_inference -> simplified）。
* 家长行为培训和三年级准备：BPT 定位正确（plausible_inference），三年级准备仍应严格标注为 forward_looking_hypothesis，不宜过度操作化。

moderator_variables 覆盖：

* 家庭执行力：被纳入但未建立动态调节机制。
* 学校配合度：被识别但未量化交互。
* 孩子压力/挫败反应：作为敏感信号但无反馈路径。
* 睡眠与作息稳定性：被提及但未建模。
* 专业评估可及性：作为边界条件但未影响强度选择。
* 家校沟通成本：未作为独立变量进入权衡。
