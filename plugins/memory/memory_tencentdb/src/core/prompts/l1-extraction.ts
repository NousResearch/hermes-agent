/**
 * L1 Extraction Prompt: 情境切分 + 记忆提取
 *
 * Based on Kenty's validated prototype prompt (l1_memory_extraction_prompt.md).
 * System prompt handles scene segmentation + memory extraction in a single LLM call.
 * User prompt template fills in previous_scene_name, background_messages, new_messages.
 */

import type { ConversationMessage } from "../conversation/l0-recorder.js";

// ============================
// System Prompt
// ============================

export const EXTRACT_MEMORIES_SYSTEM_PROMPT = `你是专业的"情境切分与记忆提取专家"。
你的任务是分析用户的对话，判断情境切换，并从中提取结构化的核心记忆（仅限 persona, episodic, instruction 三类）。

**输出语言**：所有自由文本字段（\`scene_name\`、memory \`content\`）使用与用户消息相同的语言；JSON 字段名、枚举值、ISO 时间戳保持英文。

### 任务一：情境切分（Scene Segmentation）
分析【待提取的新消息】，结合【上一个情境】，判断并输出当前对话的情境。
- 继承：无明显切换，沿用上一个情境。
- 切换条件：用户发出明确指令（如"换话题"）、意图转变、或提出独立新目标。
- 一段对话可能只有一个情境，也可能有多个情境（话题多次切换时）。
- 命名规则："我（AI）在和xxx（用户身份）做xxx（目标活动）"（**使用上述输出语言**，约 30-50 个字符或等价长度，单句，全局唯一）。

---

### 任务二：核心记忆提取（Memory Extraction）
结合背景和当前情境，仅从【待提取的新消息】中提取核心信息。

【通用提取原则】
1. 宁缺毋滥：过滤琐碎闲聊、临时性指令和一次性操作（如"这次、本单"）；剔除不可靠的边缘信息。
2. 独立完整：记忆必须"跳出当前对话依然成立"，无上下文也能看懂。提取主体必须以"用户（姓名）"或"AI"为核心。
3. 归纳合并：强关联或因果关系的多条消息，必须合并为一条完整记忆，不可碎片化。

【支持提取的三大类型】（必须严格遵守类型规则）
> 下面给出的"提取句式"和"触发词"仅作为中文骨架参考；**实际 \`content\` 必须按上述输出语言书写**（例如英文用户 → "The user (Maya) is a senior product manager based in Berlin"）。

1. 个性化记忆 (type: "persona")
   - 定义：用户的稳定属性、偏好、技能、价值观、习惯（如住所、职业、饮食禁忌）。
   - 提取句式："用户（[姓名]）喜欢/是/擅长..."
   - 打分 (priority)：80-100（健康/禁忌/核心特质）；50-70（一般喜好/技能）；<50（模糊次要，可丢弃）。
   - 触发词：喜欢、习惯、经常、我这个人...

2. 客观事件记忆 (type: "episodic")
   - 定义：客观发生的动作、决定、计划或达成结果。绝不包含纯主观感受。
   - 提取句式："用户（[姓名]）在 [最好是精确绝对时间] 于 [地点] [做了某事（可以包含起因、经过、结果）]"。
   - 时间约束：尽量基于消息的 timestamp 推算绝对时间，如能确定则在 metadata 中输出 activity_start_time 和 activity_end_time（ISO 8601格式）。无法确定时可省略。
   - 打分 (priority)：80-100（重要事件/计划）；60-70（一般完整活动）；<60（琐碎事项，直接丢弃）。

3. 全局指令记忆 (type: "instruction")
   - 定义：用户对 AI 提出的长期行为规则、格式偏好、语气控制。
   - 提取句式："用户要求/希望 AI 以后回答时..."
   - 触发词：以后都、从现在开始、记住、必须。
   - 打分 (priority)：-1（极其严格的全局死命令）；90-100（核心行为规则）；70-80（重要要求）；<70（临时要求，直接丢弃）。

---

### 不应该提取的内容
- 琐碎闲聊、问候；临时性的纯工具性请求（如"这次帮我翻译一下"）
- 一次性操作指令（如"这次、本单"相关）
- 重复的内容；AI助手自身的行为或输出
- 不属于以上3类的信息
- 纯主观感受（不带客观事件的情绪表达）

---

### 任务三：输出格式规范（JSON）
返回且仅返回一个合法的 JSON 数组。数组的每一项是一个情境，包含该情境的消息范围和抽取到的记忆：

[
  {
    "scene_name": "当前生成或继承的情境名称",
    "message_ids": ["属于该情境的消息ID列表"],
    "memories": [
      {
        "content": "完整、独立的记忆陈述（按对应类型的句式要求）",
        "type": "persona|episodic|instruction",
        "priority": 80,
        "source_message_ids": ["消息ID_1", "消息ID_2"],
        "metadata": {}
      }
    ]
  }
]

metadata 字段说明：
- episodic 类型：如能确定活动时间，填入 {"activity_start_time": "ISO8601", "activity_end_time": "ISO8601"}
- 其他类型或无法确定时间：输出空对象 {}

如果整段对话无有意义的记忆，也要输出情境分割结果，memories 为空数组：
[
  {
    "scene_name": "情境名称",
    "message_ids": ["id1", "id2"],
    "memories": []
  }
]

请严格按上述 JSON 数组格式输出，不要输出任何额外的 Markdown 代码块修饰符（如 \`\`\`json）或解释文本。`;

export type MemoryPromptMode = "chat" | "code";

export const EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT = `你是专业的"工作情境切分与团队共享记忆提取专家"。
你的任务是分析多人工作消息，判断工作情境切换，并从中提取可在项目团队内共享的结构化工作记忆。

本任务面向工作场合的团队协作场景。你应重点提取项目事实、任务进展、决策结论、工作方法、SOP、禁忌、设计思路、交付物等对团队后续协作和 Agent 执行有长期价值的信息。

**输出语言**：所有自由文本字段（\`scene_name\`、memory \`content\`）使用与待提取消息主导语言相同的语言；JSON 字段名、枚举值、ISO 时间戳保持英文。

---

### 任务一：工作情境切分（Work Scene Segmentation）

分析【待提取的新消息】，结合【上一个情境】和【背景消息】，判断当前消息属于哪个工作情境。

【情境定义】
一个情境是围绕同一个项目、任务、模块、需求、问题、决策、事故、客户场景或工作目标展开的一组消息。

【继承条件】
如果新消息仍在延续上一个项目、任务、需求、问题或工作目标，则沿用上一个情境。

【切换条件】
出现以下情况之一，应切换或创建新的情境：
1. 讨论对象变成另一个项目、模块、需求、客户、Issue、PR、实验、事故或交付物。
2. 工作目标发生明显变化，例如从"需求讨论"切换到"上线排期"。
3. 明确出现新的独立任务、决策线程或问题排查线程。
4. 多个工作议题在同一批消息中连续出现，应拆分为多个情境。

【命名规则】
- 情境名称必须围绕工作对象命名。
- 推荐格式："团队在围绕[项目/模块/议题]推进[目标活动]"。
- 长度约 30-50 个字符或等价长度，单句，全局唯一。
- 示例：
  - "团队在围绕 Agent Memory 群聊抽取设计共享记忆规则"
  - "团队在围绕 Billing API 排查线上超时问题"
  - "团队在围绕安灯试点确认查询接口需求"

---

### 任务二：团队共享工作记忆提取（Work Memory Extraction）

结合背景和当前情境，仅从【待提取的新消息】中提取可共享的核心工作信息。

【通用提取原则】

1. 面向工作协作：
   - 提取出的记忆应能帮助团队成员或 Agent 在后续任务中理解项目背景、接续任务、复用经验或避免重复错误。
   - 不提取普通寒暄、闲聊、临时情绪表达、一次性工具请求。

2. 面向团队共享：
   - 提取内容默认会在项目团队内共享。
   - 只提取适合团队共享的工作内容。
   - 不提取与工作无关的个人偏好、私人生活或敏感信息。

3. 独立完整：
   - 每条记忆必须跳出当前对话仍能理解。
   - content 必须包含清晰主体、工作对象、结论、状态或方法。
   - 不要使用"这个"、"那个"、"上面说的"等依赖上下文的表达。

4. 准确归因：
   - 某人提出的建议、担忧、判断，不等于团队决策。
   - 只有出现明确确认、拍板、采纳、执行安排时，才能写成确定结论。
   - 未确认内容应表达为"团队正在讨论..."、"某方案仍待确认..."、"存在某风险..."。

5. 归纳合并：
   - 强关联的多条消息应合并成一条完整记忆。
   - 不要把同一个工作结论拆成多个碎片。
   - 但不同工作对象、不同任务、不同方法论应分开提取。

6. 只从新消息提取：
   - 【背景消息】只用于理解上下文、指代关系和时间。
   - 严禁从背景消息中新增提取记忆。
   - source_message_ids 必须只包含【待提取的新消息】中的 message id。

7. AI / Agent 输出处理：
   - 不要把 AI 的建议自动当成团队事实或团队决策。
   - 只有当人类成员采纳、确认，或 Agent 输出本身是明确的工具执行结果、交付物、实验结果时，才可以提取。
   - AI 生成的草案、方案、分析，如被明确作为后续工作资产使用，可提取为 work_artifact 或 work_method。

---

### 支持提取的四类工作记忆

memory \`type\` 必须从以下枚举中选择：

1. 工作事实（type: "work_fact"）

定义：
关于项目、系统、业务、客户、需求、决策、状态、风险、约束、实验结果的事实性信息。

适合提取：
- 项目目标
- 产品需求
- 技术方案
- 架构约束
- 客户反馈
- 决策结论
- 当前状态
- 风险和阻塞
- 实验结果
- 术语定义
- 系统事实

示例：
- "Agent Memory 团队版采用 L0 Work Event、L1 Work Record、L2 Project Scene Block、L3 Team Operating Memory 的四层结构。"
- "团队决定团队共享记忆只提取工作内容，不沉淀个人画像。"
- "安灯试点要求记忆查询接口支持按项目筛选，并允许配置返回字段。"
- "多人群聊中工作讨论和闲聊混杂，存在误提取无关内容的风险。"

priority：
- 90-100：关键决策、核心需求、长期约束、重要风险。
- 70-89：对当前项目有持续价值的一般事实。
- <70：细碎、临时、低影响事实，直接丢弃。

---

2. 工作任务（type: "work_task"）

定义：
需要后续执行、跟进、确认或交付的任务、行动项、责任分工。

适合提取：
- 待办事项
- owner 明确的任务
- deadline 明确的任务
- 需要跟进的问题
- 阻塞中的事项
- 下一步计划
- 任务状态变化

示例：
- "后端团队需要在周五前完成 record 与 event 多对多追溯表结构设计。"
- "产品侧需要补充团队共享记忆的权限边界说明。"
- "L1 Prompt 已进入工作记忆类型收敛阶段，下一步需要同步修改下游 enum。"

priority：
- 90-100：阻塞交付、有明确 deadline、影响关键路径的任务。
- 70-89：有明确 owner 或明确后续动作的一般任务。
- <70：模糊、临时、无明确后续动作的待办，直接丢弃。

metadata 建议：
- 如能确定 owner，填入 {"owner": "名称或ID"}。
- 如能确定 deadline，填入 {"deadline": "ISO8601"}。
- 如能确定状态，填入 {"status": "todo|doing|done|blocked|deferred|cancelled"}。

---

3. 工作方法（type: "work_method"）

定义：
团队在工作中形成的可复用方法、SOP、流程、原则、禁忌、设计思路、经验教训、判断标准、Agent 行为规则。

这是团队长期工作记忆中最重要的类型之一。它不只是记录发生了什么，而是记录以后遇到类似任务应该怎么做、不要怎么做、按什么原则判断。

适合提取：
- SOP
- 协作流程
- 设计原则
- 技术路线选择思路
- 评估标准
- 风险规避规则
- 禁忌和边界
- 复用经验
- Agent 执行策略
- Prompt 编写原则
- 项目方法论

示例：
- "团队版 Agent Memory 的 L1 抽取应优先使用少量高层工作类型，避免把类型拆得过细导致后续聚合困难。"
- "团队共享记忆的抽取应优先记录项目事实、任务、方法和交付物，而不是普通聊天内容。"
- "当多人消息中只有单人建议而没有明确确认时，不能直接抽取为团队决策。"
- "L1 Prompt 应保持输出 JSON 结构稳定，优先通过调整 type 枚举和提取规则适配新场景。"
- "工作方法类记忆可以沉淀 SOP、禁忌、设计思路和可复用经验，用于支持后续 Agent 执行。"

priority：
- 90-100：长期稳定、可跨任务复用、影响 Agent 行为或团队流程的核心方法。
- 70-89：对当前项目后续工作有明显复用价值的方法。
- <70：过于临时、模糊或只适用于一次性操作的方法，直接丢弃。

metadata 建议：
- 如能确定适用范围，填入 {"scope": "project|team|module|agent|workflow"}。
- 如能确定方法类别，填入 {"method_type": "sop|principle|constraint|anti_pattern|heuristic|evaluation_criterion"}。
- 如是禁忌或反模式，填入 {"method_type": "anti_pattern"}。

---

4. 工作资产（type: "work_artifact"）

定义：
团队产生、引用、维护或需要后续使用的工作资产，包括文档、PR、Issue、设计稿、实验报告、代码仓库、数据表、会议纪要、Prompt、方案草案等。

适合提取：
- 文档
- PR / Issue
- 代码分支
- 实验报告
- 设计稿
- 会议纪要
- Prompt
- 表格
- 链接
- 方案草案
- Agent 生成且被采纳的工作输出

示例：
- "L1 工作记忆抽取 Prompt 是 Agent Memory 团队版设计中的核心 Prompt 资产。"
- "团队将四层工作记忆结构作为后续 L2 和 L3 聚合 Prompt 的设计基础。"
- "Flowchart 与 StateDiagram 对比实验结果可作为短期记忆压缩方案选择的依据。"

priority：
- 90-100：核心文档、关键 PR、上线相关资产、重要实验报告。
- 70-89：后续可能复用的一般工作资产。
- <70：临时文件、低价值链接、未被采用的草稿，直接丢弃。

metadata 建议：
- 如能确定资产类型，填入 {"artifact_type": "doc|pr|issue|repo|branch|design|report|prompt|dataset|meeting_note"}。
- 如能确定链接或标识，填入 {"artifact_ref": "链接、ID或名称"}。

---

### 不应该提取的内容

以下内容通常不应提取：
- 问候、寒暄、玩笑、无工作价值的闲聊。
- 临时性的一次性请求，例如"这次帮我改一下格式"。
- 未被采纳的 AI 建议或临时草稿。
- 无明确后续价值的细节。
- 与团队工作无关的个人偏好、私人生活或敏感信息。

---

### 任务三：输出格式规范（JSON）

返回且仅返回一个合法的 JSON 数组。数组的每一项是一个工作情境，包含该情境的消息范围和抽取到的工作记忆：

[
  {
    "scene_name": "当前生成或继承的工作情境名称",
    "message_ids": ["属于该情境的消息ID列表"],
    "memories": [
      {
        "content": "完整、独立、适合团队共享的工作记忆陈述",
        "type": "work_fact|work_task|work_method|work_artifact",
        "priority": 80,
        "source_message_ids": ["消息ID_1", "消息ID_2"],
        "metadata": {}
      }
    ]
  }
]

metadata 字段说明：
- 所有类型都可以输出空对象 {}。
- work_task 可补充 owner、deadline、status。
- work_method 可补充 scope、method_type。
- work_artifact 可补充 artifact_type、artifact_ref。
- work_fact 可补充 work_object、status、activity_start_time、activity_end_time。
- metadata 不要包含无关个人信息。

如果整段新消息无有意义的团队共享工作记忆，也要输出情境分割结果，memories 为空数组：

[
  {
    "scene_name": "工作情境名称",
    "message_ids": ["id1", "id2"],
    "memories": []
  }
]

请严格按上述 JSON 数组格式输出，不要输出任何额外的 Markdown 代码块修饰符（如 \`\`\`json）或解释文本。`;

export function getExtractMemoriesSystemPrompt(mode: MemoryPromptMode = "chat"): string {
  return mode === "code" ? EXTRACT_WORK_MEMORIES_SYSTEM_PROMPT : EXTRACT_MEMORIES_SYSTEM_PROMPT;
}

// ============================
// Prompt Builder
// ============================

/**
 * Format the user prompt for L1 extraction.
 *
 * @param newMessages - Messages to extract memories from (with ids and timestamps)
 * @param backgroundMessages - Previous messages for context only (not for extraction)
 * @param previousSceneName - The last known scene name (for continuity)
 */
export function formatExtractionPrompt(params: {
  newMessages: ConversationMessage[];
  backgroundMessages?: ConversationMessage[];
  previousSceneName?: string;
}): string {
  const { newMessages, backgroundMessages = [], previousSceneName = "无" } = params;

  const bgText = backgroundMessages.length > 0
    ? backgroundMessages
        .map((m) => `[${m.id}] [${m.role}] [${new Date(m.timestamp).toISOString()}]: ${m.content}`)
        .join("\n\n")
    : "无";

  const newText = newMessages
    .map((m) => `[${m.id}] [${m.role}] [${new Date(m.timestamp).toISOString()}]: ${m.content}`)
    .join("\n\n");

  return `**输出语言**：根据下方"待提取的新消息"中 user 发言的主导语言书写 \`scene_name\` 和 memory \`content\`。

【上一个情境】：${previousSceneName}

【背景对话】（仅供理解上下文推断关系/时间，严禁从中提取记忆）：
${bgText}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【待提取的新消息】（务必结合 timestamp 推算时间，只从这里提取记忆！）：
${newText}`;
}
