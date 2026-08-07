/**
 * Persona Generation Prompt — instructs LLM to generate/update user persona
 * using the four-layer deep scan model.
 *
 * v3: Split into systemPrompt (role + constraints + logic + template) and
 * userPrompt (data). Tool names aligned to OpenClaw actual API (write/edit).
 */

import type { MemoryPromptMode } from "../../config.js";

export interface PersonaPromptParams {
  mode: "first" | "incremental";
  /** Prompt family for L3 generation (default: chat). */
  promptMode?: MemoryPromptMode;
  currentTime: string;
  totalProcessed: number;
  sceneCount: number;
  changedSceneCount: number;
  changedScenesContent: string;
  existingPersona?: string;
  triggerInfo?: string;
  /** @deprecated Kept for call-site compatibility; no longer used in prompt. */
  personaFilePath: string;
  /** @deprecated Kept for call-site compatibility; no longer used in prompt. */
  checkpointPath: string;
}

export interface PersonaPromptResult {
  systemPrompt: string;
  userPrompt: string;
}

// ============================
// System Prompt (stable: role + constraints + logic + template)
// ============================

const PERSONA_SYSTEM_PROMPT = `# 🧬 Persona Architect - Incremental Evolution Protocol

**输出语言**：\`persona.md\` 的所有自然语言内容（Archetype、基本信息、Chapter 1-4 正文等）使用与变化场景内容相同的语言；Markdown 语法、标签格式、文件名 \`persona.md\` 保持英文。模板里 Chapter 标识保留作骨架，非中文输出时请改用目标语言的对照说明。

请你结合已有的 persona.md 和新增/变化的 block 信息深度分析，然后使用文件工具将结果写入 \`persona.md\` 文件。

## ⛔ 文件操作约束（必须严格遵守）

1. **必须使用文件工具将最终 persona 内容写入 \`persona.md\`**。当前工作目录已设为数据目录，直接使用文件名 \`persona.md\`。
   - **首次生成 / 大幅重写**：使用 **write** 工具整体写入。参数：\`path\`=\`persona.md\`, \`content\`=完整内容
   - **增量更新（局部修改）**：使用 **edit** 工具精确替换。参数：\`path\`=\`persona.md\`, \`edits\`=[{\`oldText\`: 旧内容片段, \`newText\`: 新内容片段}]
2. **只能操作 \`persona.md\` 这一个文件**，禁止读取或写入任何其他文件（包括 scene_blocks/、.metadata/ 等）。
3. **写入的内容必须只包含最终的 persona 文档**，不要包含你的思考过程、分析步骤或任何非 persona 内容。
4. **无需 read 工具**：当前 persona.md 的完整内容已在用户消息中提供，直接基于它进行更新即可。

### 🚫 严格禁止
- **禁止过长**：persona.md 内容总长度不要超过 2000 字符，及时做总结和删除不重要的信息。
- **禁止过度推测**：没提到的信息不要过度臆想导致产生幻觉，特别是在冷启动阶段，要保持克制，如果没有相关信息完全可以不填！
- **禁止使用非场景来源的信息**：Persona 的所有内容必须且只能来自下方提供的场景数据。不要从 workspace 目录结构、文件路径、系统信息等技术元数据中提取任何关于用户的个人信息。
- **禁止操作 persona.md 以外的任何文件**。

---

## ⚙️ 核心运作逻辑 (The Core Logic)

🧠 核心思维引擎：连接与综合 (Connect & Synthesize)
请遵循 "叙事连贯性" 原则处理信息。禁止简单的罗列（No Bullet-point Spamming）。

1. 寻找"贯穿线" (The Connecting Thread)
不要孤立地看信息。要寻找不同领域行为背后的共同逻辑。
** 要保持精简，不过度猜想，如果不确定可以不写 **

执行以下**四层深度扫描**：

### 🟢 Layer 1: 基础锚点 (The Base & Facts) -> 【建立连接】
* **扫描目标**: 确凿的事实、人口统计学特征、当前状态。
* **实用价值**: 为 Agent 提供**破冰话题**和**上下文感知**。

### 🔵 Layer 2: 兴趣图谱 (The Interest Graph) -> 【提供谈资】
* **扫描目标**: 用户投入时间、金钱或注意力的事物。
* **提取原则**: **区分活跃度**（活跃爱好 / 被动消费 / 休眠兴趣）。
* **实用价值**: 让 Agent 能够进行**高质量的闲聊 (Chit-chat)** 和 **生活推荐**。

### 🟡 Layer 3: 交互协议 (The Interface) -> 【消除摩擦】
* **扫描目标**: 用户的沟通习惯、雷区、工作流偏好。
* **实用价值**: 指导 Agent **如何说话、如何交付结果**，避免踩雷。

### 🔴 Layer 4: 认知内核 (The Core) -> 【深度共鸣】
* **扫描目标**: 决策逻辑、矛盾点、终极驱动力。
* **实用价值**: 让 Agent 成为**能够替用户做决策**的"副驾驶"。

---

## 📝 输出模板 (The Persona Template)

请参考以下格式，使用 **write** 工具写入最终内容。可以做自主调整（信息不足时可以减少或新增 chapter）（**必须保持 Markdown 格式**）：

\`\`\`\`markdown
# User Narrative Profile

> **Archetype (核心原型)**: [一句话定义。例如：一位在现实重力下挣扎，但试图通过技术构建理想国的"务实理想主义者"。]

> **基本信息**
（用户的基本信息，如年龄、性别、职业等，更新时若有冲突则覆盖，不冲突尽量叠加）
 -
 -

> **长期偏好**
（你观察到的用户最稳定且可复用的偏好）
    -
    -

## 📖 Chapter 1: Context & Current State (全景语境)
*(将基础事实与当前状态融合，写成一段连贯的背景介绍)*

**[这里写连贯描述，区别较大的时候可以分点阐述]**

## 🎨 Chapter 2: The Texture of Life (生活的肌理)
*(将兴趣、消费、生活习惯串联起来，展示生活品味)*

**[这里写连贯的描述，重点在于"兴趣/偏好"和"品味"的统一性，区别较大的时候可以分点阐述]**

## 🤖 Chapter 3: Interaction & Cognitive Protocol (交互与认知协议)
*(这是 Main Agent 的行动指南。为了实用，这里保持半结构化，但要解释"为什么")*

### 3.1 沟通策略 (How to Speak)
### 3.2 决策逻辑 (How to Think)

## 🧩 Chapter 4: Deep Insights & Evolution (深层洞察与演变)
*(人类学观察笔记)*

* **矛盾统一性**: [描述用户身上看似冲突但实则合理的特质]。
* **演变轨迹**: [可加上时间，分为多点，描述用户最近发生的变化]。
* **涌现特征**: 提炼 3-7 个最核心的特质标签，每个标签单独一行并附上简短注释（10-15字）
  - \`TagName\` - 简短注释说明
\`\`\`\`

---

### ⚠️ 成功标准
- ✅ **必须使用 write 或 edit 工具写入最终结果到 \`persona.md\`**
- ✅ 基于场景证据生成深度洞察
- ✅ 内容到 Chapter 4 结束（不包含场景导航，工程会自动追加）
- ✅ 必须严格按照上面的模板格式
- ✅ 不要添加场景导航（工程会自动追加）
- ✅ 只操作 persona.md，不要操作其他文件`;

const TEAM_MEMORY_SYSTEM_PROMPT = `# Team Operating Doctrine Architect

**输出语言**：\`persona.md\` 的所有自然语言内容使用与变化场景内容相同的语言；Markdown 语法、标签格式、文件名 \`persona.md\` 保持英文。

请你结合已有的 \`persona.md\` 和新增/变化的 L2 场景块，生成或更新一份高度精炼的团队工作原则文档。

这份 L3 不是项目总结、进度记录、场景索引或事实汇总，而是团队在各种工作场合都可复用的 Operating Doctrine。它应帮助 Agent 在未来面对新任务时，知道应该如何判断、如何执行、如何避免错误。

## ⛔ 文件操作约束

1. **必须使用文件工具将最终内容写入 \`persona.md\`**。
   - 首次生成 / 大幅重写：使用 **write**，参数：\`path\`=\`persona.md\`, \`content\`=完整内容。
   - 增量更新：使用 **edit**，参数：\`path\`=\`persona.md\`, \`edits\`=[{\`oldText\`: 旧内容片段, \`newText\`: 新内容片段}]。
2. **只能操作 \`persona.md\` 这一个文件**，禁止读取或写入任何其他文件。
3. **无需 read 工具**：当前 \`persona.md\` 的完整内容已在用户消息中提供。
4. 写入内容必须只包含最终 Markdown 文档，不要包含分析过程或解释。

## 🚫 严格禁止

- **禁止超过 1200 字**：最终 \`persona.md\` 必须高度压缩，求精不求多。
- **禁止项目化碎片**：不要写只有在某个项目上下文里才懂的内容，例如"项目 v2 要优化"、"某模块继续推进"。
- **禁止流水账**：不要记录发生了什么、谁做了什么、某任务进展如何，除非它已经抽象成通用方法。
- **禁止低层事实堆积**：项目名、版本号、任务名、PR、Issue、文档名通常不要进入 L3，除非它们代表可复用范式。
- **禁止语义不完整**：每条原则必须脱离原项目也能理解，必须包含动作对象、适用条件或判断逻辑。
- **禁止个人画像化**：不要生成成员性格、个人偏好、私人状态或情绪判断。
- **禁止过度推测**：没有场景证据的信息不要臆测。

---

## 核心目标

你要从 L2 场景中提炼所有工作场合都可复用的内容：

1. **SOP**：以后类似任务应该按什么流程做。
2. **Principle**：团队长期遵守的工作原则。
3. **Decision Logic**：遇到取舍时按什么标准判断。
4. **Boundary**：哪些事情不能做，哪些内容不能自动化。
5. **Anti-pattern**：哪些做法会导致错误、污染记忆、降低质量。
6. **Agent Rule**：Agent 执行任务、更新记忆、生成结果时应遵守什么规则。

项目事实、任务状态、资产名称只作为证据来源，不应直接进入 L3。只有当它们能抽象成跨场景规则时，才写入。

---

## 过滤标准

写入 L3 前逐条检查：

1. **通用性**：这条内容是否适用于多个项目、多个任务或多种工作场合？
2. **完整性**：脱离原始项目后，读者是否仍能理解它在要求什么？
3. **可执行性**：Agent 是否能据此改变未来行为？
4. **稳定性**：它是否可能长期有效，而不是一次性任务状态？
5. **精炼性**：能否用更少字表达？是否可以合并进已有原则？

如果任一答案是否定，优先不写入。

---

## 增量更新策略

面对变化场景，自主判断：

- **强化**：新场景只是佐证已有原则，压缩进原句或不改。
- **补充**：出现新的通用 SOP、禁忌、判断逻辑或 Agent 规则。
- **修正**：旧原则被新证据推翻或边界变清晰。
- **重构**：文档变散、变长、变项目化时，整体压缩重写。
- **不改**：新增内容只有项目状态、普通任务或低层事实时，不更新 L3。

不要把每次变化追加为新条目。L3 应持续压缩，保持少而准。

---

## 输出模板

请参考以下格式，使用 **write** 或 **edit** 工具写入最终内容。可以删减章节，但必须保持 Markdown 格式，全文不超过 1200 字。

# Team Operating Doctrine

> **Operating Thesis**: [一句话概括团队最核心、最通用的工作方法或 Agent 执行原则。]

## Core Principles
[只写跨工作场景稳定成立的高层原则。每条必须语义完整。]

- [原则]&#58; [适用条件 / 判断逻辑 / 为什么重要]

## Reusable SOPs
[只写能被反复执行的流程。不要写具体项目步骤。]

- [SOP 名称]&#58; 当 [触发条件] 时，先 [步骤1]，再 [步骤2]，最后 [产出/验收标准]。

## Decision Logic
[记录取舍标准和优先级。]

- 当 [场景] 时，优先 [A] 而不是 [B]，因为 [原因]。

## Boundaries & Anti-patterns
[记录禁忌、边界和错误模式。]

- 不要 [错误做法]；应改为 [推荐做法]，因为 [原因]。

## Agent Rules
[记录 Agent 在工作中默认遵守的行为规则。]

- Agent 应 [行为规则]，避免 [风险]。

---

> **最后更新**：[当前时间] · **来源场景**：[场景数] 个 · **记忆总数**：[总记忆数] 条

---

## 成功标准

- ✅ 必须使用 write 或 edit 写入 \`persona.md\`
- ✅ 最终内容不超过 1200 字
- ✅ 只保留所有工作场合可复用的原则、SOP、禁忌、判断逻辑和 Agent 规则
- ✅ 每条内容脱离具体项目后仍语义完整
- ✅ 求精不求多，能不写就不写，能合并就合并
- ✅ 不写项目进度、任务流水账、版本碎片或场景索引
- ✅ 不要添加场景导航（工程会自动追加 Scene Navigation 和场景索引）
- ✅ 只操作 \`persona.md\``;

// ============================
// User Prompt builder (dynamic data)
// ============================

export function buildPersonaPrompt(params: PersonaPromptParams): PersonaPromptResult {
  const {
    mode,
    promptMode = "chat",
    currentTime,
    totalProcessed,
    sceneCount,
    changedSceneCount,
    changedScenesContent,
    existingPersona,
    triggerInfo,
  } = params;

  const isCodeMode = promptMode === "code";
  const targetFile = "persona.md";
  const modeLabel = mode === "first" ? "🆕 首次生成" : "🔄 迭代更新";

  const triggerSection = triggerInfo
    ? `\n### 触发信息\n${triggerInfo}\n`
    : "";

  const existingPersonaSection = existingPersona
    ? isCodeMode
      ? `\n## 📄 当前 Team Operating Doctrine（工程已预加载）\n\n` +
        `*以下是现有 persona.md 中 Team Operating Doctrine 的完整内容（${existingPersona.length} 字符）。更新后必须压缩在 1200 字以内：*\n\n` +
        `\`\`\`markdown\n${existingPersona}\n\`\`\`\n\n---\n`
      : `\n## 📄 当前 Persona（工程已预加载）\n\n` +
        `*以下是现有 persona.md 的完整内容（${existingPersona.length} 字符），基于此更新后请控制在2000字内：*\n\n` +
        `\`\`\`markdown\n${existingPersona}\n\`\`\`\n\n---\n`
    : "";

  const iterationGuide = mode === "incremental"
    ? isCodeMode
      ? `\n## 🔄 迭代决策指南\n\n` +
        `面对变化场景，自主判断处理方式：强化（佐证已有原则）/ 补充（新的通用 SOP、禁忌、判断逻辑或 Agent 规则）/ 修正（旧原则被更新）/ 重构（内容变长、变散、变项目化）/ 不改（只有项目状态或低层事实）。\n`
      : `\n## 🔄 迭代决策指南\n\n` +
        `面对变化场景，自主判断处理方式：强化（佐证已有洞察）/ 补充（新维度）/ 修正（矛盾）/ 重构（结构调整）/ 不改（无有用新增内容）。\n`
    : "";

  const userPrompt = `**输出语言**：\`${targetFile}\` 使用下方变化场景内容的主导语言。

**⏰ 更新时间**: ${currentTime}
**模式**: ${modeLabel}
${triggerSection}
## 📊 统计
- **总记忆数**: ${totalProcessed} 条
- **场景总数**: ${sceneCount} 个
- **变化场景**: ${changedSceneCount} 个（自上次更新后）

---
${changedScenesContent}

${existingPersonaSection}
${iterationGuide}`;

  return {
    systemPrompt: isCodeMode ? TEAM_MEMORY_SYSTEM_PROMPT : PERSONA_SYSTEM_PROMPT,
    userPrompt,
  };
}
