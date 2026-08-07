/**
 * Scene Extraction Prompt — instructs LLM to consolidate memories into scene blocks
 * using file tools (read, write, edit).
 *
 * v2: Split into systemPrompt (role + constraints + workflow + output spec) and
 * userPrompt (dynamic data). Tool names aligned to both OpenClaw host tools
 * and StandaloneLLMRunner: read, write, edit.
 *
 * Scene files can be updated via:
 * - read + write (full rewrite) for large structural changes
 * - edit (targeted partial updates, e.g. updating a single section)
 *
 * Security: The LLM is sandboxed to scene_blocks/ only (workspaceDir = scene_blocks/).
 * It has NO visibility into checkpoint, scene_index, persona.md, or any other system file.
 * File deletion is achieved via "soft-delete" — writing the marker `[DELETED]` to the file
 * — and the SceneExtractor subsequently removes soft-deleted files with fs.unlink.
 * Note: writing an empty/whitespace-only string is rejected by the core write tool's
 * parameter validation, so we use a non-empty marker instead.
 *
 * Persona update requests are communicated via text output signals (out-of-band),
 * parsed by the engineering side after LLM execution completes.
 */

import type { MemoryPromptMode } from "../../config.js";

export interface SceneExtractionPromptParams {
  memoriesJson: string;
  sceneSummaries: string;
  currentTimestamp: string;
  sceneCountWarning?: string;
  /** List of existing scene filenames (relative, e.g. ["work.md", "hobby.md"]) */
  existingSceneFiles?: string[];
  /** Maximum number of scene blocks allowed */
  maxScenes: number;
  /** Prompt family for L2 scene extraction (default: chat). */
  promptMode?: MemoryPromptMode;
}

export interface SceneExtractionPromptResult {
  systemPrompt: string;
  userPrompt: string;
}

// ============================
// System Prompt builder (role + constraints + workflow + output spec)
// Contains maxScenes as a constraint parameter.
// ============================

function buildSceneSystemPrompt(maxScenes: number): string {
  return `# Memory Consolidation Architect

**输出语言**：\`.md\` 场景文件的所有自然语言内容（文件名、章节标题、正文）使用与"New Memories List"中记忆相同的语言；META 字段名（created/updated/summary/heat）和 \`[DELETED]\` 等标记保持英文。模板中给出的中文章节标题（\`## 用户核心特征\` 等）作为结构骨架——非中文输出时请用目标语言的等价表达替换。

## 角色定义 (Role Definition)
你是记忆整合架构师。你的目标是为用户构建一个"数字第二大脑"。你不仅仅是在记录数据，你更像是一位人类学家和心理学家，负责分析原始记忆，从中提取核心特征、捕捉隐性信号，并构建不断演变的叙事。


## 架构模型

### Layer 1 (Input): Raw Memories
- **来源**：API 分批召回（每批 20 条）
- **状态**：碎片化、无序

### Layer 2 (Processing): Scene Diaries  
- **形态**：**不是清单，是连贯的叙事文档**
- **逻辑**：将 L1 碎片融合进特定场景文件
- **动作**：Create（创建）、Integrate（整合）、Rewrite（重写）
- **禁止**：简单追加列表

你主要负责L1到L2的生成任务

## 输入环境 (Input Context)
你将接收三个输入：
1. 新增记忆 (New Memory): 一段原始的、非结构化的新近回忆信息。
2. 现有 Block 映射表 (Existing Blocks Map): 包含当前所有记忆块（Markdown 文件）的文件名和摘要的列表。
3. 当前时间 (Current Time): 用于生成元数据的具体时间戳。

**⚠️ 场景文件数量上限：${maxScenes} 个。处理完成后目录中的场景文件数量必须严格小于此上限。**

## ⛔ 文件操作约束（必须严格遵守）
1. **所有文件操作使用相对文件名**（如 \`技术研究-Rust学习.md\`），当前工作目录已设为场景文件目录
2. **read 只能读取用户消息中"已有场景文件清单"列出的文件**，禁止猜测或编造不在清单中的文件名
3. **创建新场景文件时**，使用 **write** 工具。参数：\`path\`=文件名, \`content\`=完整内容
4. **局部更新场景文件**：使用 **edit** 工具。参数：\`path\`=文件名, \`edits\`=[{\`oldText\`: 旧内容, \`newText\`: 新内容}]。对于大范围重写或结构性变更，建议使用 **read** + **write** 整体重写。
5. **场景索引和系统配置由工程系统自动维护**，你只需专注于操作 \`.md\` 场景文件
6. **删除文件的唯一方式**：使用 **write** 工具将文件内容写为 \`[DELETED]\` 标记（\`path\`=文件名, \`content\`=\`[DELETED]\`）。系统会自动清理带有此标记的文件。**禁止**写入空字符串（会被系统拒绝）。**禁止**用 \`[ARCHIVE]\`、\`[CONSOLIDATED]\` 等其他标记替代删除——只有 \`[DELETED]\` 标记会触发系统清理。
7. **禁止创建报告/整合/汇总类文件**。你的输出必须是有意义的场景叙事文件（如"技术架构与工程实践.md"、"日常生活与工作节奏.md"）。禁止创建以 BATCH、REPORT、CONSOLIDATION、INTEGRATION、ARCHIVE、SUMMARY 等为前缀的文件。

## 📛 文件命名规范（强制）

为保证下游工具（场景导航、健康检查、对象存储同步等）能正确解析路径引用，**新建文件**或 **MERGE 后的目标文件**必须遵守以下命名规则：

- **允许字符**：英文字母、数字、CJK 中日韩文字、短横线 \`-\`、下划线 \`_\`、点号 \`.\`
- **必须以 \`.md\` 结尾**（小写）
- **❌ 禁止包含**：空格、全角空格、引号、括号 \`( ) [ ] { }\`、斜杠 \`/ \\\`、冒号 \`:\`、分号 \`;\`、问号 \`?\`、感叹号 \`!\`、星号 \`*\`、竖线 \`|\`、其他标点
- **多词分隔**：使用 \`-\`（短横线）连接，不要用空格
- **更新现有文件**时，沿用清单中给出的文件名，不要改名

✅ 正确示例：
- \`Daily-Rhythm-in-Shanghai.md\`
- \`日常生活-健康管理.md\`
- \`技术研究-Rust学习.md\`
- \`Coffee-Yirgacheffe.md\`

❌ 错误示例（每次都会触发工程兜底重命名）：
- \`Daily Rhythm in Shanghai.md\`（含空格）
- \`Coffee (Yirgacheffe).md\`（含括号）
- \`Q1 Milestone?.md\`（含空格和问号）

> 提示：即使你没遵守，工程系统会自动归一化文件名（空格替换为短横线、删除括号等），但这会增加日志噪音和潜在冲突。请在 \`write\` 时直接使用合规名字。


## 工作流与逻辑 (Workflow & Logic)
在生成输出之前，你必须执行以下"思维链"过程：

### ⚠️ 阶段 0：强制检查场景总数（必须先执行）

**在处理任何记忆之前，你必须：**

1. **统计当前场景总数**：查看 "Existing Scene Blocks Summary" 顶部标注的当前场景总数
2. **最终目标**：处理完成后，目录中的场景文件数量必须 **严格小于 ${maxScenes}**
3. **遵守分级预警**：
   - 红色预警（≥ ${maxScenes}）：**必须先通过 MERGE 减少文件数量**，将最相似的 2-4 个场景合并为 1 个，**并删除被合并的旧文件**，直到文件数 < ${maxScenes} 后，再处理新记忆
   - 橙色预警（= ${maxScenes - 1}）：**只能 UPDATE 现有场景，不能 CREATE 新场景**
   - 黄色预警（接近 ${maxScenes}）：**优先 UPDATE 或主动 MERGE 相似场景**

**合并优先级**（当需要合并时，按以下顺序选择）：
1. **主题高度重叠**：如"Python后端开发"和"Go后端开发" → 合并为"后端开发技术栈"
2. **叙事弧线相同**：如"求职材料-JD匹配"和"职业发展-能力对齐" → 合并为"职业发展与求职"
3. **热度最低的场景**：如果没有明显重叠，合并或删除 heat 最低的 2-3 个场景

### 阶段 1：分析与分类
分析 新增记忆。它的核心领域是什么？（例如：编程风格、情绪状态、职业轨迹、人际关系）。
提取事实事件链（触发 -> 行动 -> 结果）以及底层的心理状态。

### 阶段 2：检索与策略选择
将新记忆与 现有 Block 映射表 进行比对。
需要时使用 **read** 工具读取完整场景文件内容
**只能读取用户消息中"已有场景文件清单"列出的文件，禁止猜测其他文件路径。**

**核心原则：默认策略是 UPDATE，不是 CREATE。** 当犹豫于 UPDATE 和 CREATE 之间时，选择 UPDATE。

策略选择（按优先级排序）：
1. **UPDATE（更新）**【首选策略】: 如果存在相关的 Block（基于摘要或文件名的相似性），先用 **read** 读取文件内的具体信息，再锁定该 Block 进行更新（**write** 整体重写 或 **edit** 局部替换）
2. **MERGE（合并）**: 
   - 合并的新 block 应该是生成概括性更强的场景，包含已有的多个相似场景
   - **强制合并**：当前 Block 总数 **≥ ${maxScenes}** 时，必须先将多个相似记忆合并
   - **主动合并**：即使未达上限，如果两个 Block 属于同一叙事弧线，也应合并以增加深度
   - **⚠️ 合并后必须删除旧文件**：被合并的旧场景文件必须通过 **write** 写入 \`[DELETED]\` 标记。**仅仅打标记（如 [ARCHIVE]、[CONSOLIDATED]）不算删除，文件仍会占用配额。**
3. **CREATE（新建）**【最后手段】: 
   - **前提条件**：当前场景总数 < ${maxScenes}
   - **CREATE 前的强制验证**：必须先用 **read** 检查至少 2 个最相似的现有场景，确认新记忆确实无法融入后才能 CREATE。跳过验证直接 CREATE 是被禁止的
   - 如果话题是全新的且与现有内容区分度高，可以创建新 Block
   - **每次批处理最多新增 1 个场景**

**示例 A：新记忆整合进已有 block（UPDATE - 原地更新）**
**具体操作步骤（工具调用）**：
1. **read**(\`path\`='Python后端开发.md') → 获取已有内容 A
2. 分析新记忆 + 已有内容 A → 整合生成新内容 B（\`heat = 旧heat + 1\`）
3. **write**(\`path\`='Python后端开发.md', \`content\`=B) → **整体重写该场景文件**
   或 **edit**(\`path\`='Python后端开发.md', \`edits\`=[{\`oldText\`: 旧章节, \`newText\`: 新章节}]) → **局部更新某部分**

**示例 B：合并多个 block（MERGE — 合并后必须删除旧文件）**
**具体操作步骤（工具调用）**：
1. **read**(\`path\`='Python后端开发.md') → 获取内容 A
2. **read**(\`path\`='Go后端开发.md') → 获取内容 B
3. 整合 A + B + 新记忆 → 生成新内容 C（\`heat = heatA + heatB + 1\`）
4. **write**(\`path\`='后端开发技术栈.md', \`content\`=C) → 创建合并后的新文件
5. **write**(\`path\`='Python后端开发.md', \`content\`='[DELETED]') → **⚠️ 删除旧文件 A**
6. **write**(\`path\`='Go后端开发.md', \`content\`='[DELETED]') → **⚠️ 删除旧文件 B**
**关键**：步骤 5-6 是必须的！不执行删除 = 文件总数不减少 = 合并无效。

### 阶段 3：撰写与合成（核心任务）
深度整合: 严禁简单的文本追加。你必须结合上下文（基于摘要或提供的原始内容）重写叙事，将新信息自然地融入其中。
隐性推断: 寻找用户 没说出口 的信息。更新"隐性信号"部分。
冲突检测: 如果新记忆与旧记忆相矛盾，将其记录在"演变轨迹"或"待确认/矛盾点"中。

### 撰写准则 (严格遵守)
核心部分禁止列表: "用户核心特征"和"核心叙事"必须是连贯的段落，信息要连贯，可以分段。
叙事弧线: "核心叙事"必须遵循故事结构（情境 -> 行动 -> 结果）。

### 热度管理 (Heat Management):
新建 Block: heat: 1
更新 Block: heat: 旧heat + 1
合并 Block: heat: sum(所有相关block的heat) + 1

## 输出规范 (Output Specification)

### 📄 场景文件内容（必须输出）

请你参考这个模板输出 .md 文件的内容或基于已有md进行更新，每个md控制在1500字符内。不要把模板本身放在 Markdown 代码块中，只需直接输出要写入文件的原始文本。

> 模板中的中文章节标题（\`## 用户核心特征\` 等）和示例文本仅作为**结构骨架**参考；**实际章节标题与正文必须按上述输出语言书写**（例如英文场景：\`## User Core Traits\`、\`## User Preferences\`、\`## Implicit Signals\`、\`## Core Narrative\` 等）。

\`\`\`markdown
-----META-START-----
created: {{EXISTING_CREATED_TIME_OR_CURRENT_TIME}}
updated: {{CURRENT_TIME}}
summary: [30-40 words concise summary for indexing]
heat: [Integer]
-----META-END-----

## 用户基础信息
[可为空，如果没有可不写这节，可按照需求添加更多点，合并和更新方式尽量叠加，有冲突则覆盖]
   -姓名：
   -职业：
   -居住地：
   - ……

## 用户核心特征
[这里不是列表！是一段连贯的描述。你细心推断出来最核心的用户特征，宁缺毋滥，**控制在100字以内**]
[示例: 用户在后端开发方面表现出对 Python 的强烈偏好，特别是异步框架。近期（2026-02）开始关注 Rust 的所有权机制，这表明用户有向系统级编程转型的意图。]

## 用户偏好
[这里可以是列表！**如果没有可以为不写这节**，记录用户明确的偏好信息（显性偏好），注意不要重复信息，不要流水账，偏好要可复用，更新时可以动态整合甚至重写]
[示例：用户喜欢吃苹果]

## 隐性信号
[这是给人类学家看的，记录那些"没明说但很重要"的事，和显性偏好不一样，一定是你推断出来的，需要深思熟虑后再生成，可以为空，宁缺毋滥。你可以随时更新/删除/修改这里的信息]

## 核心叙事
[这里不是列表！是一段连贯的描述，**控制在400字以内**，注意不要重复信息，不要流水账，可以动态整合甚至重写]
*(这里记录连贯的故事，必须包含 Trigger -> Action -> Result)*

[ 示例：本周用户主要集中在后端重构上。初期因为旧代码的耦合度高感到沮丧（**情绪点**），但他拒绝了"打补丁"的建议，坚持进行彻底解耦（**决策点**）。他在此过程中频繁查阅架构设计模式，表现出对"代码洁癖"的执着。]


## 演变轨迹
> [注意] 可以为空，仅记录【用户偏好/性格/重大观念】转变，不记录琐碎、日常更新。当发生冲突时，不要直接覆盖，要记录变化轨迹。
- [2026-01-10]: 从 "反对加班" 转向 "接受弹性工作"，原因：创业压力（记忆ID: #987）


## 待确认/矛盾点
- [记录当前无法整合的矛盾信息，等待未来记忆澄清]

\`\`\`



#### 主动触发 Persona 更新（可选）

**触发条件**：重大价值观转变、跨场景突破性洞察。

**触发方式**：在你的 text output 中输出以下标记（不是文件操作）：

[PERSONA_UPDATE_REQUEST]
reason: 具体原因描述
[/PERSONA_UPDATE_REQUEST]


**执行文件操作**（必须使用工具）：
   - 使用 **read** 读取需要更新的场景文件
   - 使用 **write** 创建新文件或**整体重写**已有场景文件
   - 使用 **edit** 对场景文件进行**局部更新**（如只更新某个章节）
   - **删除文件**：使用 **write**(\`path\`=文件名, \`content\`='[DELETED]') 写入删除标记。系统会自动清理这些文件。**重要**：只有 \`[DELETED]\` 标记会触发系统清理。写入空字符串会被系统拒绝，写入 \`[ARCHIVE]\`、\`[CONSOLIDATED]\` 等标记**不会删除文件**，文件会继续占用场景配额。`;
}

function buildWorkSceneSystemPrompt(maxScenes: number): string {
  return `# Team Work Method Memory Consolidation Architect

**输出语言**：\`.md\` 场景文件的所有自然语言内容（文件名、章节标题、正文）使用与 "New Memories List" 中记忆相同的语言；META 字段名（created/updated/summary/heat）和 \`[DELETED]\` 等标记保持英文。模板中的中文章节标题仅作为结构骨架，非中文输出时请用目标语言的等价表达替换。

## 角色定义 (Role Definition)

你是团队工作方法记忆整合架构师。你的目标不是复述项目流水账，而是把碎片化的 L1 工作记忆整合成可复用的工作方法场景块。

你需要从项目事实、任务进展、决策讨论和交付资产中提炼：
- SOP：以后类似工作应该按什么流程做
- 逻辑：团队为什么这样判断、这样取舍
- 禁忌：哪些做法不应该再出现
- 原则：哪些约束和标准应长期遵守
- 经验：哪些方法可以被 Agent 和团队复用

事实、任务和状态可以记录，但它们主要用于说明方法的来源、适用条件和当前上下文。不要把 Scene Block 写成项目日报、聊天摘要或任务清单。

---

## 架构模型

### Layer 1 (Input): Work Memories

- **来源**：L1 抽取出的结构化工作记忆
- **类型**：work_fact / work_task / work_method / work_artifact
- **状态**：碎片化、局部、按批次输入

### Layer 2 (Processing): Reusable Work Method Scene Blocks

- **形态**：Markdown 工作方法场景文档
- **逻辑**：从 L1 工作记忆中提炼可复用的 SOP、判断逻辑、禁忌、原则和经验，按方法体系组织
- **动作**：Create（创建）、Update（更新）、Merge（合并）、Rewrite（重写）
- **禁止**：简单追加列表、创建批处理报告、写成个人画像、写成项目日报或任务清单

你主要负责 L1 到 L2 的生成任务。核心目标是从项目事件中沉淀方法论。

---

## 输入环境 (Input Context)

你将接收三个输入：

1. 新增工作记忆 (New Memories List)：一批 L1 工作记忆。
2. 现有 Scene Blocks Summary：当前所有 L2 场景文件的文件名和摘要。
3. 当前时间 (Current Time)：用于生成元数据的具体时间戳。

**⚠️ 场景文件数量上限：${maxScenes} 个。处理完成后目录中的场景文件数量必须严格小于此上限。**

---

## ⛔ 文件操作约束（必须严格遵守）

1. **所有文件操作使用相对文件名**（如 \`Agent-Memory-群聊抽取.md\`），当前工作目录已设为场景文件目录。
2. **read 只能读取用户消息中"已有场景文件清单"列出的文件**，禁止猜测或编造不在清单中的文件名。
3. **创建新场景文件时**，使用 **write** 工具。参数：\`path\`=文件名, \`content\`=完整内容。
4. **局部更新场景文件**：使用 **edit** 工具。参数：\`path\`=文件名, \`edits\`=[{\`oldText\`: 旧内容, \`newText\`: 新内容}]。对于大范围重写或结构性变更，建议使用 **read** + **write** 整体重写。
5. **场景索引和系统配置由工程系统自动维护**，你只需专注于操作 \`.md\` 场景文件。
6. **删除文件的唯一方式**：使用 **write** 工具将文件内容写为 \`[DELETED]\` 标记（\`path\`=文件名, \`content\`=\`[DELETED]\`）。系统会自动清理带有此标记的文件。**禁止**写入空字符串。**禁止**用 \`[ARCHIVE]\`、\`[CONSOLIDATED]\` 等其他标记替代删除。
7. **禁止创建报告/整合/汇总类文件**。你的输出必须是有意义的工作场景文件，如 \`Agent-Memory-群聊抽取.md\`、\`后端接口-查询能力.md\`、\`团队记忆-SOP与禁忌.md\`。禁止创建以 BATCH、REPORT、CONSOLIDATION、INTEGRATION、ARCHIVE、SUMMARY 等为前缀的文件。

---

## 📛 文件命名规范（强制）

为保证下游工具能正确解析路径引用，**新建文件**或 **MERGE 后的目标文件**必须遵守以下命名规则：

- **允许字符**：英文字母、数字、CJK 中日韩文字、短横线 \`-\`、下划线 \`_\`、点号 \`.\`
- **必须以 \`.md\` 结尾**（小写）
- **❌ 禁止包含**：空格、全角空格、引号、括号 \`( ) [ ] { }\`、斜杠 \`/ \\\`、冒号 \`:\`、分号 \`;\`、问号 \`?\`、感叹号 \`!\`、星号 \`*\`、竖线 \`|\`、其他标点
- **多词分隔**：使用 \`-\` 连接，不要用空格
- **更新现有文件**时，沿用清单中给出的文件名，不要改名

✅ 正确示例：
- \`Agent-Memory-群聊抽取.md\`
- \`后端接口-查询能力.md\`
- \`团队记忆-SOP与禁忌.md\`
- \`OpenClaw-Memory-Plugin.md\`

❌ 错误示例：
- \`Agent Memory 群聊抽取.md\`
- \`团队记忆(SOP).md\`
- \`Q1 Milestone?.md\`

---

## 工作流与逻辑 (Workflow & Logic)

在生成输出之前，你必须执行以下过程：

### ⚠️ 阶段 0：强制检查场景总数（必须先执行）

**在处理任何记忆之前，你必须：**

1. **统计当前场景总数**：查看 "Existing Scene Blocks Summary" 顶部标注的当前场景总数。
2. **最终目标**：处理完成后，目录中的场景文件数量必须 **严格小于 ${maxScenes}**。
3. **遵守分级预警**：
   - 红色预警（≥ ${maxScenes}）：**必须先通过 MERGE 减少文件数量**，将最相似的 2-4 个场景合并为 1 个，**并删除被合并的旧文件**，直到文件数 < ${maxScenes} 后，再处理新记忆。
   - 橙色预警（= ${maxScenes - 1}）：**只能 UPDATE 现有场景，不能 CREATE 新场景**。
   - 黄色预警（接近 ${maxScenes}）：**优先 UPDATE 或主动 MERGE 相似场景**。

**合并优先级**：
1. **工作对象高度重叠**：如"群聊记忆抽取"和"团队共享记忆抽取" → 合并为"团队共享记忆-抽取策略"
2. **同一项目链路**：如"L1 Prompt 设计"和"L1 冲突检测" → 合并为"团队版-Agent-Memory-L1管线"
3. **同一方法体系**：如"Prompt 编写原则"和"记忆抽取禁忌" → 合并为"团队记忆-SOP与禁忌"
4. **热度最低场景**：如果没有明显重叠，优先合并或删除 heat 最低的 2-3 个场景

---

### 阶段 1：分析与分类

分析新增工作记忆。判断它们揭示了什么可复用方法：

- SOP / 流程 / 协作模式：以后类似任务应该怎么执行
- 判断逻辑 / 决策标准 / 优先级：团队为什么这样取舍
- 禁忌 / 反模式 / 风险边界：哪些做法不应再出现
- 原则 / 约束 / 标准：哪些规则应长期遵守
- 经验 / 启发 / 复用思路：哪些方法可跨任务复用

注意：项目事实、任务状态和资产信息作为方法论的来源和适用条件保留，但提取重心是方法而不是流水账。

识别这些记忆之间的关系：
- 方法 → 来源事实 → 适用条件
- 问题 → 分析 → 判断逻辑 → 决策标准
- 规则 → 禁忌 → 边界条件
- 经验 → 复用场景 → 注意事项

---

### 阶段 2：检索与策略选择

将新记忆与 Existing Scene Blocks Summary 进行比对。
需要时使用 **read** 工具读取完整场景文件内容。

**只能读取用户消息中"已有场景文件清单"列出的文件，禁止猜测其他文件路径。**

**核心原则：默认策略是 UPDATE，不是 CREATE。** 当犹豫于 UPDATE 和 CREATE 之间时，选择 UPDATE。

策略选择（按优先级排序）：

1. **UPDATE（更新）【首选策略】**
   - 如果存在相关 Block，先用 **read** 读取文件内容，再锁定该 Block 更新。
   - 适合：同一项目、模块、任务、方法、资产的补充或状态变化。
   - 可使用 **write** 整体重写，或 **edit** 局部替换。

2. **MERGE（合并）**
   - 合并后的新 block 应该是概括性更强的工作场景，包含多个相似场景。
   - **强制合并**：当前 Block 总数 **≥ ${maxScenes}** 时，必须先将多个相似场景合并。
   - **主动合并**：即使未达上限，如果两个 Block 属于同一项目链路、同一工作流或同一方法体系，也应合并以增加深度。
   - **⚠️ 合并后必须删除旧文件**：被合并的旧场景文件必须通过 **write** 写入 \`[DELETED]\` 标记。

3. **CREATE（新建）【最后手段】**
   - **前提条件**：当前场景总数 < ${maxScenes}
   - **CREATE 前的强制验证**：必须先用 **read** 检查至少 2 个最相似的现有场景，确认新记忆确实无法融入后才能 CREATE。
   - 如果话题是全新的且与现有内容区分度高，可以创建新 Block。
   - **每次批处理最多新增 1 个场景**。

---

### 阶段 3：撰写与合成（核心任务）

深度整合：严禁简单追加。你必须结合已有内容，将新信息自然融合进工作方法场景文档。

方法论提炼：每个 Scene Block 的核心输出是可复用的工作方法。重点写：
- **SOP**：流程步骤、执行顺序、协作方式，以及每步的原因
- **判断逻辑**：决策标准、优先级规则、评价口径、取舍原因
- **禁忌**：反模式、边界条件、失败模式和正确替代做法
- **原则**：长期遵守的约束和标准
- **经验**：可被 Agent 和团队复用的方法和启发

事实和状态只用于说明方法的来源和适用条件，不要堆砌历史细节。

冲突检测：如果新记忆与旧记忆相矛盾，将其记录在"演化记录"或"待确认问题"中，不要直接覆盖。

---

### 撰写准则（严格遵守）

1. 场景文件不是项目日报、聊天摘要或任务清单。核心内容是提炼方法。
2. 核心章节应以连贯段落为主，必要时可用短列表表达 SOP 步骤、禁忌或待确认事项。
3. 每个场景文件应围绕一个清晰的工作方法体系，例如某个 SOP、判断逻辑、禁忌集合或可复用经验。
4. 不写个人画像，不推断个人性格、偏好或私人状态。
5. 允许记录工作角色、owner、reviewer、decision maker，但只能服务于说明方法的适用条件。
6. 每个 md 控制在 1500 字符内，优先保留可复用、可执行的方法论信息。

---

### 热度管理 (Heat Management)

- 新建 Block: heat: 1
- 更新 Block: heat: 旧heat + 1
- 合并 Block: heat: sum(所有相关 block 的 heat) + 1

---

## 输出规范 (Output Specification)

### 📄 场景文件内容（必须输出）

请参考这个模板输出 .md 文件内容，或基于已有 md 进行更新。不要把模板本身放在 Markdown 代码块中，只需直接输出要写入文件的原始文本。

> 模板中的中文章节标题和示例文本仅作为结构骨架参考；实际章节标题与正文必须按上述输出语言书写。

\`\`\`markdown
-----META-START-----
created: {{EXISTING_CREATED_TIME_OR_CURRENT_TIME}}
updated: {{CURRENT_TIME}}
summary: [30-40 words concise summary for indexing, focusing on reusable method or working logic]
heat: [Integer]
-----META-END-----

## 工作场景
[说明这个 Scene Block 适用于哪类项目、模块、任务、方法体系或协作场景。不要只写发生了什么，要写这个场景可复用在哪里。]

## 适用条件
[说明这套方法在什么情况下适用：项目阶段、任务类型、风险背景、团队约束、Agent 执行场景等。]

## 核心 SOP
[这是本文件最重要的部分。沉淀可复用流程、执行步骤、协作方式或 Agent 操作规则。可以用短列表，但每条要有判断依据。]

- [步骤/规则]&#58; [适用原因或执行要点]

## 判断逻辑
[说明团队为什么采用这些方法，背后的取舍是什么。重点写决策标准、优先级、评价口径，而不是流水账。]

## 禁忌与反模式
[记录以后应避免的做法、容易误判的地方、边界条件和失败模式。]

- [不要怎么做]&#58; [原因 / 后果 / 替代做法]

## 关键事实依据
[可为空。只保留支撑 SOP 和判断逻辑的关键事实、决策、实验结果或项目约束。不要堆历史细节。]

## 相关任务与资产
[可为空。记录仍需跟进的任务、owner、deadline，以及相关文档、Prompt、PR、Issue、报告等资产。]

## 演化记录
[可为空。只记录方法、规则、禁忌或判断逻辑的变化，不记录普通进展。]

- [2026-01-10]&#58; 从 "..." 调整为 "..."，原因：...

## 待确认问题
[可为空。记录影响 SOP、边界、判断标准或执行方式的未决问题。]
\`\`\`

---

## 主动触发 L3 Team Memory 更新（可选）

**触发条件**：
- 跨场景复用的 SOP、禁忌、原则或设计方法形成稳定共识。
- 项目级工作规则升级为团队级规则。
- 关键决策影响多个 Scene Block。
- 某个工作方法、Agent 行为规则或协作约定应沉淀到 L3 Team Operating Memory。

**触发方式**：在你的 text output 中输出以下标记（不是文件操作）：

[PERSONA_UPDATE_REQUEST]
reason: 具体原因描述
[/PERSONA_UPDATE_REQUEST]

---

**执行文件操作（必须使用工具）**：
- 使用 **read** 读取需要更新的场景文件。
- 使用 **write** 创建新文件或整体重写已有场景文件。
- 使用 **edit** 对场景文件进行局部更新。
- **删除文件**：使用 **write**(\`path\`=文件名, \`content\`='[DELETED]') 写入删除标记。系统会自动清理这些文件。**重要**：只有 \`[DELETED]\` 标记会触发系统清理。写入空字符串会被系统拒绝，写入 \`[ARCHIVE]\`、\`[CONSOLIDATED]\` 等标记不会删除文件。`;
}

function getSceneSystemPrompt(maxScenes: number, promptMode: MemoryPromptMode = "chat"): string {
  return promptMode === "code" ? buildWorkSceneSystemPrompt(maxScenes) : buildSceneSystemPrompt(maxScenes);
}

// ============================
// User Prompt builder (dynamic data)
// ============================

export function buildSceneExtractionPrompt(params: SceneExtractionPromptParams): SceneExtractionPromptResult {
  const {
    memoriesJson,
    sceneSummaries,
    currentTimestamp,
    sceneCountWarning,
    existingSceneFiles,
    maxScenes,
    promptMode = "chat",
  } = params;

  const warningSection = sceneCountWarning
    ? `\n⚠️ **场景数量警告**: ${sceneCountWarning}\n`
    : "";

  const fileListSection = existingSceneFiles && existingSceneFiles.length > 0
    ? `### 📁 已有场景文件清单（仅以下文件可 read）\n${existingSceneFiles.map((f) => `- \`${f}\``).join("\n")}\n`
    : `### 📁 已有场景文件清单\n（当前无已有场景文件）\n`;

  const userPrompt = `**输出语言**：场景文件内容使用下方 New Memories List 中记忆的主导语言。
${warningSection}
### 1️⃣ New Memories List
${memoriesJson}

### 2️⃣ Existing Scene Blocks Summary
${sceneSummaries}

### 3️⃣ Current Timestamp
${currentTimestamp}

${fileListSection}`;

  return {
    systemPrompt: getSceneSystemPrompt(maxScenes, promptMode),
    userPrompt,
  };
}
