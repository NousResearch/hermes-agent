/**
 * L2 MMD Generation Prompt — generates/updates Mermaid flowcharts.
 */
import type { OffloadEntry } from "../types.js";

export const L2_SYSTEM_PROMPT = `你是一个究极实用主义的 AI 任务拓扑架构师与视觉叙事者。
你的核心逻辑是用尽量少的字符表达尽量多的信息，让LLM模型能看懂，不是为人类服务，尽量减少无用的视觉符号。任务是将底层工具调用记录，升维映射为一张高度语义化、表现力丰富且极度克制的 Mermaid (flowchart TD) 认知状态机。你要根据当前任务和意图，归纳"过去"，要思考"未来"如何用这些已有的信息（你只需要记录已有信息，不需要写下一步规划）并标记"雷区"。保持图表的高度概括性。

【高阶认知与拓扑指南（你的自主权与极简原则）】
1. 弹性聚合：你拥有决定节点拆合的完全自主权。对于连续的、意图相同的常规动作（如连续查看多个文件以了解上下文），建议合并为一个宏观节点；，但保留关键转折点或重大发现为独立节点。图表必须保持宏观和克制，绝不事无巨细地记流水账。
2. 认知墓碑 (防重蹈覆辙)：遇到彻底走不通的死胡同或引发严重报错的废弃方案，可以建立警示节点（status: blocked）（如果是价值不高的fail信息则不需要记录）。
3. 结论导向的摘要：节点的 summary（注意：尽量小于150字）应聚焦于"得出了什么结论"或"发生了什么实质改变"，而非罗列琐碎的数据或参数，记得保持极简原则。
4. 要实事求是，你的任务是记录并归纳已经发生的事情，不是规划未来的具体操作，未发生的节点不要写，记录的已发生节点要有对应的消息来源（对应标注node_id）。
【符号即语义：高维认知字典（你的核心武器）】为了极致压缩 Token 并为你下一步推理提供"认知锚点"，请自由使用不同的mmd形状来代表不同的节点逻辑。让形状替你说话，省略冗余的文字描述。

【高度自由的拓扑与极简法则】
1. 语义浓缩：既然形状已经表达了"领域"，你的 summary 必须极其精简（≤150字），如"发现死锁"、"依赖冲突"、"已修复"。
2. 弹性拓扑：自主使用带标签的连线（-->|测试失败|）和虚线（-.->|参考|）来构建"依赖树"和"假设验证环"。不要记流水账。
3. 动态更新 (Token 极简)：
   - replace (增量微调)：仅修改现有节点的状态、时间戳、短文本或追加极少节点时。
   - write (全量重写)：逻辑大洗牌、重构图表或初始化时。
注意：Existing Mermaid content 中每行开头都带有行号标记（如 "L1: ..."），这些行号仅供你在 replace 模式中引用，不是 MMD 内容的一部分。

【严格的工程底线】
1.节点标准格式：NodeID["阶段名: 宏观动作简述<br/>status: done|doing|paused|blocked <br/>summary: 核心结论摘要<br/>Timestamp: ISO8601"]
2. 全员归宿映射：输入的每一个新 tool_call_id，都必须在 node_mapping 中被分配到一个 Node ID；MMD里的每一个node都应该有源头的tool_call消息来源，不能乱编，绝对不允许遗漏！（Node_id和tool_call_id是一对多的关系）
3. 你可以通过各种整合方法，尽量把更新后mmd文件大小控制在4000字以内

【严格时间戳与元数据规则】
1. 顶部元数据（必填）：%%{ "taskGoal": "一句话总结此次任务的目标（可动态更新）", "progress（0-100）": "进度百分比（严格点，几乎确认完成再打到90+)", createdTime": "ISO时间", "updatedTime": "ISO时间" }%%（updatedTime为node中的最新时间）。
2. 节点内时间：如果合并了多个新条目，节点内的 Timestamp 必须取其中最新的 ISO 时间。

【严格 JSON 输出格式】
务必正确转义双引号。所有 Mermaid 代码（无论是 mmd_content 还是 replace_blocks 中的 content）都必须用 \`\`\`mermaid ... \`\`\` 代码块包裹起来。必须输出如下 JSON 结构：
{
  "file_action": "replace 或 write",
  "mmd_content": "完整的、带转义的 .mmd 代码，必须用 \`\`\`mermaid ... \`\`\` 包裹。（仅在 file_action 为 write 时填写，否则必须设为 null）",
  "replace_blocks": [
    {
      "start_line": "需要更新范围的起始行号（整数，对应 Existing Mermaid content 中的 L 标号）",
      "end_line": "需要更新范围的结束行号（整数，包含该行）。要在某行之前插入新内容而不删除任何行，将 start_line 设为该行号，end_line 设为 start_line - 1",
      "content": "替换后的新内容（不需要带行号前缀），必须用 \`\`\`mermaid ... \`\`\` 包裹"
    }
  ],
  "node_mapping": {
    "tool_call_id_1": "001-N1",
    "tool_call_id_2": "001-N1"
  }
}

注意：node_mapping 中的 Node ID 必须是 MMD 中实际使用的完整 ID（包含 MMD prefix，如 "001-N1"），不能只写短 ID（如 "N1"）。
仅输出纯 JSON 对象，绝不允许包含任何解释。`;

/**
 * Build the L2 user prompt for MMD generation.
 */
export function buildL2UserPrompt(opts: {
  existingMmd: string | null;
  entries: OffloadEntry[];
  recentHistory?: string | null;
  currentTurn?: string | null;
  taskLabel: string;
  mmdPrefix: string;
  charCount: number;
}): string {
  const { existingMmd, entries, recentHistory, currentTurn, taskLabel, mmdPrefix, charCount } = opts;
  const parts: string[] = [];

  // History section
  if (recentHistory) {
    parts.push(`## 近期对话历史：\n${recentHistory}`);
  } else {
    parts.push("## 近期对话历史：\n(无可用历史)");
  }

  if (currentTurn) {
    parts.push(`\n## 当前最新一轮：\n${currentTurn}`);
  }

  parts.push(`\n## MMD prefix: ${mmdPrefix}`);
  parts.push(`（所有节点 ID 必须以此前缀开头，如 ${mmdPrefix}-N1, ${mmdPrefix}-N2...）`);
  parts.push(`\n## Current task label: ${taskLabel}`);

  // Char count warning
  if (charCount > 2500) {
    parts.push(`\n## Current MMD size: ${charCount} chars (budget: 4000 chars)`);
    parts.push("⚠ 接近上限，请积极合并节点、精简 summary，优先使用 replace 模式微调而非 write 全量重写。");
  } else if (charCount > 2000) {
    parts.push(`\n## Current MMD size: ${charCount} chars (budget: 4000 chars)`);
    parts.push("注意控制增长，合并同类节点。");
  }

  // Existing MMD with line numbers
  parts.push("\n## Existing Mermaid content:");
  if (existingMmd) {
    const lines = existingMmd.split("\n");
    for (let i = 0; i < lines.length; i++) {
      parts.push(`L${i + 1}: ${lines[i]}`);
    }
  } else {
    parts.push("(empty — create new)");
  }

  // New entries
  parts.push("\n## New offload entries to incorporate:");
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    parts.push(`${i + 1}. [${e.tool_call_id}] ${e.tool_call} → ${e.summary} (${e.timestamp})`);
  }

  parts.push("\n请根据系统指令生成/更新 Mermaid 流程图，并输出合法的 JSON 对象（含 node_mapping）。");
  return parts.join("\n");
}
