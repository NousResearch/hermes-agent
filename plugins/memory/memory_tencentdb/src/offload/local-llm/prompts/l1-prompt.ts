/**
 * L1 Summarization Prompt — migrated from context-offload-server.
 *
 * Converts tool call/result pairs into high-density JSON summaries.
 */

// ─── System Prompt ───────────────────────────────────────────────────────────

export const L1_SYSTEM_PROMPT = `你是一个专为 AI 编码助手提供支持的"工具结果摘要器"。你的核心任务是深度理解当前的对话上下文，并将繁杂的工具调用与执行结果（一对toolcall和tool result整合成一条summary输出），提炼为高信息密度的 JSON 数组。

在生成摘要前，请务必进行以下内部思考：
1. 任务对齐：结合最近的对话记录，识别用户当前的核心目标和最新意图。若上下文存在冲突，始终以最新的用户意图为准。
2. 价值过滤：忽略工具如何工作的冗余细节，直接提取"发现了什么关键线索"、"做了什么关键动作"、"修改了什么具体内容"或"遇到了什么具体报错"。
3. 影响评估：判断该结果对当前任务的实质性影响（例如：证实了某个假设、推进了哪一步、做出了什么决策，或因为什么报错导致了阻塞）。

【输出格式要求】
你必须且只能输出一个合法的 JSON 对象数组 [{...}]，每个对象**必须**包含以下字段：
- "tool_call": 工具调用的简洁描述。处理规则如下：
  · 如果输入中该 tool pair 标记了 [NEEDS_COMPRESS]，你必须将工具名+关键参数压缩为一句简洁的描述（≤150字符），保留工具名、操作目标（如文件路径、命令意图），省略内联脚本/大段内容的细节。
    示例：exec({"command":"python3 -c 'import csv; ...200行脚本...'"}) → "exec: 运行 Python （xx/xx/xx.sh，标明具体路径和文件）脚本分析 sales_channels.csv 数据质量"
    示例：write_file({"path":"/root/app.py","content":"...5000字符..."}) → "write_file: 写入 /root/app.py (Flask 应用主文件)，大致内容是……"
  · 如果未标记 [NEEDS_COMPRESS]，直接简述工具与参数即可（系统会用原始值覆盖）。
- "summary": 融合上述思考的精炼总结（≤200个字符）。必须一针见血地说清楚结果的业务价值，以及它对任务的推进/阻塞作用。
- "tool_call_id": 原始的 tool_call_id（必须原样透传）。
- "timestamp": 原始的中国标准时间（+08:00）ISO 8601 时间戳（必须原样透传）。
- "score"（**必填**）: 结合信息密度和任务目的分析summary对于原文的可替代性，范围在0-10之间，越接近10表示summary越能替代原文。

【严格规则】
只允许输出纯 JSON 数组，严禁输出思考过程或其他解释性文本。`;

// ─── Constants ───────────────────────────────────────────────────────────────

const PARAMS_MAX_LEN = 500;
const RESULT_MAX_LEN = 2000;
const COMPRESS_THRESHOLD = 200;

// ─── Types ───────────────────────────────────────────────────────────────────

export interface L1ToolPair {
  toolName: string;
  toolCallId: string;
  params: unknown;
  result: unknown;
  timestamp: string;
}

// ─── User Prompt Builder ─────────────────────────────────────────────────────

/**
 * Build the L1 user prompt for summarization.
 * Mirrors context-offload-server/internal/service/prompt/BuildL1UserPrompt.
 */
export function buildL1UserPrompt(recentMessages: string, pairs: L1ToolPair[]): string {
  const parts: string[] = [];

  parts.push("## 最近的对话上下文（用于理解当前任务）：");
  parts.push(recentMessages);
  parts.push("\n## Tool call/result pairs to summarize:");

  for (let i = 0; i < pairs.length; i++) {
    const p = pairs[i];
    const paramsStr = truncate(stringify(p.params), PARAMS_MAX_LEN);
    const resultStr = truncate(stringify(p.result), RESULT_MAX_LEN);
    const canonical = `${p.toolName}(${stringify(p.params)})`;
    const needsCompress = canonical.length > COMPRESS_THRESHOLD;

    parts.push(`--- Tool Pair ${i + 1} ---`);
    parts.push(`tool_call_id: ${p.toolCallId}`);
    parts.push(`timestamp: ${p.timestamp}`);
    if (needsCompress) {
      parts.push(`Tool: ${p.toolName} [NEEDS_COMPRESS]`);
    } else {
      parts.push(`Tool: ${p.toolName}`);
    }
    parts.push(`Params: ${paramsStr}`);
    parts.push(`Result: ${resultStr}\n`);
  }

  parts.push("Summarize each pair into the JSON array format described.");
  return parts.join("\n");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function stringify(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen) + "...";
}
