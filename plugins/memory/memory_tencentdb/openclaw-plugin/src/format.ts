/**
 * Format recall results into prompt context.
 *
 * Output structure (mirrors original memory-tencentdb plugin):
 * - prependContext: dynamic L1 memories (changes per turn, injected before user message)
 * - appendSystemContext: stable content (Persona + Scene Nav + tools guide, appended to system prompt)
 */

import type { RecallResult } from "./hooks/recall.js";

interface L1Item {
  id: string;
  content: string;
  type: string;
  score?: number;
}

interface SceneEntry {
  path: string;
  created_at?: string;
  updated_at?: string;
}

// ── Memory Tools Guide ──
const MEMORY_TOOLS_GUIDE = `<memory-tools-guide>
## 记忆工具调用指南

当上方注入的记忆片段不足以回答用户问题时，可主动调用以下工具获取更多信息：

- **tdai_memory_search**：搜索结构化记忆（L1），适用于回忆用户偏好、历史事件、规则等。
- **tdai_conversation_search**：搜索原始对话（L0），适用于查找具体消息原文、时间线、上下文细节。
- **tdai_read_cos**：读取记忆文件详情（使用下方 Scene Navigation 中的完整相对路径，如 scene_blocks/xxx.md；也可读 persona.md）。

### ⚠️ 调用次数限制
每轮对话中，tdai_memory_search 和 tdai_conversation_search **合计最多调用 3 次**。
- 首次搜索无结果时，可换关键词或换工具重试，但总调用次数不要超过 3 次。
- 若 3 次搜索后仍无结果，说明该信息不在记忆中，请直接根据已有信息回复用户。
</memory-tools-guide>`;

/**
 * Format L1 memories as prependContext.
 */
function formatL1Memories(items: L1Item[]): string | undefined {
  if (items.length === 0) return undefined;

  const lines: string[] = [
    "<relevant-memories>",
    "",
  ];

  for (const item of items) {
    const typeTag = item.type ? `[${item.type}]` : "";
    lines.push(`- ${typeTag} ${item.content}`);
  }

  lines.push("");
  lines.push("</relevant-memories>");

  return lines.join("\n");
}

/**
 * Format stable system context: Persona + Scene Navigation + Tools Guide.
 */
function formatSystemContext(
  persona: string | null,
  scenes: SceneEntry[],
): string | undefined {
  const parts: string[] = [];

  // Persona (L3)
  if (persona) {
    parts.push("<user-persona>");
    parts.push(persona);
    parts.push("</user-persona>");
  }

  // Scene Navigation (L2 index) — only if not already in persona
  if (scenes.length > 0 && (!persona || !persona.includes("Scene Navigation"))) {
    parts.push("");
    parts.push("## 🗺️ Scene Navigation");
    parts.push("*以下是当前场景记忆索引，可使用 tdai_read_cos 读取详细内容。*");
    parts.push("");
    for (const scene of scenes) {
      parts.push(`- \`${scene.path}\``);
    }
  }

  // Tools guide (always append)
  parts.push("");
  parts.push(MEMORY_TOOLS_GUIDE);

  const result = parts.join("\n").trim();
  return result || undefined;
}

/**
 * Main format function: produce RecallResult for prompt injection.
 */
export function formatRecallResult(
  l1Items: L1Item[],
  persona: string | null,
  scenes: SceneEntry[],
): RecallResult {
  return {
    prependContext: formatL1Memories(l1Items),
    appendSystemContext: formatSystemContext(persona, scenes),
  };
}
