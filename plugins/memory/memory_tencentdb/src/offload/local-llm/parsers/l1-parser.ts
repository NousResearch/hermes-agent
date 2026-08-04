/**
 * L1 Response Parser — extracts summarization results from LLM output.
 */
import { extractJson } from "./json-utils.js";
import type { OffloadEntry } from "../../types.js";

interface RawL1Entry {
  tool_call?: string;
  summary?: string;
  tool_call_id?: string;
  timestamp?: string;
  score?: number;
}

/**
 * Parse L1 LLM response into OffloadEntry array.
 * Tolerant of markdown wrapping, missing fields, etc.
 */
export function parseL1Response(raw: string): OffloadEntry[] {
  const parsed = extractJson<RawL1Entry[]>(raw);
  if (!parsed || !Array.isArray(parsed)) return [];

  const entries: OffloadEntry[] = [];
  for (const item of parsed) {
    if (!item || typeof item !== "object") continue;

    const toolCallId = item.tool_call_id ?? "";
    if (!toolCallId) continue; // tool_call_id is required

    entries.push({
      tool_call_id: toolCallId,
      tool_call: item.tool_call ?? "",
      summary: item.summary ?? "",
      timestamp: item.timestamp ?? "",
      score: typeof item.score === "number" ? item.score : 5,
      node_id: null,
    });
  }

  return entries;
}
