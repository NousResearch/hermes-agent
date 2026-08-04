/**
 * L1.5 Response Parser — extracts task judgment from LLM output.
 */
import { extractJson } from "./json-utils.js";
import type { TaskJudgment } from "../../types.js";

interface RawL15Response {
  taskCompleted?: boolean | null;
  isContinuation?: boolean | null;
  isLongTask?: boolean | null;
  continuationMmdFile?: string | null;
  newTaskLabel?: string | null;
}

/**
 * Parse L1.5 LLM response into TaskJudgment.
 * Returns null if the response is completely unparseable or all-null (backend unavailable).
 */
export function parseL15Response(raw: string): TaskJudgment | null {
  const parsed = extractJson<RawL15Response>(raw);
  if (!parsed || typeof parsed !== "object") return null;

  // All-null check (mirrors normalizeJudgment logic)
  if (parsed.taskCompleted == null && parsed.isContinuation == null && parsed.isLongTask == null) {
    return null;
  }

  return {
    taskCompleted: Boolean(parsed.taskCompleted),
    isContinuation: Boolean(parsed.isContinuation),
    isLongTask: Boolean(parsed.isLongTask),
    continuationMmdFile:
      typeof parsed.continuationMmdFile === "string" ? parsed.continuationMmdFile : undefined,
    newTaskLabel:
      typeof parsed.newTaskLabel === "string" ? parsed.newTaskLabel : undefined,
  };
}
