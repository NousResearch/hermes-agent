/**
 * L1.5 Response Parser — extracts TaskJudgment from LLM output.
 */
import { extractJson } from "./json-utils.js";
import type { TaskJudgment } from "../types.js";

interface RawL15Response {
  taskCompleted?: boolean | null;
  isContinuation?: boolean | null;
  isLongTask?: boolean | null;
  continuationMmdFile?: string | null;
  newTaskLabel?: string | null;
}

/**
 * Parse L1.5 LLM response into TaskJudgment.
 * Returns null if completely unparseable or all-null (LLM unavailable).
 */
export function parseL15Response(raw: string): TaskJudgment | null {
  const parsed = extractJson<RawL15Response>(raw);
  if (!parsed || typeof parsed !== "object") return null;

  if (
    parsed.taskCompleted == null &&
    parsed.isContinuation == null &&
    parsed.isLongTask == null
  ) {
    return null;
  }

  return {
    taskCompleted: toBool(parsed.taskCompleted),
    isContinuation: toBool(parsed.isContinuation),
    isLongTask: toBool(parsed.isLongTask),
    continuationMmdFile:
      typeof parsed.continuationMmdFile === "string" && isSafeFilename(parsed.continuationMmdFile)
        ? parsed.continuationMmdFile
        : undefined,
    newTaskLabel:
      typeof parsed.newTaskLabel === "string" ? parsed.newTaskLabel : undefined,
  };
}

/** Safely coerce LLM value to boolean, handling string "false"/"0". */
function toBool(value: unknown): boolean {
  if (typeof value === "string") {
    return value.toLowerCase() !== "false" && value !== "0" && value !== "";
  }
  return Boolean(value);
}

/** Validate that a filename is safe (no path traversal or special chars). */
function isSafeFilename(name: string): boolean {
  if (!name) return false;
  if (name.includes("/") || name.includes("\\") || name.includes("..")) return false;
  return /^[a-zA-Z0-9_.\-]+$/.test(name);
}
