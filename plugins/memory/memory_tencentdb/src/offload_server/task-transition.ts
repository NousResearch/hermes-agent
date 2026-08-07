/**
 * Task Transition logic — manages MMD file creation/switching based on L1.5 judgment.
 */
import type { StorageAdapter } from "../core/storage/adapter.js";
import type { OffloadState, TaskJudgment, MmdMeta } from "./types.js";

/**
 * Apply task transition based on L1.5 judgment.
 * Mutates state in place; may create new MMD file in COS.
 * @param basePath - Full storage base path, e.g. "offload/default/agent_main_tui-xxx"
 */
export async function handleTaskTransition(
  state: OffloadState,
  judgment: TaskJudgment,
  storage: StorageAdapter,
  basePath: string,
): Promise<void> {
  if (judgment.taskCompleted && judgment.isLongTask && !judgment.isContinuation && judgment.newTaskLabel) {
    // CASE 1: New long task → create new MMD file
    const filename = await generateMmdFilename(storage, basePath, judgment.newTaskLabel);
    await storage.writeFile(`${basePath}/mmds/${filename}`, "");
    state.activeMmdFile = filename;
  } else if (judgment.taskCompleted && judgment.isContinuation && judgment.continuationMmdFile) {
    // CASE 2: Continue historical task → switch to that MMD
    state.activeMmdFile = judgment.continuationMmdFile;
  } else if (judgment.taskCompleted && !judgment.isLongTask) {
    // CASE 3: Short task / casual chat → clear active MMD
    state.activeMmdFile = null;
  } else if (!judgment.taskCompleted && judgment.isLongTask && !state.activeMmdFile) {
    // CASE 5: Task in progress but no active MMD yet → create one
    const label = judgment.newTaskLabel || "current-task";
    const filename = await generateMmdFilename(storage, basePath, label);
    await storage.writeFile(`${basePath}/mmds/${filename}`, "");
    state.activeMmdFile = filename;
  }
  // CASE 4: !taskCompleted + activeMmdFile already set → keep unchanged
}

/**
 * Generate a new MMD filename with auto-incrementing sequence number.
 * Format: "003-refactor-api.mmd"
 */
async function generateMmdFilename(
  storage: StorageAdapter,
  basePath: string,
  label: string,
): Promise<string> {
  const mmdsPrefix = `${basePath}/mmds/`;
  const existingFiles = await storage.readdirNames(mmdsPrefix, ".mmd");

  let maxSeq = 0;
  for (const f of existingFiles) {
    const match = f.match(/^(\d+)-/);
    if (match) {
      const seq = parseInt(match[1], 10);
      if (seq > maxSeq) maxSeq = seq;
    }
  }

  const nextSeq = String(maxSeq + 1).padStart(3, "0");
  const safeLabel = label
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 30);

  return `${nextSeq}-${safeLabel || "task"}.mmd`;
}

/**
 * Extract MmdMeta from MMD file content (parses %%{ ... }%% header).
 */
export function extractMmdMeta(filename: string, content: string): MmdMeta {
  const defaults: MmdMeta = {
    filename,
    taskGoal: "",
    doneCount: 0,
    doingCount: 0,
    todoCount: 0,
    updatedTime: null,
    nodeSummaries: [],
  };

  if (!content) return defaults;

  // Parse %%{ ... }%% metadata line
  const metaMatch = content.match(/%%\{([\s\S]*?)\}%%/);
  if (metaMatch) {
    try {
      const meta = JSON.parse(`{${metaMatch[1]}}`);
      defaults.taskGoal = meta.taskGoal ?? "";
      defaults.updatedTime = meta.updatedTime ?? null;
    } catch {
      // ignore parse errors
    }
  }

  // Count node statuses and extract summaries
  const nodeRegex = /(\w[\w-]*)\["([^"]*?)"\]/g;
  let nodeMatch;
  while ((nodeMatch = nodeRegex.exec(content)) !== null) {
    const nodeId = nodeMatch[1];
    const nodeContent = nodeMatch[2];
    const statusM = nodeContent.match(/status:\s*(done|doing|todo|paused|blocked)/i);
    const summaryM = nodeContent.match(/summary:\s*(.+?)(?:<br\/>|$)/i);
    if (statusM) {
      const s = statusM[1].toLowerCase();
      if (s === "done") defaults.doneCount++;
      else if (s === "doing") defaults.doingCount++;
      else if (s === "todo") defaults.todoCount++;
      defaults.nodeSummaries!.push({
        nodeId,
        status: s,
        summary: summaryM?.[1]?.trim() ?? "",
      });
    }
  }

  // Fallback: if regex didn't catch statuses, try simple pattern
  if (defaults.doneCount === 0 && defaults.doingCount === 0 && defaults.todoCount === 0) {
    const statusMatches = content.matchAll(/status:\s*(done|doing|todo|paused|blocked)/gi);
    for (const m of statusMatches) {
      const s = m[1].toLowerCase();
      if (s === "done") defaults.doneCount++;
      else if (s === "doing") defaults.doingCount++;
      else if (s === "todo") defaults.todoCount++;
    }
  }

  return defaults;
}
