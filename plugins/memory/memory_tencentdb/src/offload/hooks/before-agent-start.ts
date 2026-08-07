/**
 * before_agent_start hook handler.
 * Implements L1.5: Task completion judgment and active MMD management.
 *
 * Backend-only mode: local LLM judge has been removed.
 * Only normalizeJudgment and handleTaskTransition are exported for use by index.ts.
 */
import { readMmd, writeMmd, deleteMmd, type StorageContext } from "../storage.js";
import type { OffloadStateManager } from "../state-manager.js";
import type { PluginLogger, TaskJudgment } from "../types.js";

/**
 * Normalize a raw L1.5 judgment response (from backend)
 * into a safe TaskJudgment with guaranteed boolean fields.
 * Handles null/undefined values from backend fallback responses.
 */
export function normalizeJudgment(raw: Record<string, unknown>): TaskJudgment | null {
  // All-null response from backend means "LLM unavailable" — treat as no judgment
  if (raw.taskCompleted == null && raw.isContinuation == null && raw.isLongTask == null) {
    return null;
  }
  return {
    taskCompleted: Boolean(raw.taskCompleted),
    isContinuation: Boolean(raw.isContinuation),
    continuationMmdFile:
      typeof raw.continuationMmdFile === "string" ? raw.continuationMmdFile : undefined,
    newTaskLabel:
      typeof raw.newTaskLabel === "string" ? raw.newTaskLabel : undefined,
    isLongTask: Boolean(raw.isLongTask),
  };
}

export async function handleTaskTransition(
  stateManager: OffloadStateManager,
  judgment: TaskJudgment,
  logger: PluginLogger,
): Promise<void> {
  const currentMmd = stateManager.getActiveMmdFile();

  const ctx = stateManager.ctx;

  const isEmptyShellMmd = async (filename: string | null): Promise<boolean> => {
    if (!filename) return false;
    try {
      const content = await readMmd(ctx, filename);
      if (!content) return false;
      const trimmed = content.trim();
      if (trimmed.includes("%%{")) return false;
      const lines = trimmed.split("\n").filter((l) => l.trim().length > 0);
      return lines.length <= 3;
    } catch {
      return false;
    }
  };

  const cleanupIfEmptyShell = async (oldFilename: string | null) => {
    if (!oldFilename) return;
    const isShell = await isEmptyShellMmd(oldFilename);
    if (isShell) {
      try {
        await deleteMmd(ctx, oldFilename);
      } catch {
        /* ignore */
      }
    }
  };

  const createNewMmd = async (label: string) => {
    const num = await stateManager.nextMmdNumber();
    const paddedNum = String(num).padStart(3, "0");
    const filename = `${paddedNum}-${label}.mmd`;
    logger.debug?.(`[context-offload] L1.5: Creating new MMD: ${filename} (replacing ${currentMmd ?? "(none)"})`);
    await cleanupIfEmptyShell(currentMmd);
    stateManager.setActiveMmd(filename, label);
    const initialMmd = `flowchart TD\n    ${paddedNum}-N1["${label}"]\n`;
    await writeMmd(ctx, filename, initialMmd);
    logger.debug?.(`[context-offload] L1.5: New MMD created and activated: ${filename}`);
  };

  const reactivateMmd = async (contFile: string) => {
    logger.debug?.(`[context-offload] L1.5: Reactivating MMD: ${contFile} (current=${currentMmd ?? "(none)"})`);
    if (currentMmd && currentMmd !== contFile) {
      await cleanupIfEmptyShell(currentMmd);
    }
    const mmdId = contFile.replace(/^\d+-/, "").replace(/\.mmd$/, "");
    stateManager.setActiveMmd(contFile, mmdId);
    const existing = await readMmd(ctx, contFile);
    if (existing === null) {
      const prefixMatch = contFile.match(/^(\d+)-/);
      const prefix = prefixMatch ? prefixMatch[1] : "000";
      const initialMmd = `flowchart TD\n    ${prefix}-N1["${mmdId}"]\n`;
      await writeMmd(ctx, contFile, initialMmd);
      logger.warn(`[context-offload] L1.5: Reactivated MMD file was missing, wrote initial template: ${contFile}`);
    }
  };

  if (judgment.taskCompleted) {
    logger.debug?.(`[context-offload] L1.5: Task COMPLETED — continuation=${judgment.isContinuation}, longTask=${judgment.isLongTask}, contFile=${judgment.continuationMmdFile ?? "N/A"}, newLabel=${judgment.newTaskLabel ?? "N/A"}`);
    if (judgment.isContinuation && judgment.continuationMmdFile) {
      await reactivateMmd(judgment.continuationMmdFile);
    } else if (judgment.isLongTask && judgment.newTaskLabel) {
      const currentLabel = currentMmd
        ? currentMmd.replace(/^\d+-/, "").replace(/\.mmd$/, "")
        : null;
      if (currentLabel !== judgment.newTaskLabel) {
        await createNewMmd(judgment.newTaskLabel);
      }
    } else if (judgment.isContinuation && !judgment.continuationMmdFile) {
      if (!currentMmd) {
        stateManager.setActiveMmd(null, null);
      }
    } else {
      logger.debug?.("[context-offload] L1.5: No MMD needed (casual/short), clearing active MMD");
      stateManager.setActiveMmd(null, null);
    }
  } else {
    logger.debug?.(`[context-offload] L1.5: Task NOT completed — continuation=${judgment.isContinuation}, longTask=${judgment.isLongTask}, current=${currentMmd ?? "(none)"}`);
    if (judgment.isContinuation) {
      if (!currentMmd && judgment.continuationMmdFile) {
        await reactivateMmd(judgment.continuationMmdFile);
      }
    } else if (judgment.isLongTask && judgment.newTaskLabel) {
      const currentLabel = currentMmd
        ? currentMmd.replace(/^\d+-/, "").replace(/\.mmd$/, "")
        : null;
      if (currentLabel !== judgment.newTaskLabel) {
        await createNewMmd(judgment.newTaskLabel);
      }
    }
  }
}
