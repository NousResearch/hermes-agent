/**
 * SceneExtractor: LLM-driven memory extraction into scene blocks.
 *
 * Replaces the keyword-based SceneManager.processNewMemories() with an
 * LLM agent that autonomously reads/writes scene block files using tools.
 *
 * Security: The LLM is sandboxed — workspaceDir is set to scene_blocks/
 * so it can ONLY operate on .md scene files. System files (checkpoint,
 * scene_index, persona.md) are physically invisible to the LLM.
 *
 * Flow:
 *   1. Backup + load scene index + build summaries
 *   2. Assemble extraction prompt with memories + scene context
 *   3. Run via CleanContextRunner (tools enabled, sandboxed to scene_blocks/)
 *   4. Cleanup: remove soft-deletes, sync index, update navigation
 *   5. Parse LLM text output for out-of-band persona update signals
 */

import { CleanContextRunner } from "../../utils/clean-context-runner.js";
import { CheckpointManager } from "../../utils/checkpoint.js";
import { BackupManager } from "../../utils/backup.js";
import { readSceneIndex, syncSceneIndex } from "../scene/scene-index.js";
import type { SceneIndexEntry } from "../scene/scene-index.js";
import { parseSceneBlock } from "../scene/scene-format.js";
import { generateSceneNavigation, stripSceneNavigation } from "../scene/scene-navigation.js";
import { normalizeSceneFilenames } from "./filename-normalizer.js";
import type { MemoryPromptMode } from "../../config.js";
import { buildSceneExtractionPrompt } from "../prompts/scene-extraction.js";
import { report } from "../report/reporter.js";
import { reportL2LatencyMetrics } from "../report/metric-tracking-l2-latency.js";
import type { LLMRunner, Logger, TraceContext } from "../types.js";
import { buildTraceParams } from "../types.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";

const TAG = "[memory-tdai] [extractor]";

type ExtractorLogger = Logger;

export interface ExtractionResult {
  memoriesProcessed: number;
  success: boolean;
  error?: string;
  /** True if LLM ran but produced no file changes (scene_index unchanged). */
  emptyExtraction?: boolean;
}

export interface SceneExtractorOptions {
  dataDir: string;
  config: unknown;
  model?: string;
  maxScenes?: number;
  /** Prompt family for L2 scene extraction (default: chat). */
  promptMode?: MemoryPromptMode;
  sceneBackupCount?: number;
  timeoutMs?: number;
  logger?: ExtractorLogger;
  /** Plugin instance ID for metric reporting (optional) */
  instanceId?: string;
  /**
   * Host-neutral LLM runner. When provided, used instead of creating
   * a CleanContextRunner (decouples from OpenClaw runtime).
   * Must be configured with `enableTools: true`.
   */
  llmRunner?: LLMRunner;
  /** StorageAdapter for file operations (COS/local). Falls back to fs when absent. */
  storage?: StorageAdapter;
  /** langfuse 上报身份四元组（team/user/agent/session）。传下来会填充 trace 顶级字段。 */
  traceContext?: TraceContext;
}

/**
 * Parse LLM text output for a persona update request signal.
 *
 * Supports multiple formats for robustness:
 * - Block: [PERSONA_UPDATE_REQUEST]reason: xxx[/PERSONA_UPDATE_REQUEST]
 * - Inline: PERSONA_UPDATE_REQUEST: xxx
 */
export function parsePersonaUpdateSignal(text: string): { reason: string } | null {
  // Block format: [PERSONA_UPDATE_REQUEST]...[/PERSONA_UPDATE_REQUEST]
  const blockMatch = text.match(
    /\[PERSONA_UPDATE_REQUEST\]\s*(?:reason:\s*)?(.+?)\s*\[\/PERSONA_UPDATE_REQUEST\]/s,
  );
  if (blockMatch) return { reason: blockMatch[1]!.trim() };

  // Inline format: PERSONA_UPDATE_REQUEST: reason text
  const inlineMatch = text.match(
    /PERSONA_UPDATE_REQUEST:\s*(.+?)(?:\n|$)/,
  );
  if (inlineMatch) return { reason: inlineMatch[1]!.trim() };

  return null;
}

export class SceneExtractor {
  private dataDir: string;
  private runner: LLMRunner;
  private maxScenes: number;
  private sceneBackupCount: number;
  private promptMode: MemoryPromptMode;
  private timeoutMs: number;
  private logger: ExtractorLogger | undefined;
  private instanceId: string | undefined;
  private storage: StorageAdapter | undefined;
  private traceContext: TraceContext | undefined;

  constructor(opts: SceneExtractorOptions) {
    this.dataDir = opts.dataDir;
    this.maxScenes = opts.maxScenes ?? 15;
    this.promptMode = opts.promptMode ?? "chat";
    this.sceneBackupCount = opts.sceneBackupCount ?? 10;
    this.timeoutMs = opts.timeoutMs ?? 300_000; // 5 min — LLM may do multiple tool calls
    this.logger = opts.logger;
    this.instanceId = opts.instanceId;
    this.storage = opts.storage;
    this.traceContext = opts.traceContext;

    // Use injected LLMRunner if available, otherwise fall back to CleanContextRunner
    this.runner = opts.llmRunner ?? new CleanContextRunner({
      config: opts.config,
      modelRef: opts.model,
      enableTools: true,
      logger: opts.logger,
    });

    this.logger?.debug?.(`${TAG} Created: dataDir=${opts.dataDir}, model=${opts.model ?? "(default)"}, promptMode=${this.promptMode}, maxScenes=${this.maxScenes}, timeout=${this.timeoutMs}ms`);
  }

  /**
   * Extract a batch of memories into scene blocks using the LLM agent.
   *
   * @param memories - Array of raw memory records from the API
   * @returns Extraction result with count and success flag
   */
  async extract(memories: Array<{ content: string; created_at: string; id?: string }>): Promise<ExtractionResult> {
    const extractStartMs = Date.now();
    this.logger?.info(`${TAG} extract() start: ${memories.length} memories`);

    if (memories.length === 0) {
      this.logger?.debug?.(`${TAG} extract() skipped: no memories`);
      return { memoriesProcessed: 0, success: true };
    }

    const sceneBlocksDir = this.storage ? StoragePaths.sceneBlocksDir : await (async () => { const path = await import("node:path"); return path.default.join(this.dataDir, "scene_blocks"); })();
    const metadataDir = this.storage ? StoragePaths.metadataDir : await (async () => { const path = await import("node:path"); return path.default.join(this.dataDir, ".metadata"); })();

    // Ensure directories exist
    if (this.storage) {
      await this.storage.mkdir(StoragePaths.sceneBlocksDir);
      await this.storage.mkdir(StoragePaths.metadataDir);
    } else {
      const fs = await import("node:fs/promises");
      await fs.default.mkdir(sceneBlocksDir, { recursive: true });
      await fs.default.mkdir(metadataDir, { recursive: true });
    }

    // Phase 1: Backup
    const backupStartMs = Date.now();
    const cpManager = new CheckpointManager(this.dataDir, this.logger, this.storage);
    const cp = await cpManager.read();
    // BackupManager is only meaningful for local-fs deployments. In service
    // mode (StorageAdapter / COS) snapshot/restore is the storage backend's
    // responsibility, so `bm` stays undefined and the catch-block restore
    // below short-circuits to a no-op.
    let bm: BackupManager | undefined;
    if (!this.storage) {
      const path = await import("node:path");
      bm = new BackupManager(path.default.join(this.dataDir, ".backup"));
      await bm.backupDirectory(sceneBlocksDir, "scene_blocks", `offset${cp.total_processed}`, this.sceneBackupCount);
    }
    this.logger?.debug?.(`${TAG} extract() backup phase: ${Date.now() - backupStartMs}ms`);

    // Phase 2: Load scene index
    const indexStartMs = Date.now();
    const index = await readSceneIndex(this.dataDir, this.storage);
    this.logger?.debug?.(`${TAG} extract() scene index loaded: ${index.length} entries (${Date.now() - indexStartMs}ms)`);

    // Build scene summaries for the prompt (relative filenames only)
    const { summaries: sceneSummaries, filenames: existingSceneFiles } =
      this.buildSceneSummaries(index);

    // Build scene count warning (tiered system)
    let sceneCountWarning: string | undefined;
    const sceneCount = index.length;
    if (sceneCount >= this.maxScenes) {
      sceneCountWarning = `当前场景数量为 **${sceneCount} 个**，已达到或超过 ${this.maxScenes} 个上限！\n**你必须先执行 MERGE 操作**，将最相似的 2-4 个场景合并为 1 个，然后再处理新记忆。\n参考合并对象：热度最低或主题高度重叠的场景。`;
      this.logger?.warn(`${TAG} extract() scene count at limit: ${sceneCount}/${this.maxScenes}`);
    } else if (sceneCount === this.maxScenes - 1) {
      sceneCountWarning = `当前场景数量为 **${sceneCount} 个**，距离上限只差 1 个！\n本次处理**只能 UPDATE 现有场景，不能 CREATE 新场景**。`;
      this.logger?.warn(`${TAG} extract() scene count near limit (CREATE blocked): ${sceneCount}/${this.maxScenes}`);
    } else if (sceneCount >= this.maxScenes - 3) {
      sceneCountWarning = `当前场景数量为 **${sceneCount} 个**，建议优先考虑 UPDATE 或主动 MERGE 相似场景。`;
      this.logger?.debug?.(`${TAG} extract() scene count approaching limit: ${sceneCount}/${this.maxScenes}`);
    }

    // Snapshot scene index + content before LLM — used later to diff created/updated/deleted
    const preExtractIndex = new Map(index.map((e) => [e.filename, e.summary]));
    // Also snapshot scene content so we can detect content-only changes vs metadata-only changes
    const preExtractContent = new Map<string, string>();
    for (const e of index) {
      try {
        let raw: string | null;
        if (this.storage) {
          raw = await this.storage.readFile(`${StoragePaths.sceneBlocksDir}${e.filename}`);
        } else {
          const fs = await import("node:fs/promises");
          const path = await import("node:path");
          raw = await fs.default.readFile(path.default.join(this.dataDir, "scene_blocks", e.filename), "utf-8");
        }
        if (!raw) continue;
        const block = parseSceneBlock(raw, e.filename);
        preExtractContent.set(e.filename, block.content);
      } catch { /* non-fatal */ }
    }

    // Phase 3: Build prompt
    const promptStartMs = Date.now();
    const memoriesJson = JSON.stringify(
      memories.map((m) => ({
        content: m.content,
        created_at: m.created_at,
        id: m.id ?? "",
      })),
      null,
      2,
    );

    const currentTimestamp = formatTimestamp(new Date());

    const { systemPrompt, userPrompt } = buildSceneExtractionPrompt({
      memoriesJson,
      sceneSummaries: sceneSummaries || "(无已有场景)",
      currentTimestamp,
      sceneCountWarning,
      existingSceneFiles,
      maxScenes: this.maxScenes,
      promptMode: this.promptMode,
    });
    this.logger?.debug?.(`${TAG} extract() prompt built: ${userPrompt.length} chars (${Date.now() - promptStartMs}ms)`);

    // Phase 4: Run LLM agent (sandboxed to scene_blocks/)
    let llmOutput = "";
    let llmDurationMs = 0;
    try {
      this.logger?.debug?.(`${TAG} extract() starting LLM runner (timeout=${this.timeoutMs}ms, maxTokens=model default)...`);
      const runnerStartMs = Date.now();
      // langfuse trace 语义：L2 场景抽取有独立 name / 顶级 user/session 列 / 可筛选 tags。
      const traceParams = buildTraceParams("memory.scene-extract", this.traceContext);
      llmOutput = await this.runner.run({
        systemPrompt,
        prompt: userPrompt,
        taskId: `scene-extract-${Date.now()}`,
        timeoutMs: this.timeoutMs,
        // maxTokens omitted → core uses the resolved model's maxTokens from catalog
        workspaceDir: sceneBlocksDir,
        // Service mode: LLM tools read/write via StorageAdapter (COS) instead of local FS
        storage: this.storage,
        storagePrefix: this.storage ? StoragePaths.sceneBlocksDir : undefined,
        ...traceParams,
      }) ?? "";
      llmDurationMs = Date.now() - runnerStartMs;
      this.logger?.debug?.(`${TAG} extract() LLM runner completed: ${llmDurationMs}ms`);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      const totalMs = Date.now() - extractStartMs;
      this.logger?.error(`${TAG} extract() LLM runner failed after ${totalMs}ms: ${errMsg}`);

      // Restore scene_blocks/ from the Phase 1 backup so partial LLM writes
      // (or a wiped sandbox) don't leak into the next recall cycle.
      // Fail-soft: a restore failure must never mask the original LLM error.
      // In service mode (`this.storage` set) we never took a local backup,
      // so `bm` is undefined and we skip restore — the storage backend
      // owns its own snapshot/restore semantics.
      if (bm) {
        try {
          const result = await bm.restoreLatestDirectory("scene_blocks", sceneBlocksDir);
          if (result.restored) {
            this.logger?.warn(`${TAG} extract() restored scene_blocks/ from backup: ${result.from}`);
          } else {
            this.logger?.debug?.(`${TAG} extract() no scene_blocks backup to restore from (first run or empty)`);
          }
        } catch (restoreErr) {
          const rMsg = restoreErr instanceof Error ? restoreErr.message : String(restoreErr);
          this.logger?.warn(`${TAG} extract() restore failed (non-fatal, original LLM error preserved): ${rMsg}`);
        }
      }

      return { memoriesProcessed: 0, success: false, error: errMsg };
    }

    // Phase 5: Subsequent processing — safe cleanup of soft-deleted files
    //
    // Security: The LLM has no `exec` tool and cannot run shell commands.
    // Instead, it "deletes" files by writing the marker `[DELETED]` to the file
    // (writing empty/whitespace-only content is rejected by core's write tool
    // parameter validation). Here we detect and remove those soft-deleted files
    // before syncing the index, so syncSceneIndex won't re-index stale entries.
    //
    // We also detect "META-only" files — files that contain only a META header
    // (e.g. [ARCHIVE] or [CONSOLIDATED] markers) but no actual scene content.
    // These are artifacts of LLM merges that didn't properly delete old files.
    const cleanupStartMs = Date.now();
    let cleanedCount = 0;
    try {
      let allFiles: string[];
      if (this.storage) {
        allFiles = await this.storage.readdirNames(StoragePaths.sceneBlocksDir, ".md");
      } else {
        const fs = await import("node:fs/promises");
        allFiles = (await fs.default.readdir(sceneBlocksDir)).filter((f: string) => f.endsWith(".md"));
      }
      for (const file of allFiles) {
        let raw: string | null;
        if (this.storage) {
          raw = await this.storage.readFile(`${StoragePaths.sceneBlocksDir}${file}`);
        } else {
          const fs = await import("node:fs/promises");
          const path = await import("node:path");
          raw = await fs.default.readFile(path.default.join(this.dataDir, "scene_blocks", file), "utf-8");
        }
        if (!raw || raw.trim().length === 0 || raw.trim() === "[DELETED]") {
          // Empty file or [DELETED] marker — soft-delete
          if (this.storage) {
            await this.storage.unlink(`${StoragePaths.sceneBlocksDir}${file}`);
          } else {
            const fs = await import("node:fs/promises");
            const path = await import("node:path");
            await fs.default.unlink(path.default.join(this.dataDir, "scene_blocks", file));
          }
          cleanedCount++;
          this.logger?.debug?.(`${TAG} extract() removed soft-deleted file: ${file}`);
        } else {
          // Check if file has only META header but no actual content
          const block = parseSceneBlock(raw, file);
          if (!block.content || block.content.trim().length === 0) {
            if (this.storage) {
              await this.storage.unlink(`${StoragePaths.sceneBlocksDir}${file}`);
            } else {
              const fs = await import("node:fs/promises");
              const path = await import("node:path");
              await fs.default.unlink(path.default.join(this.dataDir, "scene_blocks", file));
            }
            cleanedCount++;
            this.logger?.debug?.(`${TAG} extract() removed META-only file (no content): ${file}`);
          }
        }
      }
    } catch (cleanupErr) {
      // Non-fatal — log and continue to index sync
      this.logger?.warn(`${TAG} extract() soft-delete cleanup error: ${cleanupErr instanceof Error ? cleanupErr.message : String(cleanupErr)}`);
    }
    this.logger?.debug?.(`${TAG} extract() soft-delete cleanup: removed ${cleanedCount} empty files (${Date.now() - cleanupStartMs}ms)`);

    // Phase 5b: Normalize filenames (defensive — LLM occasionally produces names
    // with spaces / punctuation despite the prompt forbidding them, e.g.
    // "Daily Rhythm in Shanghai.md". Such names break downstream consumers
    // that parse Markdown navigation refs with `\S+\.md` style regexes
    // (health-checker), shell tools, and URL-encoded path consumers.
    //
    // Renaming here — *before* syncSceneIndex — means scene_index.json and
    // every downstream reader (PersonaGenerator, recall, profile-sync) only
    // ever sees canonical filenames. Idempotent and safe to run repeatedly.
    const normStartMs = Date.now();
    try {
      const normResult = await normalizeSceneFilenames(sceneBlocksDir, this.logger);
      if (normResult.renamed > 0) {
        this.logger?.info(
          `${TAG} extract() filename normalization: renamed ${normResult.renamed}, skipped ${normResult.skipped} (${Date.now() - normStartMs}ms)`,
        );
      } else {
        this.logger?.debug?.(
          `${TAG} extract() filename normalization: skipped ${normResult.skipped} (${Date.now() - normStartMs}ms)`,
        );
      }
    } catch (normErr) {
      // Non-fatal — log and continue. Index sync below will simply pick up
      // whatever names are present on disk.
      this.logger?.warn(`${TAG} extract() filename normalization error: ${normErr instanceof Error ? normErr.message : String(normErr)}`);
    }

    // Phase 6: Sync scene index (rebuilds from remaining non-empty files)
    const syncStartMs = Date.now();
    await syncSceneIndex(this.dataDir, this.storage);
    this.logger?.debug?.(`${TAG} extract() scene index synced: ${Date.now() - syncStartMs}ms`);

    // Phase 7: Update persona.md navigation (GAP-4 fix)
    const navStartMs = Date.now();
    try {
      await this.updateSceneNavigation();
      this.logger?.debug?.(`${TAG} extract() persona.md navigation updated: ${Date.now() - navStartMs}ms`);
    } catch (navErr) {
      // Non-fatal — log and continue
      this.logger?.warn(`${TAG} extract() failed to update persona navigation: ${navErr instanceof Error ? navErr.message : String(navErr)}`);
    }

    // Phase 8: Parse LLM output for out-of-band persona update signal
    if (llmOutput) {
      const signal = parsePersonaUpdateSignal(llmOutput);
      if (signal) {
        await cpManager.setPersonaUpdateRequest(signal.reason);
        this.logger?.debug?.(`${TAG} extract() persona update requested by LLM: ${signal.reason}`);
      }
    }

    const totalMs = Date.now() - extractStartMs;
    this.logger?.info(`${TAG} extract() completed: ${memories.length} memories processed in ${totalMs}ms`);

    // ── l2_extraction metric ──
    if (this.instanceId && this.logger) {
      // Read updated scene index to report final state + diff against pre-extract snapshot
      let resultScenes: Array<{ title: string; summary: string; content: string; status: "created" | "updated" }> = [];
      let scenesCreated = 0;
      let scenesUpdated = 0;
      let scenesDeleted = 0;
      try {
        const finalIndex = await readSceneIndex(this.dataDir, this.storage);
        const postFilenames = new Set<string>();
        for (const e of finalIndex) {
          postFilenames.add(e.filename);
          const oldSummary = preExtractIndex.get(e.filename);
          // Read scene block content from disk
          let content = "";
          try {
            let raw: string | null;
            if (this.storage) {
              raw = await this.storage.readFile(`${StoragePaths.sceneBlocksDir}${e.filename}`);
            } else {
              const fs = await import("node:fs/promises");
              const path = await import("node:path");
              raw = await fs.default.readFile(path.default.join(this.dataDir, "scene_blocks", e.filename), "utf-8");
            }
            if (raw) {
              const block = parseSceneBlock(raw, e.filename);
              content = block.content;
            }
          } catch { /* file read failure is non-fatal */ }

          if (oldSummary === undefined) {
            // New scene
            scenesCreated++;
            resultScenes.push({
              title: e.filename.replace(/\.md$/, ""),
              summary: e.summary,
              content,
              status: "created",
            });
          } else {
            // Existing scene — check if content actually changed (not just metadata)
            const oldContent = preExtractContent.get(e.filename) ?? "";
            if (content !== oldContent) {
              scenesUpdated++;
              resultScenes.push({
                title: e.filename.replace(/\.md$/, ""),
                summary: e.summary,
                content,
                status: "updated",
              });
            }
            // If only metadata (summary/heat) changed but content is the same, skip
          }
        }
        // Scenes in pre-extract but missing from post-extract = deleted
        for (const [filename] of preExtractIndex) {
          if (!postFilenames.has(filename)) {
            scenesDeleted++;
          }
        }
      } catch { /* non-fatal */ }

      report("l2_extraction", {
        inputMemoryCount: memories.length,
        resultSceneCount: resultScenes.length,
        resultScenes,
        scenesCreated,
        scenesUpdated,
        scenesDeleted,
        llmDurationMs,
        totalDurationMs: totalMs,
        success: true,
        error: null,
      });

      // ── 评测指标：L2 延迟 + 场景变化 ──
      try {
        const postSceneCount = (await readSceneIndex(this.dataDir, this.storage)).length;
        reportL2LatencyMetrics({
          instanceId: this.instanceId ?? "",
          extractionLatencyMs: totalMs,
          llmDurationMs: llmDurationMs > 0 ? llmDurationMs : null,
          sceneCountBefore: preExtractIndex.size,
          sceneCountAfter: postSceneCount,
          scenesCreated,
          scenesUpdated,
          scenesDeleted,
          hasError: false,
        });
      } catch {
        // 静默忽略
      }
    }

    // Detect empty extraction: pre and post scene_index both empty means LLM didn't write anything
    const postIndex = await readSceneIndex(this.dataDir, this.storage);
    const emptyExtraction = preExtractIndex.size === 0 && postIndex.length === 0;
    if (emptyExtraction) {
      this.logger?.warn(`${TAG} extract() empty extraction detected: LLM produced no file changes`);
    }

    return { memoriesProcessed: memories.length, success: true, emptyExtraction };
  }

  /**
   * Build human-readable scene summaries for the prompt,
   * and collect the list of existing scene filenames (relative).
   *
   * Includes a capacity counter at the top (e.g. "当前场景总数：5 / 15")
   * so the LLM can immediately see how close it is to the limit.
   */
  private buildSceneSummaries(
    index: SceneIndexEntry[],
  ): { summaries: string; filenames: string[] } {
    if (index.length === 0) return { summaries: "", filenames: [] };

    const lines: string[] = [];
    const filenames: string[] = [];

    // Inject capacity counter at the top — LLM sees this first
    lines.push(`**当前场景总数：${index.length} / ${this.maxScenes}**`);
    lines.push("");

    for (const entry of index) {
      filenames.push(entry.filename);
      lines.push(`### ${entry.filename}`);
      lines.push(`**热度**: ${entry.heat} | **更新**: ${entry.updated}`);
      lines.push(`**summary**: ${entry.summary}`);
      lines.push("");
    }
    return { summaries: lines.join("\n"), filenames };
  }

  /**
   * Update the scene navigation section at the end of persona.md.
   *
   * Reads the current scene index, generates the navigation block, then
   * strips any existing navigation from persona.md and appends the new one.
   *
   * IMPORTANT: If the persona body is empty (PersonaGenerator hasn't run yet),
   * we skip writing to avoid creating a persona.md that only contains the
   * scene navigation. PersonaGenerator.generate() will write the full
   * persona + navigation when it runs.
   */
  private async updateSceneNavigation(): Promise<void> {
    const index = await readSceneIndex(this.dataDir, this.storage);
    const nav = generateSceneNavigation(index);

    let existing = "";
    try {
      if (this.storage) {
        existing = (await this.storage.readFile(StoragePaths.persona)) ?? "";
      } else {
        const fs = await import("node:fs/promises");
        const path = await import("node:path");
        existing = await fs.default.readFile(path.default.join(this.dataDir, "persona.md"), "utf-8");
      }
    } catch {
      // No persona file yet — PersonaGenerator will create it with navigation.
      // Don't write a navigation-only file.
      this.logger?.debug?.(`${TAG} updateSceneNavigation() skipped: no persona file yet, waiting for PersonaGenerator`);
      return;
    }

    if (!existing.trim() && !nav) return;

    const stripped = stripSceneNavigation(existing).trimEnd();

    // If the persona body is empty (only navigation existed), don't overwrite
    // with a navigation-only file. Let PersonaGenerator handle full generation.
    if (!stripped) {
      this.logger?.debug?.(`${TAG} updateSceneNavigation() skipped: persona body is empty, waiting for PersonaGenerator`);
      return;
    }

    const updated = nav ? `${stripped}\n\n${nav}\n` : `${stripped}\n`;

    if (this.storage) {
      await this.storage.writeFile(StoragePaths.persona, updated);
    } else {
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      await fs.default.writeFile(path.default.join(this.dataDir, "persona.md"), updated, "utf-8");
    }
  }
}

function formatTimestamp(d: Date): string {
  return d.toISOString();
}
