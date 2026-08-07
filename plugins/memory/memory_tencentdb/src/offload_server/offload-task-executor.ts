/**
 * Offload Task Executor — async L1/L1.5/L2 execution via PipelineWorker.
 */
import type { StorageAdapter } from "../core/storage/adapter.js";
import type { IStateBackend, TaskPayload } from "../core/state/types.js";
import type {
  OffloadEntry,
  OffloadState,
  OffloadExecutorConfig,
  L2ParsedResponse,
  MmdMeta,
  ToolPair,
} from "./types.js";
import { defaultOffloadState, defaultOffloadConfig } from "./types.js";
import { parseJsonl, serializeJsonl } from "./parsers/json-utils.js";
import { parseL1Response } from "./parsers/l1-parser.js";
import { parseL15Response } from "./parsers/l15-parser.js";
import { parseL2Response } from "./parsers/l2-parser.js";
import { L1_SYSTEM_PROMPT, buildL1UserPrompt } from "./prompts/l1-prompt.js";
import { L15_SYSTEM_PROMPT, buildL15UserPrompt } from "./prompts/l15-prompt.js";
import { L2_SYSTEM_PROMPT, buildL2UserPrompt } from "./prompts/l2-prompt.js";
import { handleTaskTransition, extractMmdMeta } from "./task-transition.js";
import { buildOffloadBasePath } from "./session-utils.js";
import { traceServerModelIo, traceServerTaskDecision } from "./opik-tracer.js";

// ─── LLM Client Interface ────────────────────────────────────────────────────

export interface LlmClient {
  chat(params: {
    model: string;
    messages: Array<{ role: "system" | "user"; content: string }>;
    temperature: number;
    max_tokens: number;
    timeoutMs?: number;
  }): Promise<string>;
}

// ─── Logger Interface ────────────────────────────────────────────────────────

export interface ExecutorLogger {
  info: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
}

// ─── Executor Deps ───────────────────────────────────────────────────────────

export interface OffloadExecutorDeps {
  resolveStorage: (instanceId: string) => Promise<StorageAdapter | undefined>;
  llmClient: LlmClient;
  stateBackend: IStateBackend;
  config?: OffloadExecutorConfig;
  logger: ExecutorLogger;
}

// ─── Executor Class ──────────────────────────────────────────────────────────

export class OffloadTaskExecutor {
  private deps: OffloadExecutorDeps;
  private config: OffloadExecutorConfig;

  constructor(deps: OffloadExecutorDeps) {
    this.deps = deps;
    this.config = deps.config ?? defaultOffloadConfig();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // L1: Summarize pending ToolPairs → OffloadEntry[]
  // ═══════════════════════════════════════════════════════════════════════════

  async executeOffloadL1(task: TaskPayload, _signal?: AbortSignal): Promise<void> {
    const startMs = Date.now();
    const sessionId = this.extractSessionId(task);
    if (!sessionId) return;

    const storage = await this.resolveStorageOrThrow(task.instanceId);
    const basePath = buildOffloadBasePath(sessionId);
    const pendingPath = `${basePath}/pending.jsonl`;

    // 1. Claim pending via rename (atomic: new ingest appends to fresh pending.jsonl)
    const processingPath = `${basePath}/pending-processing-${task.id}.jsonl`;
    const pendingRaw = await storage.readFile(pendingPath);
    if (!pendingRaw || !pendingRaw.trim()) return;

    try {
      await storage.rename(pendingPath, processingPath);
    } catch {
      // rename failed (file gone — another L1 claimed it, or doesn't exist)
      return;
    }

    // Re-read from processing file (rename succeeded, this is our exclusive copy)
    const claimedRaw = await storage.readFile(processingPath);
    if (!claimedRaw || !claimedRaw.trim()) {
      await storage.unlink(processingPath);
      return;
    }

    let toolPairs = parseJsonl<Record<string, unknown>>(claimedRaw, (line, err) => {
      this.deps.logger.warn(`[offload-server] L1: bad JSONL line in pending: ${line}`, err);
    });
    if (toolPairs.length === 0) {
      await storage.unlink(processingPath);
      return;
    }

    // 2. Dedup: remove toolPairs whose toolCallId already exists in entries.jsonl
    const entriesPath = `${basePath}/entries.jsonl`;
    const existingEntriesRaw = await storage.readFile(entriesPath);
    if (existingEntriesRaw) {
      const existingIds = new Set<string>();
      for (const e of parseJsonl<OffloadEntry>(existingEntriesRaw)) {
        if (e.tool_call_id) existingIds.add(e.tool_call_id);
      }
      const before = toolPairs.length;
      toolPairs = toolPairs.filter((tp) => {
        const id = tp.toolCallId as string;
        return !id || !existingIds.has(id);
      });
      if (toolPairs.length < before) {
        this.deps.logger.info(`[offload-server] L1: dedup removed ${before - toolPairs.length}/${before} duplicate toolPairs`);
      }
      if (toolPairs.length === 0) {
        await storage.unlink(processingPath);
        return;
      }
    }

    // 3. Build prompt & call LLM
    const recentContext = await storage.readFile(`${basePath}/recent-context.txt`) ?? "";
    const userPrompt = buildL1UserPrompt(recentContext, toolPairs as unknown as ToolPair[]);

    let newEntries: OffloadEntry[];
    const l1LlmStart = Date.now();
    let l1RawResponse: string | undefined;
    try {
      const response = await this.deps.llmClient.chat({
        model: this.config.l1Model,
        messages: [
          { role: "system", content: L1_SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
        temperature: this.config.l1Temperature,
        max_tokens: this.config.l1MaxTokens,
        timeoutMs: this.config.l1TimeoutMs,
      });
      l1RawResponse = response;
      newEntries = parseL1Response(response);
      traceServerModelIo({
        sessionId,
        stage: "L1",
        model: this.config.l1Model,
        systemPrompt: L1_SYSTEM_PROMPT,
        userPrompt,
        responseContent: response,
        status: "ok",
        durationMs: Date.now() - l1LlmStart,
        logger: this.deps.logger,
      });
    } catch (err) {
      traceServerModelIo({
        sessionId,
        stage: "L1",
        model: this.config.l1Model,
        systemPrompt: L1_SYSTEM_PROMPT,
        userPrompt,
        responseContent: l1RawResponse ?? "",
        status: "error",
        errorMessage: String(err),
        durationMs: Date.now() - l1LlmStart,
        logger: this.deps.logger,
      });
      this.deps.logger.error(`[offload-server] L1 LLM failed:`, err);
      throw err; // Let Worker retry (processing file remains for re-claim)
    }

    // 4. Ensure all toolPairs are covered — fallback for missing entries
    const parsedIds = new Set(newEntries.map((e) => e.tool_call_id));
    // Build toolCallId → original timestamp map for reliable boundary matching
    const tpTimestampMap = new Map<string, string>();
    for (const tp of toolPairs) {
      const id = tp.toolCallId as string;
      if (id && tp.timestamp) tpTimestampMap.set(id, tp.timestamp as string);
    }
    // Overwrite entry timestamps with original tool pair timestamps (don't trust LLM output)
    for (const entry of newEntries) {
      const origTs = tpTimestampMap.get(entry.tool_call_id);
      if (origTs) entry.timestamp = origTs;
    }
    for (const tp of toolPairs) {
      const id = tp.toolCallId as string;
      if (id && !parsedIds.has(id)) {
        newEntries.push({
          tool_call_id: id,
          tool_call: (tp.toolName as string) ?? "",
          summary: "[L1 parse incomplete]",
          timestamp: (tp.timestamp as string) ?? new Date().toISOString(),
          score: 2,
          node_id: null,
        });
      }
    }

    // 5. Write refs: store original tool result content for each pair
    for (const tp of toolPairs) {
      const id = tp.toolCallId as string;
      if (!id) continue;
      const result = tp.result ?? tp.error ?? "";
      const resultStr = typeof result === "string" ? result : JSON.stringify(result, null, 2);
      if (!resultStr || resultStr.length < 20) continue; // skip trivially short results
      const toolName = (tp.toolName as string) ?? "unknown";
      const timestamp = (tp.timestamp as string) ?? new Date().toISOString();
      const header = `# Tool Result: ${toolName}\n\n**tool_call_id:** ${id}\n**Timestamp:** ${timestamp}\n\n---\n\n`;
      const refPath = `${basePath}/refs/${id}.md`;
      await storage.writeFile(refPath, header + resultStr);
      // Set result_ref on the matching entry (full relative path for tdai_read_cos)
      const entry = newEntries.find((e) => e.tool_call_id === id);
      if (entry) entry.result_ref = `${basePath}/refs/${id}.md`;
    }

    // 6. Write entries (atomic append) + delete processing file
    await storage.appendFile(entriesPath, serializeJsonl(newEntries));
    await storage.unlink(processingPath);

    // 7. Check L2 trigger: group null entries by boundary targetMmd
    const state = await this.readState(storage, basePath);
    const allEntriesRaw = await storage.readFile(entriesPath);
    if (allEntriesRaw && state.boundaries.length > 0) {
      const allEntries = parseJsonl<OffloadEntry>(allEntriesRaw, (line, err) => {
        this.deps.logger.warn(`[offload-server] L1: bad JSONL line in entries: ${line}`, err);
      });
      const nodeMapping = await this.readNodeMapping(storage, basePath);

      // Group null entries by resolved targetMmd
      const nullByMmd = new Map<string, number>();
      for (const e of allEntries) {
        if (this.getEffectiveNodeId(e, nodeMapping) !== null || !e.timestamp) continue;
        const boundary = this.findBoundaryByTimestamp(state.boundaries, e.timestamp);
        if (!boundary) continue; // no boundary → ignore
        if (boundary.targetMmd === "_pending" || !boundary.targetMmd) continue; // pending → ignore
        nullByMmd.set(boundary.targetMmd, (nullByMmd.get(boundary.targetMmd) ?? 0) + 1);
      }

      // Trigger L2 for each MMD that reached threshold
      for (const [mmdFile, count] of nullByMmd) {
        if (count >= this.config.l2NullThreshold) {
          // Threshold met: short delay (1s) so concurrent L1s merge into one L2
          await this.deps.stateBackend.setTimerIfEarlier(
            task.instanceId,
            `offload-l2:${task.instanceId}:${sessionId}:${mmdFile}`,
            Date.now() + 1_000,
          );
          this.deps.logger.info(`[offload-server] L2 timer set (fast, mmd=${mmdFile}, nullCount=${count})`);
        } else if (count > 0) {
          await this.deps.stateBackend.setTimerIfEarlier(
            task.instanceId,
            `offload-l2:${task.instanceId}:${sessionId}:${mmdFile}`,
            Date.now() + 30_000,
          );
          this.deps.logger.info(`[offload-server] L2 timer set (mmd=${mmdFile}, nullCount=${count})`);
        }
      }
    }

    this.deps.logger.info(
      `[offload-server] L1 complete: ${newEntries.length} entries produced (${Date.now() - startMs}ms)`,
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // L1.5: Task judgment — triggered by user new message
  // ═══════════════════════════════════════════════════════════════════════════

  async executeOffloadL15(task: TaskPayload, _signal?: AbortSignal): Promise<void> {
    const startMs = Date.now();
    const { sessionId, recentMessages, boundaryTimestamp } = task.data as {
      sessionId: string;
      recentMessages?: string;
      boundaryTimestamp: string;
    };
    const storage = await this.resolveStorageOrThrow(task.instanceId);
    const basePath = buildOffloadBasePath(sessionId);

    // ─── Phase 1: Read-only snapshot + LLM call (NO LOCK) ─────────────────
    // Multiple L1.5 tasks can execute this phase concurrently. Each operates
    // on its own boundary and the LLM call (3-10s) does not block others.

    // 1. Read current active MMD (snapshot for prompt building)
    const preState = await this.readState(storage, basePath);
    let currentMmd: { filename: string; content: string } | null = null;
    if (preState.activeMmdFile) {
      const content = await storage.readFile(`${basePath}/mmds/${preState.activeMmdFile}`);
      if (content) {
        currentMmd = { filename: preState.activeMmdFile, content };
      }
    }

    // 2. List available MMDs → extract metas
    const mmdsPrefix = `${basePath}/mmds/`;
    const mmdFiles = await storage.readdirNames(mmdsPrefix, ".mmd");
    const metas: MmdMeta[] = [];
    for (const f of mmdFiles) {
      const content = await storage.readFile(`${mmdsPrefix}${f}`);
      if (content) {
        metas.push(extractMmdMeta(f, content));
      }
    }

    // 3. Build prompt & call LLM (no lock held — this is the expensive part)
    const userPrompt = buildL15UserPrompt(
      recentMessages ?? "",
      currentMmd,
      metas,
    );

    let judgment;
    let rawResponse: string | undefined;
    const l15LlmStart = Date.now();
    try {
      const response = await this.deps.llmClient.chat({
        model: this.config.l15Model,
        messages: [
          { role: "system", content: L15_SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
        temperature: this.config.l15Temperature,
        max_tokens: this.config.l15MaxTokens,
        timeoutMs: this.config.l15TimeoutMs,
      });
      rawResponse = response;
      judgment = parseL15Response(response);
      traceServerModelIo({
        sessionId,
        stage: "L1.5",
        model: this.config.l15Model,
        systemPrompt: L15_SYSTEM_PROMPT,
        userPrompt,
        responseContent: response,
        status: "ok",
        durationMs: Date.now() - l15LlmStart,
        logger: this.deps.logger,
      });
    } catch (err) {
      traceServerModelIo({
        sessionId,
        stage: "L1.5",
        model: this.config.l15Model,
        systemPrompt: L15_SYSTEM_PROMPT,
        userPrompt,
        responseContent: rawResponse ?? "",
        status: "error",
        errorMessage: String(err),
        durationMs: Date.now() - l15LlmStart,
        logger: this.deps.logger,
      });
      this.deps.logger.error(`[offload-server] L1.5 LLM failed:`, err);
      throw err;
    }

    if (!judgment) {
      this.deps.logger.warn(`[offload-server] L1.5: null response (parse failed), raw=${rawResponse?.slice(0, 500) ?? "(empty)"}`);
      return;
    }

    this.deps.logger.info(
      `[offload-server] L1.5: completed=${judgment.taskCompleted}, long=${judgment.isLongTask}, cont=${judgment.isContinuation}, label=${judgment.newTaskLabel ?? "none"} (LLM=${Date.now() - l15LlmStart}ms)`,
    );

    // Trace L1.5 decision
    traceServerTaskDecision({
      sessionId,
      judgment: judgment as unknown as Record<string, unknown>,
      durationMs: Date.now() - startMs,
      logger: this.deps.logger,
    });

    // ─── Phase 2: Write phase (SHORT LOCK) ────────────────────────────────
    // Only lock during state.json read-modify-write. Lock TTL is short (10s)
    // because this phase only does file I/O (no LLM calls).
    const lockKey = `offload-state:${task.instanceId}:${sessionId}`;
    const lockOwner = task.id;
    let locked = false;
    for (let attempt = 0; attempt < 10; attempt++) {
      locked = await this.deps.stateBackend.acquireLock(lockKey, lockOwner, 10_000);
      if (locked) break;
      await new Promise((r) => setTimeout(r, 100 + attempt * 50));
    }
    if (!locked) {
      this.deps.logger.warn(`[offload-server] L1.5: failed to acquire write lock after 10 attempts (ts=${boundaryTimestamp}), skipping`);
      return;
    }

    try {
      // Re-read state (fresh, may have been modified by another L1.5 that finished earlier)
      const state = await this.readState(storage, basePath);

      // CAS check: find boundary by timestamp
      const boundaryIdx = state.boundaries.findIndex((b) => b.timestamp === boundaryTimestamp);
      if (boundaryIdx < 0) {
        this.deps.logger.warn(`[offload-server] L1.5: boundary not found for ts=${boundaryTimestamp}, skipping`);
        return;
      }

      // CAS check: already backfilled by another concurrent L1.5? Skip.
      if (state.boundaries[boundaryIdx].targetMmd !== "_pending") {
        this.deps.logger.info(`[offload-server] L1.5: boundary[ts=${boundaryTimestamp}] already backfilled, skipping`);
        return;
      }

      // Apply task transition
      await handleTaskTransition(state, judgment, storage, basePath);

      // Backfill boundary targetMmd
      state.boundaries[boundaryIdx].targetMmd = state.activeMmdFile;

      // Update state
      await this.writeState(storage, basePath, state);

      // Check L2 trigger: scan ALL null entries grouped by targetMmd.
      // This covers the case where L1 completed before L1.5 backfill —
      // L1 skipped entries with _pending boundaries, so L1.5 must pick them up.
      const entriesRaw = await storage.readFile(`${basePath}/entries.jsonl`);
      if (entriesRaw) {
        const allEntries = parseJsonl<OffloadEntry>(entriesRaw);
        const l15NodeMapping = await this.readNodeMapping(storage, basePath);

        // Group null entries by their resolved targetMmd (using all boundaries, not just current)
        const nullByMmd = new Map<string, number>();
        for (const e of allEntries) {
          if (this.getEffectiveNodeId(e, l15NodeMapping) !== null || !e.timestamp) continue;
          const boundary = this.findBoundaryByTimestamp(state.boundaries, e.timestamp);
          if (!boundary) continue;
          if (boundary.targetMmd === "_pending" || !boundary.targetMmd) continue;
          nullByMmd.set(boundary.targetMmd, (nullByMmd.get(boundary.targetMmd) ?? 0) + 1);
        }

        for (const [mmdFile, nullCount] of nullByMmd) {
          if (nullCount >= this.config.l2NullThreshold) {
            await this.deps.stateBackend.setTimerIfEarlier(
              task.instanceId,
              `offload-l2:${task.instanceId}:${sessionId}:${mmdFile}`,
              Date.now() + 1_000,
            );
            this.deps.logger.info(`[offload-server] L1.5: L2 timer set (fast, mmd=${mmdFile}, nullCount=${nullCount})`);
          } else if (nullCount > 0) {
            await this.deps.stateBackend.setTimerIfEarlier(
              task.instanceId,
              `offload-l2:${task.instanceId}:${sessionId}:${mmdFile}`,
              Date.now() + 30_000,
            );
            this.deps.logger.info(`[offload-server] L1.5: L2 timer set after backfill (mmd=${mmdFile}, nullCount=${nullCount})`);
          }
        }
      }
    } finally {
      if (locked) {
        await this.deps.stateBackend.releaseLock(lockKey, lockOwner);
      }
    }

    this.deps.logger.info(
      `[offload-server] L1.5 complete: boundary[ts=${boundaryTimestamp}] (total=${Date.now() - startMs}ms)`,
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // L2: MMD generation / update + node_id backfill
  // ═══════════════════════════════════════════════════════════════════════════

  async executeOffloadL2(task: TaskPayload, _signal?: AbortSignal): Promise<void> {
    const startMs = Date.now();
    const sessionId = this.extractSessionId(task);
    if (!sessionId) return;

    let targetMmdFile: string | undefined;
    const data = task.data as Record<string, unknown>;
    targetMmdFile = data.targetMmdFile as string | undefined;

    // Extract targetMmdFile from timer member if available
    // Format: "offload-l2:{instanceId}:{sessionId}:{mmdFile}" → extract last segment ending in .mmd
    if (!targetMmdFile && data.timerMember) {
      const timerMember = data.timerMember as string;
      const mmdMatch = timerMember.match(/(\d+-[^:]+\.mmd)$/);
      if (mmdMatch) {
        targetMmdFile = mmdMatch[1];
      }
    }

    const storage = await this.resolveStorageOrThrow(task.instanceId);
    const basePath = buildOffloadBasePath(sessionId);
    const state = await this.readState(storage, basePath);

    // Resolve targetMmdFile: from task data, timer member, or state (fallback)
    if (!targetMmdFile) {
      targetMmdFile = state.activeMmdFile ?? undefined;
    }
    if (!targetMmdFile) return;

    // 1. Read all entries + node mapping
    const entriesRaw = await storage.readFile(`${basePath}/entries.jsonl`);
    if (!entriesRaw) return;
    const allEntries = parseJsonl<OffloadEntry>(entriesRaw);
    const nodeMapping = await this.readNodeMapping(storage, basePath);

    // 2. Filter: entries belonging to targetMmdFile with null node_id (after join)
    const relevantEntries = allEntries.filter((e) => {
      if (this.getEffectiveNodeId(e, nodeMapping) !== null) return false;
      if (!e.timestamp) return false;
      const boundary = this.findBoundaryByTimestamp(state.boundaries, e.timestamp);
      return boundary?.targetMmd === targetMmdFile;
    });

    if (relevantEntries.length === 0) return;

    // Guard: if oldest null entry is older than 10 minutes, give up and assign fallback node_id
    // to prevent infinite L2 retries when LLM consistently fails to map certain entries.
    const L2_MAX_AGE_MS = 10 * 60 * 1000;
    const oldestTs = relevantEntries.reduce((min, e) => {
      const t = new Date(e.timestamp).getTime();
      return t < min ? t : min;
    }, Infinity);
    if (Date.now() - oldestTs > L2_MAX_AGE_MS) {
      this.deps.logger.warn(
        `[offload-server] L2: ${relevantEntries.length} entries exceeded max age (10min), assigning fallback node_id`,
      );
      const fallbackMappings = relevantEntries.map((e) => ({
        tool_call_id: e.tool_call_id,
        node_id: `${targetMmdFile!.replace(/\.mmd$/, "")}-orphan`,
      }));
      const mappingPath = `${basePath}/node-mapping.jsonl`;
      await storage.appendFile(mappingPath, serializeJsonl(fallbackMappings));
      return;
    }

    // 3. Read existing MMD
    const mmdPath = `${basePath}/mmds/${targetMmdFile}`;
    const existingMmd = await storage.readFile(mmdPath);

    // 4. Build prompt (include recent context for better MMD generation)
    const recentHistory = await storage.readFile(`${basePath}/recent-context.txt`) ?? null;
    const taskLabel = targetMmdFile.replace(/^\d+-/, "").replace(/\.mmd$/, "") || "task";
    const prefixMatch = targetMmdFile.match(/^(\d+)-/);
    const mmdPrefix = prefixMatch ? prefixMatch[1] : "000";
    const charCount = existingMmd?.length ?? 0;

    const userPrompt = buildL2UserPrompt({
      existingMmd: existingMmd || null,
      entries: relevantEntries,
      recentHistory,
      taskLabel,
      mmdPrefix,
      charCount,
    });

    // 5. Call LLM
    let result: L2ParsedResponse | null;
    let rawL2Response: string | undefined;
    const l2LlmStart = Date.now();
    try {
      const response = await this.deps.llmClient.chat({
        model: this.config.l2Model,
        messages: [
          { role: "system", content: L2_SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
        temperature: this.config.l2Temperature,
        max_tokens: this.config.l2MaxTokens,
        timeoutMs: this.config.l2TimeoutMs,
      });
      rawL2Response = response;
      result = parseL2Response(response);
      traceServerModelIo({
        sessionId: sessionId!,
        stage: "L2",
        model: this.config.l2Model,
        systemPrompt: L2_SYSTEM_PROMPT,
        userPrompt,
        responseContent: response,
        status: "ok",
        durationMs: Date.now() - l2LlmStart,
        logger: this.deps.logger,
      });
    } catch (err) {
      traceServerModelIo({
        sessionId: sessionId!,
        stage: "L2",
        model: this.config.l2Model,
        systemPrompt: L2_SYSTEM_PROMPT,
        userPrompt,
        responseContent: rawL2Response ?? "",
        status: "error",
        errorMessage: String(err),
        durationMs: Date.now() - l2LlmStart,
        logger: this.deps.logger,
      });
      this.deps.logger.error(`[offload-server] L2 LLM failed:`, err);
      throw err;
    }

    if (!result) {
      this.deps.logger.warn(`[offload-server] L2: parse failed, raw=${rawL2Response?.slice(0, 500) ?? "(empty)"}`);
      // Schedule retry so remaining null entries are not orphaned
      await this.deps.stateBackend.setTimerIfEarlier(
        task.instanceId,
        `offload-l2:${task.instanceId}:${sessionId}:${targetMmdFile}`,
        Date.now() + 30_000,
      );
      return;
    }

    // 6. Apply MMD update
    const updatedMmd = this.applyL2Result(existingMmd ?? "", result);
    await storage.writeFile(mmdPath, updatedMmd);

    // 7. Backfill: write node mappings to separate file (avoids overwriting entries.jsonl)
    const mappingEntries = Object.entries(result.nodeMapping).map(([toolCallId, nodeId]) => ({
      tool_call_id: toolCallId,
      node_id: nodeId,
    }));
    if (mappingEntries.length > 0) {
      const mappingPath = `${basePath}/node-mapping.jsonl`;
      await storage.appendFile(mappingPath, serializeJsonl(mappingEntries));
    }

    // 8. Check remaining null entries for this MMD — re-read node-mapping for accurate count
    //    (another L2 may have concurrently written mappings, or LLM may have missed some)
    const freshNodeMapping = await this.readNodeMapping(storage, basePath);
    const remainingNull = relevantEntries.filter(
      (e) => !freshNodeMapping.has(e.tool_call_id) && !result!.nodeMapping[e.tool_call_id],
    ).length;
    if (remainingNull > 0) {
      await this.deps.stateBackend.setTimerIfEarlier(
        task.instanceId,
        `offload-l2:${task.instanceId}:${sessionId}:${targetMmdFile}`,
        Date.now() + 30_000,
      );
      this.deps.logger.info(`[offload-server] L2 retry timer set (mmd=${targetMmdFile}, remainingNull=${remainingNull})`);
    }

    this.deps.logger.info(
      `[offload-server] L2 complete: ${Object.keys(result.nodeMapping).length} entries mapped, action=${result.fileAction} (${Date.now() - startMs}ms)`,
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Helpers
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Extract sessionId from task data or task.sessionId.
   */
  private extractSessionId(task: TaskPayload): string | undefined {
    const data = task.data as Record<string, unknown> | undefined;
    // Prefer explicit sessionId in data
    if (data?.sessionId && typeof data.sessionId === "string") {
      return data.sessionId;
    }
    // Fallback: task-level sessionId
    if (task.sessionId) {
      return task.sessionId;
    }
    return undefined;
  }

  private async resolveStorageOrThrow(instanceId: string): Promise<StorageAdapter> {
    const storage = await this.deps.resolveStorage(instanceId);
    if (!storage) throw new Error(`Storage unavailable for instance ${instanceId}`);
    return storage;
  }

  private async readState(storage: StorageAdapter, basePath: string): Promise<OffloadState> {
    const raw = await storage.readFile(`${basePath}/state.json`);
    if (!raw) return defaultOffloadState();
    try {
      return { ...defaultOffloadState(), ...JSON.parse(raw) };
    } catch {
      return defaultOffloadState();
    }
  }

  private async writeState(storage: StorageAdapter, basePath: string, state: OffloadState): Promise<void> {
    await storage.writeFile(`${basePath}/state.json`, JSON.stringify(state));
  }

  private findBoundaryByTimestamp(
    boundaries: OffloadState["boundaries"],
    entryTimestamp: string,
  ): OffloadState["boundaries"][number] | null {
    if (boundaries.length === 0) return null;
    // Find the last boundary whose timestamp <= entryTimestamp
    let result: OffloadState["boundaries"][number] | null = null;
    for (const b of boundaries) {
      if (b.timestamp <= entryTimestamp) result = b;
      else break;
    }
    return result;
  }

  private applyL2Result(existingMmd: string, result: L2ParsedResponse): string {
    if (result.fileAction === "write" && result.mmdContent) {
      return result.mmdContent;
    }

    if (result.fileAction === "replace" && result.replaceBlocks?.length) {
      const lines = existingMmd.split("\n");
      // Sort blocks by startLine descending to avoid offset issues
      const sorted = [...result.replaceBlocks]
        .filter((block) => block.startLine >= 1 && block.endLine >= block.startLine && block.startLine <= lines.length)
        .sort((a, b) => b.startLine - a.startLine);
      for (const block of sorted) {
        const start = block.startLine - 1; // 0-based
        const end = Math.min(block.endLine, lines.length); // clamp endLine
        const deleteCount = end - block.startLine + 1;
        const newLines = block.content.split("\n");
        lines.splice(start, deleteCount, ...newLines);
      }
      return lines.join("\n");
    }

    return existingMmd;
  }

  /**
   * Read node-mapping.jsonl and build a tool_call_id → node_id map.
   */
  private async readNodeMapping(storage: StorageAdapter, basePath: string): Promise<Map<string, string>> {
    const raw = await storage.readFile(`${basePath}/node-mapping.jsonl`);
    const map = new Map<string, string>();
    if (!raw) return map;
    const lines = parseJsonl<{ tool_call_id: string; node_id: string }>(raw);
    for (const line of lines) {
      if (line.tool_call_id && line.node_id) {
        map.set(line.tool_call_id, line.node_id);
      }
    }
    return map;
  }

  /**
   * Get effective node_id for an entry: check node-mapping first, then entry's own field.
   */
  private getEffectiveNodeId(entry: OffloadEntry, nodeMapping: Map<string, string>): string | null {
    return nodeMapping.get(entry.tool_call_id) ?? entry.node_id;
  }
}
