/**
 * Offload Ingest Handler — fast path: COS write + task enqueue.
 * Branch A: toolPairs only → write pending.jsonl, enqueue async L1
 * Branch B: recentMessages only (no toolPairs) → L1.5 path
 * Branch C: toolPairs + recentMessages → write pending + cache recentMessages, trigger L1 (skip L1.5)
 */
import type http from "node:http";
import type { StorageAdapter } from "../core/storage/adapter.js";
import type { IStateBackend, TaskPayload } from "../core/state/types.js";
import type { OffloadExecutorConfig, OffloadState } from "./types.js";
import { defaultOffloadState } from "./types.js";
import { IngestRequestSchema } from "./schemas.js";
import { serializeJsonl } from "./parsers/json-utils.js";
import { buildOffloadBasePath } from "./session-utils.js";

// ─── Per-session in-process mutex for COS append serialization ───
// Layer 1 (local): ensures at most one inflight appendFile per session within
// the same Node.js process (no I/O overhead, zero-latency queue).
// Layer 2 (distributed): stateBackend.acquireLock across multiple offload server
// instances prevents concurrent AppendObject calls to the same COS key.
const pendingMutexes = new Map<string, Promise<void>>();

async function withSessionMutex<T>(sessionKey: string, fn: () => Promise<T>): Promise<T> {
  const prev = pendingMutexes.get(sessionKey) ?? Promise.resolve();
  let resolve!: () => void;
  const next = new Promise<void>((r) => { resolve = r; });
  pendingMutexes.set(sessionKey, next);

  try {
    await prev;
    return await fn();
  } finally {
    resolve();
    if (pendingMutexes.get(sessionKey) === next) {
      pendingMutexes.delete(sessionKey);
    }
  }
}

/** Max attempts to acquire the distributed lock. */
const APPEND_LOCK_MAX_ATTEMPTS = 10;
/** TTL for the distributed append lock (ms). Short-lived since append is fast. */
const APPEND_LOCK_TTL_MS = 5000;
/** Delay between lock acquisition retries (ms). Uses exponential backoff: base * 2^attempt. */
const APPEND_LOCK_RETRY_BASE_MS = 50;

export interface IngestDeps {
  storage: StorageAdapter;
  stateBackend?: IStateBackend;
  config: OffloadExecutorConfig;
  logger: { info: (...args: unknown[]) => void; warn: (...args: unknown[]) => void };
}

export async function handleIngest(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  auth: { serviceId: string },
  deps: IngestDeps,
  requestId: string,
  parseJsonBody: <T>(req: http.IncomingMessage) => Promise<T>,
  sendJson: (res: http.ServerResponse, status: number, body: unknown) => void,
  successEnvelope: <T>(data: T, requestId: string) => unknown,
  errorEnvelope: (code: number, message: string, requestId: string) => unknown,
): Promise<void> {
  const body = await parseJsonBody(req);
  const parsed = IngestRequestSchema.safeParse(body);
  if (!parsed.success) {
    sendJson(res, 400, errorEnvelope(400, parsed.error.message, requestId));
    return;
  }

  const { session_id: sessionId, tool_pairs: toolPairs, prompt, recent_messages: recentMessages } = parsed.data;
  const { storage, stateBackend, config } = deps;
  const basePath = buildOffloadBasePath(sessionId);

  // Build context string from structured prompt + recentMessages (for L1/L1.5)
  let contextText: string | undefined;
  if (prompt || (recentMessages && recentMessages.length > 0)) {
    const parts: string[] = [];
    if (recentMessages && recentMessages.length > 0) {
      parts.push("历史消息，可作为参考：");
      for (const m of recentMessages) {
        parts.push(`[${m.role === "user" ? "User" : "Assistant"}]: ${m.content}`);
      }
    }
    if (prompt) {
      parts.push(`\n最新user message：\n[User]: ${prompt}`);
    }
    contextText = parts.join("\n");
  }

  // ─── Session skip filter (shared by L1 and L1.5) ───
  const INTERNAL_SESSION_RE = /memory-.*-session-\d+/;
  const shouldSkipSession = INTERNAL_SESSION_RE.test(sessionId) || sessionId.includes("subagent");

  // ─── Branch L1: toolPairs non-empty → write pending + trigger L1 ───
  if (toolPairs.length > 0) {
    if (shouldSkipSession) {
      deps.logger.info(`[offload-server] ingest: L1 skipped (session=${sessionId})`);
      sendJson(res, 200, successEnvelope({}, requestId));
      return;
    }
    // Save context for L1 executor (if available)
    if (contextText) {
      await storage.writeFile(`${basePath}/recent-context.txt`, contextText);
    }

    const pendingPath = `${basePath}/pending.jsonl`;
    // Map API snake_case → internal camelCase for JSONL storage
    const camelPairs = toolPairs.map((tp: Record<string, unknown>) => ({
      toolName: tp.tool_name,
      toolCallId: tp.tool_call_id,
      params: tp.params,
      result: tp.result,
      error: tp.error,
      timestamp: tp.timestamp,
      durationMs: tp.duration_ms,
    }));
    const lines = serializeJsonl(camelPairs);

    // Two-layer serialization to prevent COS AppendPositionErr:
    // Layer 1 (local mutex): queues concurrent requests within this process.
    // Layer 2 (distributed lock via stateBackend): prevents races across server instances.
    // If the lock cannot be acquired, return 409 so the client retries (NOT proceed without lock).
    const lockAcquired = await withSessionMutex(pendingPath, async () => {
      const lockKey = `offload-pending:${auth.serviceId}:${sessionId}`;
      const lockOwner = requestId;
      let locked = false;

      if (stateBackend) {
        for (let attempt = 0; attempt < APPEND_LOCK_MAX_ATTEMPTS; attempt++) {
          locked = await stateBackend.acquireLock(lockKey, lockOwner, APPEND_LOCK_TTL_MS);
          if (locked) break;
          // Exponential backoff: 50, 100, 200, 400, 800, 1600... capped at 2000ms
          const delay = Math.min(APPEND_LOCK_RETRY_BASE_MS * 2 ** attempt, 2000);
          await new Promise((r) => setTimeout(r, delay));
        }
        if (!locked) {
          deps.logger.warn(`[offload-server] ingest: append lock failed after ${APPEND_LOCK_MAX_ATTEMPTS} attempts (session=${sessionId}), returning 409`);
          return false;
        }
      }

      try {
        await storage.appendFile(pendingPath, lines);

        if (stateBackend) {
          const raw = await storage.readFile(pendingPath);
          const lineCount = raw ? raw.split("\n").filter(Boolean).length : 0;

          if (lineCount >= config.forceTriggerThreshold) {
            const task: TaskPayload = {
              id: `offload-l1-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
              type: "offload-l1" as TaskPayload["type"],
              instanceId: auth.serviceId,
              sessionId,
              priority: 0,
              data: { sessionId, instanceId: auth.serviceId },
              createdAt: Date.now(),
            };
            await stateBackend.enqueueTask(task);
            deps.logger.info(`[offload-server] ingest: ${toolPairs.length} pairs, triggered=true (lines=${lineCount})`);
          } else {
            await stateBackend.setTimerIfEarlier(
              auth.serviceId,
              `offload-l1:${auth.serviceId}:${sessionId}`,
              Date.now() + config.pendingMaxAgeSeconds * 1000,
            );
            deps.logger.info(`[offload-server] ingest: ${toolPairs.length} pairs, lines=${lineCount}/${config.forceTriggerThreshold}, timer set`);
          }
        }
        return true;
      } finally {
        if (locked && stateBackend) {
          await stateBackend.releaseLock(lockKey, lockOwner);
        }
      }
    });

    if (!lockAcquired) {
      sendJson(res, 409, errorEnvelope(409, "Concurrent write conflict, please retry", requestId));
      return;
    }

    sendJson(res, 200, successEnvelope({}, requestId));
    return;
  }

  // ─── Branch L1.5: toolPairs empty + prompt → task judgment ───
  // Skip L1.5 for inter-session messages, internal sessions, and system prompts
  const shouldSkipL15 = shouldSkipSession || !!(
    prompt && (
      prompt.startsWith("[Inter-session message]") ||
      prompt.startsWith("Pre-compaction")
    )
  );
  if (prompt && stateBackend && !shouldSkipL15) {
    const lockKey = `offload-state:${auth.serviceId}:${sessionId}`;
    const lockOwner = requestId;
    let locked = false;

    // Use exponential backoff, same policy as L1 append lock.
    for (let attempt = 0; attempt < APPEND_LOCK_MAX_ATTEMPTS; attempt++) {
      locked = await stateBackend.acquireLock(lockKey, lockOwner, APPEND_LOCK_TTL_MS);
      if (locked) break;
      const delay = Math.min(APPEND_LOCK_RETRY_BASE_MS * 2 ** attempt, 2000);
      await new Promise((r) => setTimeout(r, delay));
    }

    if (!locked) {
      // Do NOT proceed without the lock: a concurrent writer would overwrite state.json
      // and silently drop this boundary, causing the L1.5 executor to skip the task
      // (findIndex returns -1 → return without error).
      deps.logger.warn(`[offload-server] ingest: L1.5 state lock failed after ${APPEND_LOCK_MAX_ATTEMPTS} attempts (session=${sessionId}), returning 409`);
      sendJson(res, 409, errorEnvelope(409, "Concurrent write conflict, please retry", requestId));
      return;
    }

    try {
      const state = await readState(storage, basePath);
      const boundaryTimestamp = new Date().toISOString();
      state.boundaries.push({
        targetMmd: "_pending",
        timestamp: boundaryTimestamp,
      });
      await writeState(storage, basePath, state);

      const task: TaskPayload = {
        id: `offload-l15-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        type: "offload-l15" as TaskPayload["type"],
        instanceId: auth.serviceId,
        sessionId,
        priority: 0,
        data: { sessionId, recentMessages: contextText, boundaryTimestamp, instanceId: auth.serviceId },
        createdAt: Date.now(),
      };
      await stateBackend.enqueueTask(task);
      deps.logger.info(`[offload-server] ingest: L1.5 triggered, boundaryTs=${boundaryTimestamp}, prompt=${prompt.slice(0, 200)}`);
    } finally {
      await stateBackend.releaseLock(lockKey, lockOwner);
    }
  }

  // ─── Fast return ───
  sendJson(res, 200, successEnvelope({}, requestId));
}

// ─── State helpers (duplicated intentionally to keep ingest-handler self-contained) ──

async function readState(storage: StorageAdapter, basePath: string): Promise<OffloadState> {
  const raw = await storage.readFile(`${basePath}/state.json`);
  if (!raw) return defaultOffloadState();
  try {
    return { ...defaultOffloadState(), ...JSON.parse(raw) };
  } catch {
    return defaultOffloadState();
  }
}

async function writeState(storage: StorageAdapter, basePath: string, state: OffloadState): Promise<void> {
  await storage.writeFile(`${basePath}/state.json`, JSON.stringify(state));
}
