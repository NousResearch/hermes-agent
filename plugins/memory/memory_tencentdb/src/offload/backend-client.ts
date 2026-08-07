/**
 * Backend HTTP Client for Context Offload.
 *
 * When `backendUrl` is configured, L1/L1.5/L2/L4 LLM calls are routed
 * through this client to the backend service. The backend handles
 * prompt construction + LLM invocation; the client handles data
 * collection and file I/O.
 *
 * All methods throw on failure — callers are responsible for fallback.
 */
import type { OffloadEntry, ToolPair, TaskJudgment, PluginLogger } from "./types.js";
import { traceOffloadModelIo } from "./opik-tracer.js";
import * as https from "node:https";
import * as http from "node:http";

// ─── Request / Response Types ────────────────────────────────────────────────

export interface L1Request {
  recentMessages: string;
  toolPairs: Array<{
    toolName: string;
    toolCallId: string;
    params: unknown;
    result: unknown;
    timestamp: string;
  }>;
  pluginConfig?: Record<string, unknown>;
}

export interface L1Response {
  entries: OffloadEntry[];
}

export interface L15Request {
  recentMessages: string;
  currentMmd?: {
    filename: string;
    content: string;
    path: string;
  } | null;
  availableMmdMetas: Array<{
    filename: string;
    path: string;
    taskGoal: string;
    doneCount: number;
    doingCount: number;
    todoCount: number;
    updatedTime?: string | null;
    nodeSummaries?: Array<{ nodeId: string; status: string; summary: string }>;
  }>;
}

export interface L15Response extends TaskJudgment {}

export interface L2Request {
  existingMmd: string | null;
  newEntries: Array<{
    tool_call_id: string;
    tool_call: string;
    summary: string;
    timestamp: string;
  }>;
  recentHistory: string | null;
  currentTurn: string | null;
  taskLabel: string;
  mmdPrefix: string;
  mmdCharCount: number;
}

export interface L2Response {
  fileAction: "write" | "replace";
  mmdContent?: string;
  replaceBlocks?: Array<{
    startLine: number;
    endLine: number;
    content: string;
  }>;
  nodeMapping: Record<string, string>;
}

export interface L4Request {
  mmdFilename: string;
  mmdContent: string;
  offloadEntries: OffloadEntry[];
  skillFocus: string | null;
}

export interface L4Response {
  skillName: string;
  skillDescription: string;
  skillContent: string;
}

/**
 * Arbitrary key/value payload uploaded to the backend `/offload/v1/store` endpoint.
 * The backend stores the raw JSON body verbatim; see `internal/handler/store.go`.
 */
export type StoreStatePayload = Record<string, unknown>;

export interface StoreStateResponse {
  insertedId?: string;
}

// ─── BackendClient ───────────────────────────────────────────────────────────

export class BackendClient {
  private baseUrl: string;
  private apiKey: string | undefined;
  /** Hardcoded timeout for all backend calls (L1/L1.5/L2/L4) */
  private static readonly TIMEOUT_MS = 120_000;
  private logger: PluginLogger;
  private sessionKeyFn: () => string | null;
  /** Resolves the value of the `X-User-Id` header sent on every call. */
  private userIdFn: () => string | null;
  /** Resolves the value of the `X-Task-Id` header sent on every call (optional). */
  private taskIdFn: () => string | null;

  constructor(
    baseUrl: string,
    logger: PluginLogger,
    apiKey?: string,
    _defaultTimeoutMs?: number, // kept for backward compat, ignored
    sessionKeyFn?: () => string | null,
    userIdFn?: () => string | null,
    taskIdFn?: () => string | null,
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
    this.logger = logger;
    this.sessionKeyFn = sessionKeyFn ?? (() => null);
    this.userIdFn = userIdFn ?? (() => null);
    this.taskIdFn = taskIdFn ?? (() => null);
  }

  /** L1 Summarize — synchronous await (used by assemble flush + force trigger) */
  async l1Summarize(req: L1Request): Promise<L1Response> {
    const pairNames = req.toolPairs.map((p) => `${p.toolName}(${p.toolCallId})`).join(", ");
    this.logger.debug?.(`[context-offload] L1 >>> summarize ${req.toolPairs.length} pairs: [${pairNames}]`);
    const startMs = Date.now();
    const resp = await this.post<L1Response>("/offload/v1/l1/summarize", req, BackendClient.TIMEOUT_MS);
    const durationMs = Date.now() - startMs;
    const entryCount = resp.entries?.length ?? 0;
    const scores = resp.entries?.map((e) => `${e.tool_call_id}:score=${e.score}`).join(", ") ?? "";
    this.logger.debug?.(`[context-offload] L1 <<< ${entryCount} entries [${scores}]`);
    traceOffloadModelIo({
      sessionKey: this.sessionKeyFn(),
      stage: "L1.backend",
      provider: "backend",
      model: `backend:${this.baseUrl}`,
      url: `${this.baseUrl}/offload/v1/l1/summarize`,
      systemPrompt: "(constructed by backend)",
      userPrompt: JSON.stringify(req),
      responseContent: JSON.stringify(resp),
      usage: { entriesCount: entryCount },
      status: "ok",
      durationMs,
      logger: this.logger,
    });
    return resp;
  }

  /** L1.5 Task Judgment — synchronous await, uses unified timeout */
  async l15Judge(req: L15Request): Promise<L15Response> {
    this.logger.debug?.(
      `[context-offload] L1.5 >>> judge: currentMmd=${req.currentMmd?.filename ?? "null"}, availableMmds=${req.availableMmdMetas.length}, recentMessages=${req.recentMessages.length} chars`,
    );
    const startMs = Date.now();
    const resp = await this.post<L15Response>("/offload/v1/l15/judge", req, BackendClient.TIMEOUT_MS);
    const durationMs = Date.now() - startMs;
    this.logger.debug?.(
      `[context-offload] L1.5 <<< completed=${resp.taskCompleted}, continuation=${resp.isContinuation}, continuationFile=${resp.continuationMmdFile ?? "null"}, newLabel=${resp.newTaskLabel ?? "null"}, longTask=${resp.isLongTask}`,
    );
    traceOffloadModelIo({
      sessionKey: this.sessionKeyFn(),
      stage: "L1.5.backend",
      provider: "backend",
      model: `backend:${this.baseUrl}`,
      url: `${this.baseUrl}/offload/v1/l15/judge`,
      systemPrompt: "(constructed by backend)",
      userPrompt: JSON.stringify(req),
      responseContent: JSON.stringify(resp),
      status: "ok",
      durationMs,
      logger: this.logger,
    });
    return resp;
  }

  /** L2 MMD Generation — async background, uses unified timeout */
  async l2Generate(req: L2Request): Promise<L2Response> {
    const entryIds = req.newEntries.map((e) => e.tool_call_id).join(", ");
    this.logger.debug?.(
      `[context-offload] L2 >>> generate: task=${req.taskLabel}, prefix=${req.mmdPrefix}, entries=${req.newEntries.length} [${entryIds}], existingMmd=${req.existingMmd ? `${req.mmdCharCount} chars` : "null (new)"}`,
    );
    const startMs = Date.now();
    const resp = await this.post<L2Response>("/offload/v1/l2/generate", req, BackendClient.TIMEOUT_MS);
    const durationMs = Date.now() - startMs;
    const mappingCount = Object.keys(resp.nodeMapping ?? {}).length;
    const mappingStr = Object.entries(resp.nodeMapping ?? {}).map(([k, v]) => `${k}->${v}`).join(", ");
    this.logger.debug?.(
      `[context-offload] L2 <<< action=${resp.fileAction}, mmdContent=${resp.mmdContent ? `${resp.mmdContent.length} chars` : "null"}, replaceBlocks=${resp.replaceBlocks?.length ?? 0}, nodeMapping=${mappingCount} [${mappingStr}]`,
    );
    traceOffloadModelIo({
      sessionKey: this.sessionKeyFn(),
      stage: "L2.backend",
      provider: "backend",
      model: `backend:${this.baseUrl}`,
      url: `${this.baseUrl}/offload/v1/l2/generate`,
      systemPrompt: "(constructed by backend)",
      userPrompt: JSON.stringify(req),
      responseContent: JSON.stringify(resp),
      status: "ok",
      durationMs,
      logger: this.logger,
    });
    return resp;
  }

  /** L4 Skill Generation — synchronous await, uses unified timeout */
  async l4Generate(req: L4Request): Promise<L4Response> {
    this.logger.debug?.(
      `[context-offload] L4 >>> generate: mmd=${req.mmdFilename}, entries=${req.offloadEntries.length}, skillFocus=${req.skillFocus ?? "null"}`,
    );
    const startMs = Date.now();
    const resp = await this.post<L4Response>("/offload/v1/l4/generate", req, BackendClient.TIMEOUT_MS);
    const durationMs = Date.now() - startMs;
    this.logger.debug?.(
      `[context-offload] L4 <<< skill="${resp.skillName}", content=${resp.skillContent?.length ?? 0} chars`,
    );
    traceOffloadModelIo({
      sessionKey: this.sessionKeyFn(),
      stage: "L4.backend",
      provider: "backend",
      model: `backend:${this.baseUrl}`,
      url: `${this.baseUrl}/offload/v1/l4/generate`,
      systemPrompt: "(constructed by backend)",
      userPrompt: JSON.stringify(req),
      responseContent: JSON.stringify(resp),
      status: "ok",
      durationMs,
      logger: this.logger,
    });
    return resp;
  }

  /**
   * Upload an arbitrary state payload to the backend `/offload/v1/store` endpoint.
   * Fire-and-forget style — the caller is expected to `.catch(...)` rejections.
   * Uses a short timeout so reporting never blocks hook execution meaningfully.
   */
  async storeState(payload: StoreStatePayload): Promise<StoreStateResponse> {
    // Short timeout — reporting must never stall the plugin
    const timeoutMs = 10_000;
    const startMs = Date.now();
    try {
      const resp = await this.post<StoreStateResponse>("/offload/v1/store", payload, timeoutMs);
      const durationMs = Date.now() - startMs;
      this.logger.debug?.(
        `[context-offload] store <<< insertedId=${resp.insertedId ?? "?"} (${durationMs}ms)`,
      );
      return resp;
    } catch (err) {
      const durationMs = Date.now() - startMs;
      this.logger.warn(`[context-offload] store !!! failed after ${durationMs}ms: ${err}`);
      throw err;
    }
  }

  // ─── Internal ──────────────────────────────────────────────────────────

  private async post<T>(path: string, body: unknown, timeoutMs: number): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const startMs = Date.now();

    const bodyStr = JSON.stringify(body);
    this.logger.debug?.(`[context-offload] HTTP >>> POST ${url} (${bodyStr.length} bytes, timeout=${timeoutMs}ms)`);

    const reqHeaders: Record<string, string> = {
      "Content-Type": "application/json",
      "Content-Length": String(Buffer.byteLength(bodyStr)),
    };
    if (this.apiKey) {
      reqHeaders["Authorization"] = `Bearer ${this.apiKey}`;
    }
    // Propagate identity headers so the backend can key stored state by
    // `X-User-Id` (used as Mongo `_id` in /store) and scope by task.
    try {
      const uid = this.userIdFn();
      if (uid) reqHeaders["X-User-Id"] = uid;
    } catch { /* ignore — identity headers are best-effort */ }
    try {
      const tid = this.taskIdFn();
      if (tid) reqHeaders["X-Task-Id"] = tid;
    } catch { /* ignore */ }

    const parsed = new URL(url);
    const isHttps = parsed.protocol === "https:";
    const transport = isHttps ? https : http;

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        req.destroy(new Error("timeout"));
      }, timeoutMs);

      const req = transport.request(
        {
          hostname: parsed.hostname,
          port: parsed.port || (isHttps ? 443 : 80),
          path: parsed.pathname + parsed.search,
          method: "POST",
          headers: reqHeaders,
          ...(isHttps ? { rejectUnauthorized: false } : {}),
        },
        (res) => {
          let data = "";
          res.on("data", (chunk: Buffer) => {
            data += chunk.toString();
          });
          res.on("end", () => {
            clearTimeout(timer);
            const durationMs = Date.now() - startMs;

            if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
              this.logger.warn(
                `[context-offload] HTTP <<< ${path}: ${res.statusCode} ${res.statusMessage} (${durationMs}ms) body=${data.slice(0, 500)}`,
              );
              reject(new Error(`Backend API error ${res.statusCode}: ${data}`));
              return;
            }

            try {
              const parsed = JSON.parse(data) as T;
              this.logger.debug?.(
                `[context-offload] HTTP <<< ${path}: ${res.statusCode} (${durationMs}ms, ${data.length} bytes)`,
              );
              resolve(parsed);
            } catch {
              reject(new Error(`Backend response JSON parse error: ${data.slice(0, 500)}`));
            }
          });
        },
      );

      req.on("error", (err: Error) => {
        clearTimeout(timer);
        const durationMs = Date.now() - startMs;
        const errMsg = err.message;
        const isTimeout = errMsg.includes("timeout");
        this.logger.warn(
          `[context-offload] HTTP !!! ${path}: ${isTimeout ? "TIMEOUT" : "ERROR"} after ${durationMs}ms — ${errMsg}`,
        );
        reject(err);
      });

      req.write(bodyStr);
      req.end();
    });
  }
}
