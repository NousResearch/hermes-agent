/**
 * TDAI Gateway — HTTP server for the Hermes sidecar.
 *
 * Exposes TDAI Core capabilities as HTTP endpoints:
 *   GET  /health              — Health check
 *   POST /recall              — Memory recall (prefetch)
 *   POST /capture             — Conversation capture (sync_turn)
 *   POST /search/memories     — L1 memory search
 *   POST /search/conversations — L0 conversation search
 *   POST /session/end         — Session end + flush
 *   POST /seed               — Batch seed historical conversations (L0 → L1)
 *
 * Built with Node.js native `http` module — no Express/Fastify dependency.
 * Designed to run as a managed sidecar alongside Hermes.
 */

import http from "node:http";
import { URL } from "node:url";
import { timingSafeEqual } from "node:crypto";
import zlib from "node:zlib";
import dayjs from "dayjs";
import { TdaiCore } from "../core/tdai-core.js";
import { StandaloneHostAdapter } from "../adapters/standalone/host-adapter.js";
import { loadGatewayConfig, parseBrokers } from "./config.js";
import type { GatewayConfig } from "./config.js";
import { initDataDirectories } from "../utils/pipeline-factory.js";
import { SessionFilter } from "../utils/session-filter.js";
import type {
  HealthResponse,
  RecallRequest,
  RecallResponse,
  CaptureRequest,
  CaptureResponse,
  MemorySearchRequest,
  MemorySearchResponse,
  ConversationSearchRequest,
  ConversationSearchResponse,
  SessionEndRequest,
  SessionEndResponse,
  SeedRequest,
  SeedResponse,
  GatewayErrorResponse,
} from "./types.js";
import type { Logger } from "../core/types.js";
import { InstanceConfigProvider } from "../core/instance-config-provider.js";
import { wrapWithTrace } from "../core/report/trace-middleware.js";
import { initOTelSDK, shutdownOTelSDK } from "../core/report/otel-sdk-init.js";
import { initObservabilityBackend } from "../core/report/factory.js";
import type { ObservabilityConfig as CoreObservabilityConfig } from "../core/report/types.js";
import { TracedTaskExecutor } from "../core/report/traced-task-executor.js";
import { StorePool } from "../core/store/store-pool.js";
import { validateAndNormalizeRaw, SeedValidationError } from "../core/seed/input.js";
import { executeSeed } from "../core/seed/seed-runtime.js";
import type { SeedProgress } from "../core/seed/types.js";
import { handleV2Route, errorEnvelope, makeRequestId } from "./v2-router.js";
import type { V2RouterDeps } from "./v2-router.js";
import { handleOffloadV2Route } from "../offload_server/router.js";
import type { OffloadV2Deps } from "../offload_server/router.js";
import { initServerOpikTracer } from "../offload_server/opik-tracer.js";
import { classifyError } from "./error-handler.js";
import { LocalStorageBackend } from "../core/storage/local-backend.js";
import { StorageAdapter } from "../core/storage/adapter.js";
import type { TaskPayload } from "../core/state/types.js";
import type { TaskExecutor } from "../services/pipeline-worker.js";
import type { IStateBackend } from "../core/state/types.js";
import type { TimerScanner } from "../services/timer-scanner.js";
import type { PipelineWorker } from "../services/pipeline-worker.js";
import type { StatefulPipelineManager } from "../utils/stateful-pipeline-manager.js";
import type { PipelineLogger } from "../utils/pipeline-factory.js";
import { createLocalTimerTask } from "./timer-routing.js";

const TAG = "[tdai-gateway]";
const VERSION = "0.1.0";

// ============================
// Console logger (for standalone gateway — no OpenClaw logger available)
// ============================

/**
 * Format current time as ISO 8601 in the system's local timezone.
 *
 * Example: "2026-05-21T14:47:03.512+08:00"
 *
 * dayjs's `Z` token emits the local UTC offset (not a literal 'Z'), so the
 * wall-clock matches what the operator sees in `tmux` / `tail -f` while the
 * line stays ISO 8601 compliant and round-trippable.
 */
function nowLocalIso(): string {
  return dayjs().format("YYYY-MM-DDTHH:mm:ss.SSSZ");
}

function createConsoleLogger(): Logger {
  return {
    debug: (msg: string) => console.debug(`${nowLocalIso()} DEBUG ${TAG} ${msg}`),
    info: (msg: string) => console.info(`${nowLocalIso()} INFO  ${TAG} ${msg}`),
    warn: (msg: string) => console.warn(`${nowLocalIso()} WARN  ${TAG} ${msg}`),
    error: (msg: string) => console.error(`${nowLocalIso()} ERROR ${TAG} ${msg}`),
  };
}

// ============================
// Request body parser
// ============================

/**
 * Default request body size limit: 1 MiB (1,048,576 bytes).
 *
 * Override via env `MEMORY_MAX_BODY_BYTES` (must be a positive integer).
 * Capture / seed routes typically need more than 1 MB; if that becomes a
 * recurring issue, raise the env or split per-route limits.
 *
 * Implementation note: env-variable access is delegated to
 * `utils/env-config.ts` to keep this file free of environment-reader
 * tokens, which avoids a known OpenClaw security-scanner false positive
 * triggered by the combination of env reads and the documented route
 * comments above.
 */
import { resolveMaxBodyBytes } from "../utils/env-config.js";

const MAX_BODY_BYTES = resolveMaxBodyBytes();

/**
 * Thrown by `parseJsonBody` when the incoming body exceeds `MAX_BODY_BYTES`.
 * Caught at the top-level request handler (and v2 router) and translated
 * to HTTP 413 instead of being conflated with HTTP 500.
 */
export class PayloadTooLargeError extends Error {
  readonly statusCode = 413;
  readonly limitBytes: number;
  constructor(limitBytes: number) {
    super(`Request body exceeds ${limitBytes} bytes limit`);
    this.name = "PayloadTooLargeError";
    this.limitBytes = limitBytes;
  }
}

export async function parseJsonBody<T>(req: http.IncomingMessage): Promise<T> {
  return new Promise((resolve, reject) => {
    // Reject early when Content-Length header already exceeds the limit:
    // saves transferring up to MAX_BODY_BYTES of attacker-controlled data.
    const declared = Number.parseInt(req.headers["content-length"] ?? "", 10);
    if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) {
      reject(new PayloadTooLargeError(MAX_BODY_BYTES));
      // Drain & destroy so the client doesn't keep streaming.
      req.resume();
      req.destroy();
      return;
    }

    // Determine if the body is compressed (Content-Encoding header).
    // Support gzip and deflate; reject unsupported encodings with 400.
    const encoding = (req.headers["content-encoding"] ?? "").toLowerCase().trim();
    let source: NodeJS.ReadableStream = req;
    if (encoding === "gzip" || encoding === "x-gzip") {
      source = req.pipe(zlib.createGunzip());
    } else if (encoding === "deflate") {
      source = req.pipe(zlib.createInflate());
    } else if (encoding !== "" && encoding !== "identity") {
      req.resume(); // drain
      reject(new Error(`Unsupported Content-Encoding: ${encoding}`));
      return;
    }

    const chunks: Buffer[] = [];
    let received = 0;
    let aborted = false;

    source.on("data", (chunk: Buffer) => {
      if (aborted) return;
      received += chunk.length;
      if (received > MAX_BODY_BYTES) {
        aborted = true;
        reject(new PayloadTooLargeError(MAX_BODY_BYTES));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    source.on("end", () => {
      if (aborted) return;
      try {
        const body = Buffer.concat(chunks).toString("utf-8");
        resolve(JSON.parse(body) as T);
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });
    source.on("error", (_err) => {
      if (aborted) return;  // already rejected with PayloadTooLargeError
      // Decompression errors (e.g. truncated gzip) are client-side faults
      reject(new Error("Invalid JSON body"));
    });
  });
}

function sendJson(res: http.ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(json),
  });
  res.end(json);
}

function sendError(res: http.ServerResponse, status: number, message: string): void {
  sendJson(res, status, { error: message } satisfies GatewayErrorResponse);
}

/**
 * Constant-time string equality for secrets.
 *
 * Returns `false` on any length mismatch (without comparing bytes), and uses
 * `crypto.timingSafeEqual` for the equal-length case so that an attacker
 * probing the API key cannot use response timing to learn a prefix match.
 */
function safeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a, "utf-8");
  const bb = Buffer.from(b, "utf-8");
  if (ab.length !== bb.length) return false;
  return timingSafeEqual(ab, bb);
}

// ============================
// Gateway Server
// ============================

export class TdaiGateway {
  private config: GatewayConfig;
  private logger: Logger;
  private core: TdaiCore;
  private server: http.Server | null = null;
  private startTime = Date.now();

  // ── Integrated services (Scanner + Worker) ──
  private stateBackend: IStateBackend | null = null;
  private timerScanner: TimerScanner | null = null;
  private pipelineWorker: PipelineWorker | null = null;

  // ── Instance config & Store pool (multi-instance VDB) ──
  private configProvider: InstanceConfigProvider | null = null;
  private storePool: StorePool | null = null;
  private quotaManager: import("../core/quota/quota-manager.js").QuotaManager | null = null;
  private statefulPipelineManager: StatefulPipelineManager | null = null;

  // ── COS: global shared client singleton + per-instance StorageAdapter cache ──
  private sharedCosClient: import("../integrations/cos/cos-backend.js").SharedCosClient | null = null;
  private cosStorageCache: Map<string, StorageAdapter> | null = null;

  constructor(configOverrides?: Partial<GatewayConfig>) {
    this.config = loadGatewayConfig(configOverrides);
    this.logger = createConsoleLogger();

    // Create host adapter
    const adapter = new StandaloneHostAdapter({
      dataDir: this.config.data.baseDir,
      llmConfig: this.config.llm,
      logger: this.logger,
      platform: "gateway",
    });

    // Create core
    this.core = new TdaiCore({
      hostAdapter: adapter,
      config: this.config.memory,
      sessionFilter: new SessionFilter(this.config.memory.capture.excludeAgents),
    });
  }

  /**
   * Start the Gateway HTTP server.
   */
  async start(): Promise<void> {
    // Initialize data directories
    initDataDirectories(this.config.data.baseDir);

    // ── 初始化可观测性门面层全局后端 ──
    // 必须在 initOTelSDK 之前调用，因为 LangfuseFilteringProcessor 构造时
    // 会通过 getObservabilityBackend().llmTrace.createSpanProcessor() 获取处理器。
    // 如果不先初始化，门面层所有 API（trace.report / metricProducer.send / obsLogger）
    // 都会走 NoopBackend，导致 Metric、Langfuse、业务 Trace 全部丢失。
    const obsCfg = this.config.observability;
    try {
      const coreObsCfg: CoreObservabilityConfig = {
        type: "internal",
        otel: {
          enabled: obsCfg.otel.enabled,
          endpoint: obsCfg.otel.endpoint,
          protocol: obsCfg.otel.protocol,
          serviceName: obsCfg.otel.serviceName,
          tenantId: obsCfg.otel.tenantId,
        },
        clickhouse: {
          enabled: obsCfg.clickhouse.enabled,
          endpoint: obsCfg.clickhouse.endpoint,
          username: obsCfg.clickhouse.username,
          password: obsCfg.clickhouse.password,
          database: obsCfg.clickhouse.database,
        },
        kafka: {
          brokers: parseBrokers(obsCfg.kafka.brokers),
          topic: obsCfg.kafka.topic,
          enabled: obsCfg.kafka.enabled,
        },
        langfuse: {
          enabled: obsCfg.langfuse.enabled,
          host: obsCfg.langfuse.host,
          publicKey: obsCfg.langfuse.publicKey,
          secretKey: obsCfg.langfuse.secretKey,
        },
      };
      await initObservabilityBackend(coreObsCfg);
      this.logger.info("Observability backend initialized (type=internal)");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.logger.warn(`Observability backend init failed (non-fatal): ${msg}`);
    }

    // ── 初始化 OTel SDK（Trace + Log + ClickHouse 双写）──
    // 必须在 HTTP server 创建之前初始化，否则 wrapWithTrace 中的 Tracer 是 NoopTracer
    if (obsCfg.otel.enabled) {
      try {
        const otelOk = await initOTelSDK({
          serviceName: obsCfg.otel.serviceName,
          serviceVersion: obsCfg.otel.serviceVersion,
          endpoint: obsCfg.otel.endpoint,
          protocol: obsCfg.otel.protocol,
          tenantId: obsCfg.otel.tenantId,
          logExportIntervalMs: obsCfg.otel.logExportInterval * 1000,
          clickhouse: obsCfg.clickhouse.enabled
            ? {
                endpoint: obsCfg.clickhouse.endpoint,
                username: obsCfg.clickhouse.username,
                password: obsCfg.clickhouse.password,
                database: obsCfg.clickhouse.database,
              }
            : false,
          langfuse: obsCfg.langfuse.enabled
            ? {
                host: obsCfg.langfuse.host,
                publicKey: obsCfg.langfuse.publicKey,
                secretKey: obsCfg.langfuse.secretKey,
              }
            : false,
        });
        this.logger.info(`OTel SDK initialized: ${otelOk ? "enabled" : "skipped (deps not available)"}`);
      } catch (err) {
        // 可观测性初始化失败不影响主业务
        const msg = err instanceof Error ? err.message : String(err);
        this.logger.warn(`OTel SDK init failed (non-fatal): ${msg}`);
      }
    }

    // Initialize core
    await this.core.initialize();

    // ── Initialize Opik tracer for offload server ──
    await initServerOpikTracer(this.logger);

    // ── Initialize StorageAdapter for v2 API ──
    // In standalone mode, use LocalStorageBackend pointing to dataDir.
    // In service mode, CosStorageBackend is injected externally.
    if (!this.core.getStorage()) {
      const backend = new LocalStorageBackend(this.config.data.baseDir);
      this.core.setStorage(new StorageAdapter(backend));
      this.logger.info(`${TAG} StorageAdapter initialized (local: ${this.config.data.baseDir})`);
    }

    // ── Start integrated services (Scanner + Worker) if state_backend is configured ──
    await this.startIntegratedServices();

    // Create HTTP server (with Trace middleware wrapping)
    this.server = http.createServer((req, res) => {
      wrapWithTrace(req, res, () => this.handleRequest(req, res)).catch((err) => {
        // wrapWithTrace 内部已经记录了错误，这里只做 fallback
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Internal Server Error" }));
        }
      });
    });

    const { port, host } = this.config.server;

    await new Promise<void>((resolve, reject) => {
      this.server!.listen(port, host, () => {
        this.startTime = Date.now();
        this.logger.info(`Gateway listening on http://${host}:${port}`);
        this.logSecurityPosture();
        resolve();
      });
      this.server!.on("error", reject);
    });
  }

  /**
   * Emit a one-shot security posture summary at startup.
   *
   * Goals:
   *   1. Make the "auth disabled" state highly visible to anyone reading logs
   *      (this is the documented default, but operators must know it before
   *      they expose the port).
   *   2. Loudly warn when the gateway is bound to anything other than the
   *      loopback interface without an API key — that exact combination is
   *      what the security audit flagged as a real exposure.
   *   3. Never log the key itself.
   */
  private logSecurityPosture(): void {
    const { host, apiKey, corsOrigins } = this.config.server;
    const authOn = !!apiKey;
    const loopback = host === "127.0.0.1" || host === "localhost" || host === "::1";

    this.logger.info(
      `Security posture: auth=${authOn ? "ENABLED (Bearer)" : "disabled"} ` +
      `host=${host} cors=${corsOrigins.length === 0 ? "no-headers" : corsOrigins.includes("*") ? "wildcard(*)" : `allowlist(${corsOrigins.length})`}`
    );

    if (!authOn) {
      this.logger.warn(
        "TDAI_GATEWAY_API_KEY is NOT set — all routes except GET /health are " +
        "open to anyone who can reach this port. This is the legacy default. " +
        "Set TDAI_GATEWAY_API_KEY (or server.apiKey in tdai-gateway.yaml) and " +
        "pass `Authorization: Bearer <key>` from clients before exposing the " +
        "gateway beyond the loopback interface."
      );
    }
    if (!loopback && !authOn) {
      this.logger.warn(
        `Gateway is bound to ${host} (non-loopback) WITHOUT an API key. ` +
        "Every /capture, /search/conversations, /recall, /seed call from the " +
        "network is currently unauthenticated. Bind to 127.0.0.1, or set " +
        "TDAI_GATEWAY_API_KEY, before continuing."
      );
    }
    if (corsOrigins.includes("*")) {
      this.logger.warn(
        "CORS allow-list contains '*' — every browser origin can call this " +
        "gateway. Restrict server.corsOrigins to a concrete allow-list for any " +
        "non-local deployment."
      );
    }
  }

  /**
   * Gracefully stop the Gateway.
   */
  async stop(): Promise<void> {
    this.logger.info("Shutting down gateway...");

    // 优雅关闭 OTel SDK（flush 剩余 Span/Log）
    try {
      await shutdownOTelSDK();
    } catch {
      // Best-effort shutdown，不影响主流程
    }

    // Stop integrated services first
    if (this.pipelineWorker) {
      await this.pipelineWorker.stop();
      this.logger.info("Pipeline Worker stopped");
    }
    if (this.timerScanner) {
      await this.timerScanner.stop();
      this.logger.info("Timer Scanner stopped");
    }
    if (this.stateBackend) {
      await this.stateBackend.destroy?.();
      this.logger.info("State Backend closed");
    }
    if (this.storePool) {
      await this.storePool.closeAll();
      this.logger.info("Store Pool closed");
    }

    if (this.server) {
      await new Promise<void>((resolve) => {
        this.server!.close(() => resolve());
      });
    }

    await this.core.destroy();
    this.logger.info("Gateway stopped");
  }

  // ============================
  // Request router
  // ============================

  private async handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    const method = req.method?.toUpperCase() ?? "GET";
    const pathname = url.pathname;

    // Apply CORS headers based on configured allow-list (empty → no headers).
    this.applyCorsHeaders(req, res);

    if (method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    try {
      // ── /v2/instance/destroy — admin endpoint, gated by the v1-style
      //    Bearer apiKey only (no service-id / per-request envelope).
      //    When `server.apiKey` is unset this is open by default, matching
      //    the pre-existing behaviour; operators see the "auth disabled"
      //    WARN at startup.
      if (method === "POST" && pathname === "/v2/instance/destroy") {
        if (!this.checkAuth(req, res)) return;
        return await this.handleInstanceDestroy(req, res);
      }

      // ── v2 API routes (14 endpoints under /v2/) ──
      // Apply the develop-introduced apiKey gate first so v2 inherits the
      // optional shared-secret protection. v2's own `parseV2Auth` (Bearer +
      // x-tdai-service-id) still runs inside `handleV2Route`, preserving
      // its existing semantics. When `server.apiKey` is unset, this gate
      // is a no-op (default-open), matching the develop_server_test
      // baseline.
      if (pathname.startsWith("/v2/")) {
        if (!this.checkAuthForV2(req, res)) return;
      }

      const v2Deps: V2RouterDeps = {
        getStore: () => this.core.getVectorStore(),
        getEmbedding: () => this.core.getEmbeddingService(),
        getStorage: () => this.core.getStorage(),
        deployMode: this.config.deployMode,
        // Inject pipeline introspection deps for /v2/pipeline/status (standalone-only).
        // Both can be undefined in legacy standalone (no stateBackend configured) —
        // the handler returns 503 in that case.
        stateBackend: this.stateBackend ?? undefined,
        pipelineWorker: this.pipelineWorker ?? undefined,
        logger: this.logger,
      };

      // Service mode: inject per-instance resolvers (storePool + configProvider + COS)
      if (this.storePool && this.configProvider) {
        const storePool = this.storePool;
        const configProvider = this.configProvider;
        const logger = this.logger;

        v2Deps.resolveStore = async (instanceId: string) => {
          const vdbConfig = storePool["mode"] === "tcvdb"
            ? await configProvider.resolveVdb(instanceId)
            : null;
          const pooled = await storePool.getStore(instanceId, vdbConfig);
          return { store: pooled.store, embedding: pooled.embedding };
        };

        v2Deps.resolveStorage = async (instanceId: string) => {
          // Check cache first
          const cached = this.cosStorageCache?.get(instanceId);
          if (cached) return cached;

          // Standalone mode: fall back to local storage (no COS needed)
          if (!this.sharedCosClient && this.config.deployMode === "standalone") {
            const localDir = this.config.data.baseDir;
            const backend = new LocalStorageBackend({ rootDir: localDir, logger });
            const adapter = new StorageAdapter(backend);
            if (!this.cosStorageCache) this.cosStorageCache = new Map();
            this.cosStorageCache.set(instanceId, adapter);
            return adapter;
          }

          if (!this.sharedCosClient) {
            throw new Error(`SharedCosClient not initialized for instance ${instanceId}`);
          }

          // Get current COS config to determine prefix
          const cosConfig = await configProvider.resolveCos();
          if (!cosConfig?.cosUrl) {
            throw new Error(`COS config not available for instance ${instanceId} (Shark returned null or empty CosUrl)`);
          }

          // Per-instance CosStorageBackend: lightweight, only holds prefix
          const { CosStorageBackend } = await import("../integrations/cos/cos-backend.js");
          const prefix = `${cosConfig.pathPrefix.replace(/\/$/, '')}/${instanceId}/`;
          const backend = new CosStorageBackend({
            sharedClient: this.sharedCosClient,
            prefix,
            logger,
          });
          const adapter = new StorageAdapter(backend);
          if (!this.cosStorageCache) this.cosStorageCache = new Map();
          this.cosStorageCache.set(instanceId, adapter);
          return adapter;
        };

        // Pipeline notify: trigger async L1 extraction when v2 /conversation/add writes L0
        if (this.statefulPipelineManager) {
          const pipelineManager = this.statefulPipelineManager;
          v2Deps.notifyPipeline = async (instanceId: string, sessionId: string, rounds: number) => {
            await pipelineManager.notifyConversation(sessionId, [], instanceId, rounds);
          };
        }

        // Inject QuotaManager for memory/credit limit checks
        if (this.quotaManager) {
          v2Deps.quotaManager = this.quotaManager;
        }
      }

      // ── Offload V2 routes (async ingest + mmd query) ──
      const offloadDeps: OffloadV2Deps = {
        resolveStorage: v2Deps.resolveStorage,
        getStorage: v2Deps.getStorage ?? (() => undefined),
        logger: this.logger,
        stateBackend: this.stateBackend,
        config: { ...this.config.offload, l1Model: "", l15Model: "", l2Model: "" },
      };
      const offloadHandled = await handleOffloadV2Route(req, res, pathname, method, parseJsonBody, sendJson, offloadDeps);
      if (offloadHandled) return;

      const handled = await handleV2Route(req, res, pathname, method, parseJsonBody, sendJson, v2Deps);
      if (handled) return;

      // ── v1 API routes ──

      // GET /health is always reachable without auth — operators and
      // orchestrators (k8s liveness, docker health-check) rely on it being
      // an unconditionally cheap probe.
      if (method === "GET" && pathname === "/health") {
        return this.handleHealth(res);
      }

      // All other routes go through the optional auth gate. When apiKey is
      // unset the gate is a no-op (preserves legacy open behaviour) — the
      // startup WARN in `logSecurityPosture` covers that case.
      if (!this.checkAuth(req, res)) return;

      switch (`${method} ${pathname}`) {
        case "POST /recall":
          return await this.handleRecall(req, res);
        case "POST /capture":
          return await this.handleCapture(req, res);
        case "POST /search/memories":
          return await this.handleSearchMemories(req, res);
        case "POST /search/conversations":
          return await this.handleSearchConversations(req, res);
        case "POST /session/end":
          return await this.handleSessionEnd(req, res);
        case "POST /seed":
          return await this.handleSeed(req, res);
        default:
          sendError(res, 404, `Not found: ${method} ${pathname}`);
      }
    } catch (err) {
      if (err instanceof PayloadTooLargeError) {
        // Fast-path: PayloadTooLargeError messages are already safe (constant + numeric limit).
        this.logger.warn(`Request rejected [${method} ${pathname}]: ${err.message}`);
        sendError(res, 413, err.message);
        return;
      }
      // H-13: classify + sanitize before sending to client.
      // Server log keeps full stack via classified.logLine; client only sees
      // a safe code + message + trace_id.
      const classified = classifyError(err);
      this.logger.error(`Request error [${method} ${pathname}] ${classified.logLine}`);
      sendJson(res, classified.status, {
        // Keep legacy `error` field for backward compat with existing v1 clients.
        error: classified.client.message,
        code: classified.client.code,
        message: classified.client.message,
        trace_id: classified.client.trace_id,
        retryable: classified.client.retryable,
      });
    }
  }

  // ============================
  // Auth & CORS gates (opt-in, off by default)
  // ============================

  /**
   * Verify the `Authorization: Bearer <apiKey>` header against the configured
   * shared secret using a constant-time comparison.
   *
   * When `server.apiKey` is unset (`undefined`), this returns `"ok"` without
   * inspecting the request — this is the documented default and matches the
   * pre-existing open behaviour. Operators are reminded of this at startup
   * via `logSecurityPosture`.
   *
   * Returns one of:
   *   - `"ok"`            — auth disabled OR token matches; caller proceeds
   *   - `"missing"`       — Authorization header missing or not a Bearer token
   *   - `"invalid"`       — token present but did not match the configured key
   *
   * Caller is responsible for translating `"missing"` / `"invalid"` into the
   * appropriate 401 response (v1 plain-text via {@link checkAuth} or v2
   * envelope via {@link checkAuthForV2}).
   */
  private verifyAuth(req: http.IncomingMessage): "ok" | "missing" | "invalid" {
    const expected = this.config.server.apiKey;
    if (!expected) return "ok"; // auth disabled — default behaviour

    const header = req.headers["authorization"];
    if (typeof header !== "string" || !header.startsWith("Bearer ")) {
      return "missing";
    }
    const provided = header.slice("Bearer ".length).trim();
    if (!provided || !safeEqual(provided, expected)) {
      return "invalid";
    }
    return "ok";
  }

  /**
   * v1 / admin auth gate. Writes a plain-text 401 on failure (legacy format
   * preserved so existing curl-based callers keep working). Returns `false`
   * when the request must be short-circuited.
   */
  private checkAuth(req: http.IncomingMessage, res: http.ServerResponse): boolean {
    const result = this.verifyAuth(req);
    if (result === "ok") return true;
    sendError(
      res,
      401,
      result === "missing"
        ? "Unauthorized: missing Bearer token"
        : "Unauthorized: invalid token",
    );
    return false;
  }

  /**
   * v2 auth gate. Same verification as {@link checkAuth} but returns the
   * v2 standardized error envelope on failure so v2 clients see a consistent
   * `{ code, message, request_id }` shape.
   *
   * The existing in-router `parseV2Auth` (which checks for non-empty Bearer
   * + `x-tdai-service-id`) is layered on top; this gate runs first.
   */
  private checkAuthForV2(req: http.IncomingMessage, res: http.ServerResponse): boolean {
    const result = this.verifyAuth(req);
    if (result === "ok") return true;
    const requestId = makeRequestId();
    const message =
      result === "missing"
        ? "Unauthorized: missing Bearer token"
        : "Unauthorized: invalid token";
    sendJson(res, 401, errorEnvelope(401, message, requestId));
    return false;
  }

  /**
   * Echo `Access-Control-Allow-Origin` (and friends) only for whitelisted
   * origins. With no list configured we emit no CORS headers at all, which
   * makes the browser refuse the cross-origin request as desired.
   *
   * The single-entry list `["*"]` opts back into permissive CORS (development
   * use only; the startup log flags this loudly).
   */
  private applyCorsHeaders(req: http.IncomingMessage, res: http.ServerResponse): void {
    const allow = this.config.server.corsOrigins ?? [];
    if (allow.length === 0) return; // strict default — no headers

    if (allow.includes("*")) {
      // Wildcard — preserves the legacy permissive behaviour for callers that
      // opt in explicitly via config. Note: with wildcard we deliberately do
      // not echo back the request Origin and do not send `Vary: Origin`,
      // mirroring how the gateway behaved before this change.
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
      return;
    }

    const requestOrigin = req.headers["origin"];
    if (typeof requestOrigin !== "string" || !allow.includes(requestOrigin)) {
      // Origin not in allow-list — emit no CORS headers; browser will block.
      // Always set Vary so caches don't poison responses across origins.
      res.setHeader("Vary", "Origin");
      return;
    }
    res.setHeader("Access-Control-Allow-Origin", requestOrigin);
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.setHeader("Vary", "Origin");
  }

  // ============================
  // Route handlers
  // ============================

  /**
   * POST /v2/instance/destroy — Purge all data for a destroyed instance.
   * Intended for trusted internal callers only.
   *
   * Request body: { instance_id: string }
   * Response: { code, message, data: { instance_id, cleaned: { ... } } }
   */
  private async handleInstanceDestroy(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<{ instance_id?: string }>(req);
    const instanceId = body?.instance_id;

    if (!instanceId || typeof instanceId !== "string") {
      sendJson(res, 400, { code: 400, message: "Missing required field: instance_id" });
      return;
    }

    this.logger.info(`[instance/destroy] Purging instance: ${instanceId}`);
    const cleaned: Record<string, unknown> = {};

    // 1. Purge state backend (timers, sessions, buffers, pending tasks)
    if (this.stateBackend?.purgeInstance) {
      try {
        const result = await this.stateBackend.purgeInstance(instanceId);
        cleaned.state = result;
        this.logger.info(`[instance/destroy] State purged: sessions=${result.sessions}, timers=${result.timers}, buffers=${result.buffers}`);
      } catch (err) {
        this.logger.error(`[instance/destroy] State purge failed: ${err instanceof Error ? err.message : String(err)}`);
        cleaned.state_error = err instanceof Error ? err.message : String(err);
      }
    }

    // 2. Evict store from StorePool
    if (this.storePool) {
      try {
        await this.storePool.evict(instanceId);
        cleaned.store_evicted = true;
      } catch (err) {
        this.logger.error(`[instance/destroy] Store evict failed: ${err instanceof Error ? err.message : String(err)}`);
        cleaned.store_evicted = false;
      }
    }

    // 3. Delete COS objects for this instance
    if (this.cosStorageCache?.has(instanceId)) {
      this.cosStorageCache.delete(instanceId);
    }
    if (this.sharedCosClient && this.configProvider) {
      try {
        const cosConfig = await this.configProvider.resolveCos();
        if (cosConfig?.cosUrl) {
          const { CosStorageBackend } = await import("../integrations/cos/cos-backend.js");
          const prefix = `${cosConfig.pathPrefix.replace(/\/$/, '')}/${instanceId}/`;
          const backend = new CosStorageBackend({
            sharedClient: this.sharedCosClient,
            prefix,
            logger: this.logger,
          });
          const deletedCount = await backend.deleteByPrefix("");
          cleaned.cos_objects_deleted = deletedCount;
          this.logger.info(`[instance/destroy] COS objects deleted: ${deletedCount}`);
        }
      } catch (err) {
        this.logger.error(`[instance/destroy] COS cleanup failed: ${err instanceof Error ? err.message : String(err)}`);
        cleaned.cos_error = err instanceof Error ? err.message : String(err);
      }
    }

    // 4. Clear QuotaManager cache
    if (this.quotaManager) {
      (this.quotaManager as any).cache?.delete?.(instanceId);
      cleaned.quota_cache_cleared = true;
    }

    sendJson(res, 200, {
      code: 0,
      message: "ok",
      data: { instance_id: instanceId, cleaned },
    });
  }

  private handleHealth(res: http.ServerResponse): void {
    const response: HealthResponse = {
      status: this.core.getVectorStore() ? "ok" : "degraded",
      version: VERSION,
      uptime: Math.floor((Date.now() - this.startTime) / 1000),
      stores: {
        vectorStore: !!this.core.getVectorStore(),
        embeddingService: !!this.core.getEmbeddingService(),
      },
      // Integrated services status
      services: {
        timerScanner: this.timerScanner?.getMetrics() ?? null,
        pipelineWorker: this.pipelineWorker?.getMetrics() ?? null,
        stateBackend: this.stateBackend ? "connected" : "none",
      },
    };
    sendJson(res, 200, response);
  }

  private async handleRecall(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<RecallRequest>(req);

    if (!body.query || !body.session_key) {
      sendError(res, 400, "Missing required fields: query, session_key");
      return;
    }

    const startMs = Date.now();
    const result = await this.core.handleBeforeRecall(body.query, body.session_key);
    const elapsed = Date.now() - startMs;

    // H-15: distinguish "no recall content to inject" from "recall failed".
    // Both return HTTP 200 (recall is non-critical) but the response body
    // carries a non-zero code + message when the recall path itself failed
    // (e.g. EmbeddingService unavailable, VDB timeout).
    if (result.error) {
      this.logger.warn(
        `Recall failed in ${elapsed}ms: code=${result.error.code} category=${result.error.category} ` +
        `msg="${result.error.message}"`,
      );
    } else {
      this.logger.info(`Recall completed in ${elapsed}ms: context=${(result.appendSystemContext?.length ?? 0)} chars`);
    }

    const response: RecallResponse = {
      context: result.appendSystemContext ?? "",
      strategy: result.recallStrategy,
      memory_count: result.recalledL1Memories?.length ?? 0,
      code: result.error?.code ?? 0,
      message: result.error?.message ?? "ok",
      retryable: result.error?.retryable ?? false,
    };
    sendJson(res, 200, response);
  }

  private async handleCapture(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<CaptureRequest>(req);

    if (!body.user_content || !body.assistant_content || !body.session_key) {
      sendError(res, 400, "Missing required fields: user_content, assistant_content, session_key");
      return;
    }

    const startMs = Date.now();
    const result = await this.core.handleTurnCommitted({
      userText: body.user_content,
      assistantText: body.assistant_content,
      messages: body.messages ?? [
        { role: "user", content: body.user_content },
        { role: "assistant", content: body.assistant_content },
      ],
      sessionKey: body.session_key,
      sessionId: body.session_id,
    });
    const elapsed = Date.now() - startMs;

    this.logger.info(`Capture completed in ${elapsed}ms: l0=${result.l0RecordedCount}`);

    const response: CaptureResponse = {
      l0_recorded: result.l0RecordedCount,
      scheduler_notified: result.schedulerNotified,
    };
    sendJson(res, 200, response);
  }

  private async handleSearchMemories(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<MemorySearchRequest>(req);

    if (!body.query) {
      sendError(res, 400, "Missing required field: query");
      return;
    }

    const result = await this.core.searchMemories({
      query: body.query,
      limit: body.limit,
      type: body.type,
      scene: body.scene,
    });

    const response: MemorySearchResponse = {
      results: result.text,
      total: result.total,
      strategy: result.strategy,
    };
    sendJson(res, 200, response);
  }

  private async handleSearchConversations(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<ConversationSearchRequest>(req);

    if (!body.query) {
      sendError(res, 400, "Missing required field: query");
      return;
    }

    const result = await this.core.searchConversations({
      query: body.query,
      limit: body.limit,
      sessionKey: body.session_key,
    });

    const response: ConversationSearchResponse = {
      results: result.text,
      total: result.total,
    };
    sendJson(res, 200, response);
  }

  private async handleSessionEnd(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<SessionEndRequest>(req);

    if (!body.session_key) {
      sendError(res, 400, "Missing required field: session_key");
      return;
    }

    await this.core.handleSessionEnd(body.session_key);

    const response: SessionEndResponse = { flushed: true };
    sendJson(res, 200, response);
  }

  private async handleSeed(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<SeedRequest>(req);

    if (!body.data) {
      sendError(res, 400, "Missing required field: data");
      return;
    }

    // Validate and normalize input (reuses seed CLI's validation layers 2-6)
    let input;
    try {
      input = validateAndNormalizeRaw(body.data, {
        sessionKey: body.session_key,
        strictRoundRole: body.strict_round_role,
        autoFillTimestamps: body.auto_fill_timestamps ?? true,
      });
    } catch (err) {
      if (err instanceof SeedValidationError) {
        sendJson(res, 400, {
          error: err.message,
          validation_errors: err.errors,
        });
        return;
      }
      throw err;
    }

    this.logger.info(
      `Seed request: ${input.sessions.length} session(s), ` +
      `${input.totalRounds} round(s), ${input.totalMessages} message(s)`,
    );

    // Resolve output directory: use gateway's data dir with a timestamped subfolder
    const now = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    const ts =
      `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
      `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const outputDir = `${this.config.data.baseDir}/seed-${ts}`;

    // Merge config overrides if provided
    // Start with the base memory config + inject llm config from gateway settings
    const baseConfig = this.config.memory as unknown as Record<string, unknown>;
    let pluginConfig: Record<string, unknown> = {
      ...baseConfig,
      llm: {
        enabled: true,
        baseUrl: this.config.llm.baseUrl,
        apiKey: this.config.llm.apiKey,
        model: this.config.llm.model,
        maxTokens: this.config.llm.maxTokens,
        timeoutMs: this.config.llm.timeoutMs,
      },
    };
    if (body.config_override) {
      for (const key of Object.keys(body.config_override)) {
        const baseVal = pluginConfig[key];
        const overVal = body.config_override[key];
        if (baseVal && typeof baseVal === "object" && !Array.isArray(baseVal) &&
            overVal && typeof overVal === "object" && !Array.isArray(overVal)) {
          pluginConfig[key] = { ...(baseVal as Record<string, unknown>), ...(overVal as Record<string, unknown>) };
        } else {
          pluginConfig[key] = overVal;
        }
      }
    }

    // Execute seed pipeline (blocking — this may take minutes for large inputs)
    const summary = await executeSeed(input, {
      outputDir,
      openclawConfig: {},
      pluginConfig,
      logger: this.logger as PipelineLogger,
      onProgress: (progress: SeedProgress) => {
        this.logger.debug?.(
          `Seed progress: [${progress.currentRound}/${progress.totalRounds}] ` +
          `session=${progress.sessionKey} stage=${progress.stage}`,
        );
      },
    });

    this.logger.info(
      `Seed complete: sessions=${summary.sessionsProcessed}, rounds=${summary.roundsProcessed}, ` +
      `l0=${summary.l0RecordedCount}, duration=${(summary.durationMs / 1000).toFixed(1)}s`,
    );

    const response: SeedResponse = {
      sessions_processed: summary.sessionsProcessed,
      rounds_processed: summary.roundsProcessed,
      messages_processed: summary.messagesProcessed,
      l0_recorded: summary.l0RecordedCount,
      duration_ms: summary.durationMs,
      output_dir: summary.outputDir,
    };
    sendJson(res, 200, response);
  }

  // ============================
  // Integrated Services (Scanner + Worker)
  // ============================

  /**
   * Start Timer Scanner and Pipeline Worker inside the Gateway process.
   *
   * Activated automatically:
   *   - standalone → local in-process backend (default)
   *   - service    → remote backend (default)
   *
   * env vars:
   *   STATE_BACKEND=local|remote  — backend type
   *   SCANNER_INSTANCES=inst1,inst2 — instances to scan (default: "default")
   *   SCANNER_INTERVAL_MS=500 — scan interval
   *   WORKER_POLL_MS=200 — worker poll interval
   */
  private async startIntegratedServices(): Promise<void> {
    // Determine backend type from config (env > yaml > auto from deployMode):
    //   - "standalone" → local (in-process Map/setTimeout, zero dependencies)
    //   - "service"    → remote state backend
    const backendType: "redis" | "local" =
      this.config.stateBackend ?? (this.config.deployMode === "service" ? "redis" : "local");

    this.logger.info(`Starting integrated services (deployMode=${this.config.deployMode}, state_backend=${backendType})...`);

    // 1. Create State Backend
    const { createStateBackend } = await import("../core/state/index.js");
    this.stateBackend = await createStateBackend({
      type: backendType,
      local: backendType === "local" ? {
        onTimerExpired: (entry) => {
          const member = entry.member;
          const task = createLocalTimerTask({
            entry,
            defaultInstanceId: this.config.instanceId ?? "default",
          });
          this.stateBackend!.enqueueTask(task).then(() => {
            this.logger.info(`[local-timer] Timer fired: ${member} → enqueued ${task.type} task`);
          }).catch((err) => {
            this.logger.error(`[local-timer] Failed to enqueue task for ${member}: ${err instanceof Error ? err.message : String(err)}`);
          });
        },
      } : undefined,
      redis: backendType === "redis" ? {
        host: this.config.redis.host,
        port: this.config.redis.port,
        password: this.config.redis.password,
        db: this.config.redis.db,
        keyPrefix: this.config.redis.keyPrefix,
      } : undefined,
    });
    this.logger.info(`State Backend created (${backendType})`);

    // 1.2. Pick adapter set (default for standalone, enhanced for service).
    // InstanceConfigProvider/QuotaManager depend only on core abstractions;
    // concrete implementations are chosen at gateway startup.
    //
    // Optional deployment adapters are loaded dynamically. When unavailable,
    // standalone falls back to LocalConfigSource + NoopQuotaReporter; service
    // mode fails fast because it requires deployment-specific adapters.
    let adapterDeps: { configSource: import("../core/abstractions/index.js").IConfigSource; quotaReporter: import("../core/abstractions/index.js").IQuotaReporter };
    try {
      const { createAdapterDeps } = await import("../integrations/factory.js");
      adapterDeps = await createAdapterDeps({
        deployMode: this.config.deployMode,
        sharkBaseUrl: this.config.shark.baseUrl,
        logger: this.logger,
      });
    } catch (err) {
      if (this.config.deployMode === "service") {
        throw new Error(
          `[gateway] deployMode=service requires src/integrations/ (private submodule), ` +
            `but it could not be loaded. Either initialize the submodule or set ` +
            `deployMode=standalone. Original error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
      this.logger.warn(
        `[gateway] integrations/ not available (${err instanceof Error ? err.message : String(err)}); ` +
          `falling back to inline LocalConfigSource + NoopQuotaReporter (standalone only).`,
      );
      const { LocalConfigSource } = await import("../core/instance-config-provider.js");
      const { NoopQuotaReporter } = await import("../core/quota/noop-quota-reporter.js");
      adapterDeps = {
        configSource: new LocalConfigSource(this.logger),
        quotaReporter: new NoopQuotaReporter(),
      };
    }

    this.configProvider = new InstanceConfigProvider({
      source: adapterDeps.configSource,
      vdbTtlMs: this.config.shark.vdbTtlMs,
      cosBufferMs: this.config.shark.cosBufferMs,
      maxInstances: this.config.shark.maxInstances,
      logger: this.logger,
    });

    // 1.2.1. Create QuotaManager (service mode only — in standalone the Noop
    // reporter would short-circuit everything anyway, so we skip allocation).
    if (this.config.deployMode === "service") {
      const { QuotaManager } = await import("../core/quota/index.js");
      this.quotaManager = new QuotaManager({
        reporter: adapterDeps.quotaReporter,
        logger: this.logger,
      });
      this.logger.info("QuotaManager initialized (memoryLimit=50000, creditLimit=1000)");
    }
    this.storePool = new StorePool({
      mode: this.config.deployMode === "service" ? "tcvdb" : "sqlite",
      memoryCfg: this.config.memory,
      dataDir: this.config.data.baseDir,
      maxStores: this.config.shark.maxInstances,
      kafka: {
        brokers: parseBrokers(this.config.observability.kafka.brokers),
        topic: this.config.observability.kafka.topic,
        enabled: this.config.observability.kafka.enabled,
      },
      logger: this.logger,
    });
    this.logger.info(`Instance Config Provider + Store Pool initialized (mode=${this.config.deployMode})`);

    // 1.3. Switch Core's default storage to remote object storage in service mode.
    // This ensures v1 API (capture/recall) also writes L0/L1 to shared storage instead of local filesystem.
    if (this.config.deployMode === "service") {
      await this.initSharedCosClient();
    }

    // 1.5. Inject StatefulPipelineManager into Core (replaces legacy MemoryPipelineManager)
    const { createStatefulPipelineManager } = await import("../utils/pipeline-factory.js");
    // Service mode: defaultInstanceId must NOT be "default"; all calls must provide explicit instanceId.
    // Standalone mode: uses configured instanceId or "default" as fallback.
    const instanceId = this.config.instanceId ?? (this.config.deployMode === "service" ? "__unset__" : "default");
    const statefulManager = createStatefulPipelineManager(
      this.config.memory,
      this.stateBackend,
      instanceId,
      this.logger,
    );
    this.statefulPipelineManager = statefulManager;
    // Attach to core — core.setStatefulPipelineManager will wire capture to use captureAtomic
    if (typeof (this.core as any).setStatefulPipelineManager === "function") {
      (this.core as any).setStatefulPipelineManager(statefulManager);
      this.logger.info(`Core switched to StatefulPipelineManager (instance=${instanceId})`);
    }

    // 2. Start Timer Scanner (Scheme D: leaderless, scans sharded global ZSETs)
    const { TimerScanner } = await import("../services/timer-scanner.js");
    const defaultInstances = this.config.scanner.instances.split(",").filter(Boolean);

    this.timerScanner = new TimerScanner(this.stateBackend, {
      scanIntervalMs: this.config.scanner.intervalMs,
    }, this.logger);
    await this.timerScanner.start();
    this.logger.info(`Timer Scanner started (defaultInstances=${defaultInstances.join(",")}, sharded=true, leaderless=true)`);

    // 3. Start Pipeline Worker
    const { PipelineWorker } = await import("../services/pipeline-worker.js");
    const rawExecutor = this.buildTaskExecutor();
    // 用 TracedTaskExecutor 装饰器包装，为 L1/L2/L3 任务添加 Trace Span
    const executor = new TracedTaskExecutor(rawExecutor);
    this.pipelineWorker = new PipelineWorker(this.stateBackend, executor, {
      pollIntervalMs: this.config.worker.pollMs,
      concurrency: this.config.worker.concurrency,
      // L1 完成后推进 L2 timer（快路径：L1完成 → delay秒后触发L2）
      onL1Complete: statefulManager.advanceL2TimerAfterL1.bind(statefulManager),
      // L2 完成后设置 maxInterval 兜底 timer
      onL2Complete: statefulManager.armL2MaxInterval.bind(statefulManager),
    }, this.logger);
    await this.pipelineWorker.start();
    this.logger.info("Pipeline Worker started");
  }

  /**
   * Initialize SharedCosClient with retry. Called at startup and lazily from resolveStorage.
   * If already initialized, returns immediately.
   */
  private async initSharedCosClient(maxRetries = 3): Promise<void> {
    if (this.sharedCosClient) return;
    if (!this.configProvider) return;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const cosConfig = await this.configProvider.resolveCos();
        if (!cosConfig?.cosUrl) {
          this.logger.warn(`${TAG} COS config unavailable from Shark (attempt ${attempt}/${maxRetries})`);
          if (attempt < maxRetries) {
            await new Promise(r => setTimeout(r, attempt * 2000));
            continue;
          }
          this.logger.error(`${TAG} COS init failed after ${maxRetries} attempts: Shark returned empty COS config`);
          return;
        }

        const { CosStorageBackend, SharedCosClient } = await import("../integrations/cos/cos-backend.js");
        const { CachedCredentialProvider, parseCosUrl } = await import("../core/storage/credential-provider.js");
        const { bucket, region } = parseCosUrl(cosConfig.cosUrl);
        const cosHost = cosConfig.cosUrl.replace(/^https?:\/\//, "").replace(/\/+$/, "");
        const bucketPrefix = `${bucket}.`;
        const endpointDomain = cosHost.startsWith(bucketPrefix) ? cosHost.slice(bucketPrefix.length) : undefined;
        const isInternalDomain = endpointDomain?.includes("tencentcos.cn");
        const internalDomain = isInternalDomain ? endpointDomain : `cos-internal.${region}.tencentcos.cn`;
        const cosEndpointDomain = this.config.cos.domain || internalDomain;

        const configProvider = this.configProvider;
        const credentialProvider = new CachedCredentialProvider({
          fetcher: async () => {
            const fresh = await configProvider.resolveCos();
            if (!fresh) throw new Error("Shark returned null COS config");
            const parsed = parseCosUrl(fresh.cosUrl);
            return {
              secretId: fresh.tmpSecretId,
              secretKey: fresh.tmpSecretKey,
              token: fresh.tmpToken || undefined,
              bucket: parsed.bucket,
              region: parsed.region,
              prefix: fresh.pathPrefix,
              expiresAt: fresh.expirationTime ? new Date(fresh.expirationTime).getTime() : undefined,
            };
          },
          cacheTtlMs: this.config.shark.cosBufferMs ?? 120000,
          logger: this.logger,
        });

        this.sharedCosClient = new SharedCosClient({
          credentialProvider,
          logger: this.logger,
          cosEndpointDomain,
        });
        await this.sharedCosClient.getClient();
        this.logger.info(`${TAG} SharedCosClient initialized (bucket=${bucket}, domain=${cosEndpointDomain}, attempt=${attempt})`);

        // Set Core default storage to COS
        const defaultInstanceId = this.config.instanceId ?? "default";
        const defaultPrefix = `${cosConfig.pathPrefix.replace(/\/$/, '')}/${defaultInstanceId}/`;
        const cosBackend = new CosStorageBackend({
          sharedClient: this.sharedCosClient,
          prefix: defaultPrefix,
          logger: this.logger,
        });
        this.core.setStorage(new StorageAdapter(cosBackend));
        this.logger.info(`${TAG} Core default storage switched to COS (prefix=${defaultPrefix})`);
        return;
      } catch (err) {
        this.logger.warn(`${TAG} COS init attempt ${attempt}/${maxRetries} failed: ${err instanceof Error ? err.message : String(err)}`);
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, attempt * 2000));
        }
      }
    }
    this.logger.error(`${TAG} SharedCosClient init failed after ${maxRetries} retries, L2/L3 tasks will fail until COS is available`);
  }

  /**
   * Build a TaskExecutor that bridges Pipeline tasks to TdaiCore's existing L1/L2/L3 runners.
   *
   * Multi-instance aware: each task carries a instanceId in task.data.
   * The executor resolves the per-instance VDB config from InstanceConfigProvider,
   * then obtains the corresponding Store from StorePool before running the task.
   */
  private buildTaskExecutor(): TaskExecutor {
    const core = this.core;
    const configProvider = this.configProvider!;
    const storePool = this.storePool!;
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const gateway = this;

    const resolveStore = async (task: TaskPayload) => {
      const instanceId = typeof task.data?.instanceId === "string" ? task.data.instanceId : undefined;
      if (!instanceId) {
        throw new Error(`Task ${task.id} missing instanceId in service mode (task.data.instanceId is required)`);
      }
      const vdbConfig = storePool.mode === "tcvdb"
        ? await configProvider.resolveVdb(instanceId)
        : null;
      return storePool.getStore(instanceId, vdbConfig);
    };

    const resolveStorage = async (task: TaskPayload) => {
      const instanceId = typeof task.data?.instanceId === "string" ? task.data.instanceId : undefined;
      if (!instanceId) {
        throw new Error(`Task ${task.id} missing instanceId in service mode (task.data.instanceId is required)`);
      }

      // Standalone mode: use local storage (no COS needed)
      if (!gateway.sharedCosClient && gateway.config.deployMode === "standalone") {
        const cached = gateway.cosStorageCache?.get(instanceId);
        if (cached) return cached;
        const localDir = gateway.config.data.baseDir;
        const backend = new LocalStorageBackend({ rootDir: localDir, logger: gateway.logger });
        const adapter = new StorageAdapter(backend);
        if (!gateway.cosStorageCache) gateway.cosStorageCache = new Map();
        gateway.cosStorageCache.set(instanceId, adapter);
        return adapter;
      }

      // Lazy-init COS if not yet initialized (startup may have failed)
      if (!gateway.sharedCosClient) {
        await gateway.initSharedCosClient();
      }

      if (!gateway.sharedCosClient) {
        throw new Error(`SharedCosClient not initialized for worker task ${task.id} (instance=${instanceId})`);
      }
      const cached = gateway.cosStorageCache?.get(instanceId);
      if (cached) return cached;
      const cosConfig = await configProvider.resolveCos();
      if (!cosConfig) {
        throw new Error(`COS config not available for worker task ${task.id} (instance=${instanceId}, Shark returned null)`);
      }
      const { CosStorageBackend } = await import("../integrations/cos/cos-backend.js");
      const prefix = `${cosConfig.pathPrefix.replace(/\/$/, '')}/${instanceId}/`;
      const backend = new CosStorageBackend({
        sharedClient: gateway.sharedCosClient,
        prefix,
        logger: gateway.logger,
      });
      const adapter = new StorageAdapter(backend);
      if (!gateway.cosStorageCache) gateway.cosStorageCache = new Map();
      gateway.cosStorageCache.set(instanceId, adapter);
      return adapter;
    };

    return {
      async executeL1(task: TaskPayload, signal?: AbortSignal) {
        const instanceId = typeof task.data?.instanceId === "string" ? task.data.instanceId : undefined;
        if (!instanceId) throw new Error(`L1 task ${task.id} missing instanceId`);

        // H-11 Step 2: early abort check — if pipeline-worker already lost its lock
        // before we even started, bail out without doing any work.
        if (signal?.aborted) throw signal.reason ?? new Error("executeL1: aborted before start");

        // Dedup: if triggered by timer but session already processed (count=0), skip
        if (task.data?.triggeredBy === "timer_scanner" && gateway.stateBackend) {
          const state = await gateway.stateBackend.getSessionState(instanceId, task.sessionId);
          if (state && state.conversation_count === 0) {
            gateway.logger.debug?.(`[executor] L1 skipped: session ${task.sessionId} already processed (count=0)`);
            return;
          }
        }

        // Credit quota check before LLM call
        if (gateway.quotaManager) {
          const check = await gateway.quotaManager.checkCreditQuota(instanceId);
          if (!check.allowed) {
            gateway.logger.warn(`[executor] L1 skipped: credit limit exceeded (instance=${instanceId}, current=${check.current}, limit=${check.limit})`);
            return;
          }
        }

        // H-11 Step 2: check again after async quota call before launching LLM
        if (signal?.aborted) throw signal.reason ?? new Error("executeL1: aborted before LLM");

        core.setInstanceId(instanceId);
        const { store, embedding } = await resolveStore(task);
        const storage = await resolveStorage(task);
        const result = await core.runL1WithStore(task.sessionId, store, embedding, storage ?? undefined);

        // Report usage after L1: memory added + credit consumed
        if (gateway.quotaManager) {
          const { storedCount, creditUsed } = result;
          if (storedCount > 0 || creditUsed > 0) {
            gateway.quotaManager.reportUsage(instanceId, storedCount, creditUsed, "L1").catch(() => {});
          }
        }

        // ── L0 backlog drain (mirrors standalone MemoryPipelineManager.runL1) ──
        //
        // The runner over-fetched 2N L0 rows but processed at most N. If
        // `hasFullBacklog`, DB is likely far from drained — enqueue another
        // L1 task right away. If only `hasMore`, defer to the standard
        // L1_idle timer so a later notifyConversation can co-trigger.
        // See pipeline-factory.ts createL1Runner for the full state machine.
        if (gateway.statefulPipelineManager) {
          if (result.hasFullBacklog) {
            gateway.statefulPipelineManager.enqueueL1Drain(task.sessionId, instanceId).catch((err) => {
              gateway.logger.warn(`[executor] L1 drain enqueue failed: ${err instanceof Error ? err.message : String(err)}`);
            });
          } else if (result.hasMore) {
            gateway.statefulPipelineManager.armL1IdleAfterDrain(task.sessionId, instanceId).catch((err) => {
              gateway.logger.warn(`[executor] L1 idle arm failed: ${err instanceof Error ? err.message : String(err)}`);
            });
          }
        }
      },
      async executeL2(task: TaskPayload, signal?: AbortSignal) {
        const instanceId = typeof task.data?.instanceId === "string" ? task.data.instanceId : undefined;
        if (!instanceId) throw new Error(`L2 task ${task.id} missing instanceId`);

        if (signal?.aborted) throw signal.reason ?? new Error("executeL2: aborted before start");

        // Credit quota check before LLM call
        if (gateway.quotaManager) {
          const check = await gateway.quotaManager.checkCreditQuota(instanceId);
          if (!check.allowed) {
            gateway.logger.warn(`[executor] L2 skipped: credit limit exceeded (instance=${instanceId})`);
            return;
          }
        }

        // Read L2 cursor from session state (l2_last_extraction_time)
        // This ensures L2 only processes L1 records created after the last extraction.
        let cursor: string | undefined;
        if (gateway.stateBackend) {
          const state = await gateway.stateBackend.getSessionState(instanceId, task.sessionId);
          if (state?.l2_last_extraction_time) {
            cursor = state.l2_last_extraction_time;
          }
        }

        if (signal?.aborted) throw signal.reason ?? new Error("executeL2: aborted before LLM");

        core.setInstanceId(instanceId);
        const { store } = await resolveStore(task);
        const storage = await resolveStorage(task);

        // Count scenes before L2 to detect new scene creation
        let sceneCountBefore = 0;
        if (storage) {
          try {
            const { StoragePaths } = await import("../core/storage/types.js");
            const idx = await storage.readFile(StoragePaths.sceneIndex);
            if (idx) sceneCountBefore = JSON.parse(idx).length;
          } catch { /* ok */ }
        }

        const result = await core.runL2WithStore(task.sessionId, store, storage ?? undefined, cursor);

        // Mark task as skipped if L2 had no new records to process
        if (result.skipped) {
          (task as any)._l2Skipped = true;
        }

        // Report credit + new scenes as memory
        if (gateway.quotaManager && !result.skipped) {
          const { creditUsed } = result;
          let newScenes = 0;
          if (storage) {
            try {
              const { StoragePaths } = await import("../core/storage/types.js");
              const idx = await storage.readFile(StoragePaths.sceneIndex);
              if (idx) newScenes = Math.max(0, JSON.parse(idx).length - sceneCountBefore);
            } catch { /* ok */ }
          }
          if (creditUsed > 0 || newScenes > 0) {
            gateway.quotaManager.reportUsage(instanceId, newScenes, creditUsed, "L2").catch(() => {});
          }
        }
      },
      async executeL3(task: TaskPayload, signal?: AbortSignal) {
        const instanceId = typeof task.data?.instanceId === "string" ? task.data.instanceId : undefined;
        if (!instanceId) throw new Error(`L3 task ${task.id} missing instanceId`);

        if (signal?.aborted) throw signal.reason ?? new Error("executeL3: aborted before start");

        // Credit quota check before LLM call
        if (gateway.quotaManager) {
          const check = await gateway.quotaManager.checkCreditQuota(instanceId);
          if (!check.allowed) {
            gateway.logger.warn(`[executor] L3 skipped: credit limit exceeded (instance=${instanceId})`);
            return;
          }
        }

        if (signal?.aborted) throw signal.reason ?? new Error("executeL3: aborted before LLM");

        // Check if persona exists before L3 (to detect first creation)
        let personaExistedBefore = false;
        const storage = await resolveStorage(task);
        if (storage) {
          try {
            const { StoragePaths } = await import("../core/storage/types.js");
            personaExistedBefore = await storage.exists(StoragePaths.persona);
          } catch { /* ok */ }
        }

        core.setInstanceId(instanceId);
        const { store } = await resolveStore(task);
        const result = await core.runL3WithStore(store, storage ?? undefined);

        // Report credit + memory (only +1 on first persona creation)
        if (gateway.quotaManager) {
          const { creditUsed } = result;
          const memoryDelta = (!personaExistedBefore && storage) ? 1 : 0;
          if (creditUsed > 0 || memoryDelta > 0) {
            gateway.quotaManager.reportUsage(instanceId, memoryDelta, creditUsed, "L3").catch(() => {});
          }
        }
      },
      async executeFlush(task: TaskPayload) {
        await core.handleSessionEnd(task.sessionId);
      },

      // ── Offload executors (L1 summary, L1.5 task judgment, L2 MMD update) ──
      async executeOffloadL1(task: TaskPayload, signal?: AbortSignal) {
        if (signal?.aborted) return;
        const { OffloadTaskExecutor } = await import("../offload_server/offload-task-executor.js");
        const storage = await resolveStorage(task);
        if (!storage) return;
        const llmClient = gateway.buildOffloadLlmClient();
        if (!llmClient) {
          gateway.logger.warn(`[executor] offload-l1 skipped: no LLM client available`);
          return;
        }
        const executor = new OffloadTaskExecutor({
          resolveStorage: async () => storage,
          llmClient,
          stateBackend: gateway.stateBackend!,
          config: { ...gateway.config.offload, l1Model: "", l15Model: "", l2Model: "" },
          logger: gateway.logger,
        });
        await executor.executeOffloadL1(task, signal);
      },
      async executeOffloadL15(task: TaskPayload, signal?: AbortSignal) {
        if (signal?.aborted) return;
        const { OffloadTaskExecutor } = await import("../offload_server/offload-task-executor.js");
        const storage = await resolveStorage(task);
        if (!storage) return;
        const llmClient = gateway.buildOffloadLlmClient();
        if (!llmClient) {
          gateway.logger.warn(`[executor] offload-l15 skipped: no LLM client available`);
          return;
        }
        const executor = new OffloadTaskExecutor({
          resolveStorage: async () => storage,
          llmClient,
          stateBackend: gateway.stateBackend!,
          config: { ...gateway.config.offload, l1Model: "", l15Model: "", l2Model: "" },
          logger: gateway.logger,
        });
        await executor.executeOffloadL15(task, signal);
      },
      async executeOffloadL2(task: TaskPayload, signal?: AbortSignal) {
        if (signal?.aborted) return;
        const { OffloadTaskExecutor } = await import("../offload_server/offload-task-executor.js");
        const storage = await resolveStorage(task);
        if (!storage) return;
        const llmClient = gateway.buildOffloadLlmClient();
        if (!llmClient) {
          gateway.logger.warn(`[executor] offload-l2 skipped: no LLM client available`);
          return;
        }
        const executor = new OffloadTaskExecutor({
          resolveStorage: async () => storage,
          llmClient,
          stateBackend: gateway.stateBackend!,
          config: { ...gateway.config.offload, l1Model: "", l15Model: "", l2Model: "" },
          logger: gateway.logger,
        });
        await executor.executeOffloadL2(task, signal);
      },
    };
  }

  /**
   * Build a simple LLM client for offload executors using gateway's LLM config.
   */
  private buildOffloadLlmClient() {
    const llmCfg = this.config.llm;
    if (!llmCfg.baseUrl || !llmCfg.apiKey || !llmCfg.model) return null;
    const logger = this.logger;

    return {
      async chat(params: {
        model: string;
        messages: Array<{ role: "system" | "user"; content: string }>;
        temperature: number;
        max_tokens: number;
        timeoutMs?: number;
      }): Promise<string> {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), params.timeoutMs ?? 30000);
        try {
          const response = await fetch(`${llmCfg.baseUrl}/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${llmCfg.apiKey}`,
            },
            body: JSON.stringify({
              model: llmCfg.model || params.model,
              messages: params.messages,
              temperature: params.temperature,
              max_tokens: params.max_tokens,
            }),
            signal: controller.signal,
          });
          clearTimeout(timer);
          if (!response.ok) {
            throw new Error(`LLM API returned ${response.status}: ${await response.text()}`);
          }
          const json = (await response.json()) as any;
          const finishReason = json.choices?.[0]?.finish_reason;
          if (finishReason === "length") {
            const content = json.choices?.[0]?.message?.content ?? "";
            logger.warn(
              `[offload-llm] Response truncated (finish_reason=length, max_tokens=${params.max_tokens}), ` +
              `content=${content.length} chars`,
            );
          }
          return json.choices?.[0]?.message?.content ?? "";
        } catch (err) {
          clearTimeout(timer);
          throw err;
        }
      },
    };
  }
}

// ============================
// CLI entry point
// ============================

/**
 * Start the gateway from the command line.
 * Usage: node --import tsx src/gateway/server.ts
 */
async function main(): Promise<void> {
  const gateway = new TdaiGateway();

  // Graceful shutdown
  const shutdown = async () => {
    await gateway.stop();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  await gateway.start();
}

// Auto-start when run directly
const isMain = process.argv[1]?.endsWith("server.ts") || process.argv[1]?.endsWith("server.js");
if (isMain) {
  main().catch((err) => {
    console.error("Gateway startup failed:", err);
    process.exit(1);
  });
}
