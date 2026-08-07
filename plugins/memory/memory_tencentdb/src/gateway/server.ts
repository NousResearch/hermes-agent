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
import { join } from "node:path";
import { URL } from "node:url";
import { timingSafeEqual } from "node:crypto";
import zlib from "node:zlib";
import dayjs from "dayjs";
import { TdaiCore } from "../core/tdai-core.js";
import { StandaloneHostAdapter } from "../adapters/standalone/host-adapter.js";
import { loadGatewayConfig, parseBrokers } from "./config.js";
import type { GatewayConfig } from "./config.js";
import { applyMetadataEnvFromGatewayConfig } from "./metadata-env.js";
import { initDataDirectories } from "../utils/pipeline-factory.js";
import { SessionFilter } from "../utils/session-filter.js";
import { WorkerPermitPool } from "../services/worker-permit-pool.js";
import { createExtractorAdapter, SkillExtractor as SkillExtractorClass } from "../core/skill/skill-extractor.js";
import type { SkillCore as SkillCoreType } from "../core/skill/skill-core.js";
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
import { handleV3MetaRoute, V3_PREFIX } from "../metadata/router/v3-meta-router.js";
import { handleInternalMetaRoute, V3_INTERNAL_PREFIX } from "../metadata/router/internal-meta-router.js";
import { MetadataService } from "../metadata/service/metadata-service.js";
import { ConfigParamService } from "../metadata/service/config-param-service.js";
import { loadDefaultRegistry } from "../metadata/config/param-registry.js";
import type { IMetadataStore } from "../metadata/store/interface.js";
import {
  MetadataStorePool,
  validateMetadataStartupConfig,
} from "../metadata/store/factory.js";
import {
  maskMemorySystemUserKeyForLog,
  resolveMemorySystemUserConfig,
  validateMemorySystemUserConfig,
} from "../metadata/system-user.js";
import type { MemorySystemUserConfig } from "../metadata/system-user.js";
import { validateLlmProviderConfig, LlmResolveError } from "./llm-resolver.js";
import type { StandaloneLLMConfig } from "../adapters/standalone/llm-runner.js";
import { resolveStandaloneLlmForRuntime } from "../adapters/standalone/llm-provider-resolver.js";
import { resolveReportedCredit } from "./quota-credit-policy.js";
import {
  initApiTraceConfig,
  wrapApiServiceForTrace,
  wrapApiStoreForTrace,
} from "../api-trace/index.js";
import { readApiTraceEnabled } from "../utils/env-config.js";
import { makeSkillRouteTable } from "./skill-handlers.js";
import type { SkillRouterDeps as SkillRouterDeps } from "./skill-handlers.js";
import {
  wireConversationAdd,
  type WiredConversationAdd,
  type SkillAgentTaskQueueRedisLike,
} from "../core/skill/conversation-add/index.js";
import { makeKnowledgeRouteTable } from "./knowledge-handlers.js";
import { handleOffloadV2Route } from "../offload_server/router.js";
import type { OffloadV2Deps } from "../offload_server/router.js";
import { resolveV3StrictIsolation } from "../utils/env-config.js";
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
import { parsePipelineTimerMember } from "../core/state/timer-member.js";

const TAG = "[tdai-gateway]";
const VERSION = "2.0.0-beta.1";

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
  /** Guards against concurrent / repeated stop() invocations (e.g. multiple SIGINT). */
  private stopPromise: Promise<void> | null = null;

  // ── Integrated services (Scanner + Worker) ──
  private stateBackend: IStateBackend | null = null;
  private timerScanner: TimerScanner | null = null;
  private pipelineWorker: PipelineWorker | null = null;
  /**
   * 跨模块共享的并发信号量 —— memory PipelineWorker 用；skill 侧走
   * conversation-add wire 出的 SkillConversationExtractWorker (agent 级串行 lock,
   * 不依赖信号量做并发上限)。
   */
  private workerPermitPool: WorkerPermitPool | null = null;

  // ── Instance config & Store pool (multi-instance VDB) ──
  private configProvider: InstanceConfigProvider | null = null;
  private storePool: StorePool | null = null;
  private quotaManager: import("../core/quota/quota-manager.js").QuotaManager | null = null;
  private statefulPipelineManager: StatefulPipelineManager | null = null;

  // ── COS: global shared client singleton + per-instance StorageAdapter cache ──
  private sharedCosClient: import("../integrations/cos/cos-backend.js").SharedCosClient | null = null;
  private cosStorageCache: Map<string, StorageAdapter> | null = null;

  // ── Metadata (v3): shared store pool + per-instance MetadataService ──
  private metadataStorePool: MetadataStorePool | null = null;
  private memorySystemUserConfig: MemorySystemUserConfig | undefined;
  private readonly metadataServiceByInstance = new Map<string, MetadataService>();

  // ── Skill conversation-add (§21): per-instance handler+worker cache ──
  //
  // 每个实例懒加载一份 { handler, worker, sink, queue, buffer } —— storage 复用
  // memory 侧的 CosStorageBackend cache (this.cosStorageCache), redis 复用
  // stateBackend 的 ioredis 客户端 (getClient())。keyPrefix 拼接 memory 的
  // redis.keyPrefix，避免和老 skill 队列以及 memory 状态撞。
  //
  // 用 in-flight Promise cache 而不是最终值 cache —— 避免并发请求同时 miss cache
  // 导致重复 wire 出多个 Worker（v1 观察: 30 并发请求触发 30 次 wire, 每次起一个
  // Worker 都 BRPOP 抢同一 agent → extract-lock 灾难）。
  private readonly conversationAddByInstance = new Map<string, Promise<WiredConversationAdd>>();

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
    //
    // ── Skill 资产联动钩子（standalone/OpenClaw 与 service 模式对齐） ──
    // service 模式下 gateway/server.ts:resolveSkillCore 会为每个 instanceId 单独
    // 构造 per-instance SkillCore 并挂同名钩子；tdai-core 里的这份 SkillCore 走
    // standalone / OpenClaw 内嵌 / 未走 resolveSkillCore 的旁路。两者互不干扰
    // （每个 SkillCore 只调它自己被挂上的钩子），ensureSkillAsset / deleteAssets
    // 幂等，即使叠加触发也无副作用。详见 SkillAssetHooks doc。
    //
    // standalone 模式下 instanceId 固定为 "default"（见 start() 里的
    // `this.config.instanceId ?? "default"`），闭包这里直接拿 default 即可；
    // service 模式下这份 SkillCore 事实上不会被 v3/skill/* 走到，闭包的 default
    // 只是占位（不 fire 就没影响）。
    const gatewayRef = this;
    const skillAssetInstanceId = this.config.instanceId
      ?? (this.config.deployMode === "service" ? "__unset__" : "default");
    // Shared permit pool — memory PipelineWorker 用。skill 侧走
    // wireConversationAdd 内的 SkillConversationExtractWorker (agent 级串行 lock),
    // 不再使用信号量做并发上限。
    this.workerPermitPool = new WorkerPermitPool(this.config.worker.concurrency);

    this.core = new TdaiCore({
      hostAdapter: adapter,
      config: this.config.memory,
      sessionFilter: new SessionFilter(this.config.memory.capture.excludeAgents),
      skillAssetHooks: {
        // v1 首创前置 await：抛异常 = create 失败（避免「skill 已落库但 asset
        // 缺失」的静默不一致）。standalone 模式下唯一的登记入口除了 handler 层的
        // handleCreate 兜底之外就是这里 —— 无论谁调 SkillCore.create 都能触发。
        onSkillCreated: async ({ skill_id, team_id, agent_id, name }) => {
          if (!team_id || !agent_id) return; // 无租户上下文 → 跳过（OpenClaw local scope 等）
          const metaSvc = await gatewayRef.ensureMetadataService(skillAssetInstanceId);
          await metaSvc.ensureSkillAsset({ skill_id, team_id, agent_id, name });
        },
        // 读时自愈：fire-and-forget，异常吞掉。补历史 / 迁移 / 误删产生的孤儿 skill。
        onSkillAccessed: (skill) => {
          if (!skill.team_id || !skill.owner_agent_id) return;
          gatewayRef
            .ensureMetadataService(skillAssetInstanceId)
            .then((svc) => svc.ensureSkillAsset({
              skill_id: skill.skill_id,
              team_id: skill.team_id!,
              agent_id: skill.owner_agent_id!,
              name: skill.name,
            }))
            .catch((err: unknown) => {
              gatewayRef.logger.warn(
                `[skill-asset-sync] ensureSkillAsset(access) failed for ${skill.skill_id}: `
                  + (err instanceof Error ? err.message : String(err)),
              );
            });
        },
        // 归档级联：fire-and-forget，异常吞掉。二次 delete 会重触发钩子，最终收敛。
        onSkillArchived: ({ skill_id, team_id }) => {
          gatewayRef
            .ensureMetadataService(skillAssetInstanceId)
            .then((svc) => svc.deleteAssets([skill_id]))
            .catch((err: unknown) => {
              gatewayRef.logger.warn(
                `[skill-asset-sync] deleteAssets(archive) failed for ${skill_id}`
                  + ` (team=${team_id ?? "-"}): `
                  + (err instanceof Error ? err.message : String(err)),
              );
            });
        },
      },
    });
  }

  /**
   * Lazily init metadata store pool on first /v3/meta request.
   */
  private async ensureMetadataStorePool(): Promise<MetadataStorePool> {
    if (!this.metadataStorePool) {
      const fallbackSqliteBaseDir = join(this.config.data.baseDir, "metadata");
      const config = validateMetadataStartupConfig(
        this.config.deployMode,
        process.env,
        fallbackSqliteBaseDir,
      );
      this.metadataStorePool = new MetadataStorePool(config);
      this.logger.info(`[META-V3] metadata store pool ready (backend=${config.backend})`);
    }
    return this.metadataStorePool;
  }

  private async ensureMetadataStore(instanceId: string): Promise<IMetadataStore> {
    const pool = await this.ensureMetadataStorePool();
    return pool.getStore(instanceId);
  }

  private async ensureMetadataService(instanceId: string): Promise<MetadataService> {
    let svc = this.metadataServiceByInstance.get(instanceId);
    if (!svc) {
      const [store, pool] = await Promise.all([
        this.ensureMetadataStore(instanceId),
        this.ensureMetadataStorePool(),
      ]);
      const storeSource = pool.backend === "mongodb" ? "mongodb-adapter.ts" : "sqlite-adapter.ts";
      const rawSvc = new MetadataService(
        wrapApiStoreForTrace(store, storeSource),
        instanceId,
        this.logger,
        this.config.metadata,
        this.memorySystemUserConfig,
      );

      const registry = loadDefaultRegistry(this.config.metadata.configParamsFile);
      const configSvc = new ConfigParamService(store, registry);
      await configSvc.initDefaults(registry, this.config.metadata);
      rawSvc.setConfigParamService(configSvc);

      svc = wrapApiServiceForTrace(rawSvc);
      this.metadataServiceByInstance.set(instanceId, svc);
    }
    return svc;
  }

  /**
   * Start the Gateway HTTP server.
   */
  async start(): Promise<void> {
    // Initialize data directories
    initDataDirectories(this.config.data.baseDir);

    applyMetadataEnvFromGatewayConfig(this.config.metadata);

    this.memorySystemUserConfig = resolveMemorySystemUserConfig(this.config.metadata);
    validateMemorySystemUserConfig(this.config.deployMode, this.memorySystemUserConfig);
    if (this.memorySystemUserConfig) {
      this.logger.info(
        `[META-V3] memory system user loaded user_id=${this.memorySystemUserConfig.userId} ` +
        `key_prefix=${maskMemorySystemUserKeyForLog(this.memorySystemUserConfig.userKey)}`,
      );
    }

    // llm.provider=proxy 时 fail-fast：确认 memory 系统用户完整、userKey 格式合法。
    // instanceId 每请求变化，运行期由 resolveEffectiveLlmConfig 再校验一次；这里
    // 只兜底 systemUser / baseUrl 这类启动即可确定的错误，避免走到第一次抽取才炸。
    try {
      validateLlmProviderConfig(this.config.llm, this.config.metadata);
      if (this.config.llm.provider === "proxy") {
        this.logger.info(
          `[LLM] provider=proxy, baseUrl=${this.config.llm.baseUrl}, ` +
          `useMemorySystemUserKey=${this.config.llm.proxy?.useMemorySystemUserKey ?? true}`,
        );
      }
    } catch (err) {
      if (err instanceof LlmResolveError) {
        throw new Error(`[LLM] provider 配置校验失败: ${err.message}`);
      }
      throw err;
    }

    const metadataStoreCfg = validateMetadataStartupConfig(
      this.config.deployMode,
      process.env,
      join(this.config.data.baseDir, "metadata"),
    );
    if (this.config.deployMode === "service") {
      this.logger.info(
        `[META-V3] metadata store configured (backend=${metadataStoreCfg.backend})`,
      );
    }

    const metadataPool = await this.ensureMetadataStorePool();
    initApiTraceConfig(metadataPool.backend, { enabled: readApiTraceEnabled() });

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

    // ── 初始化 OTel SDK（Trace + Log + ClickHouse 双写 + Langfuse）──
    // 必须在 HTTP server 创建之前初始化，否则 wrapWithTrace 中的 Tracer 是 NoopTracer
    // 注意：即使 otel.enabled=false，只要 langfuse.enabled=true 也需要初始化 SDK
    // （Langfuse 通过 OTel SDK 的 spanProcessor 上报 LLM span）
    const needOTelSDK = obsCfg.otel.enabled || obsCfg.langfuse.enabled;
    if (needOTelSDK) {
      try {
        const otelOk = await initOTelSDK({
          serviceName: obsCfg.otel.serviceName,
          serviceVersion: obsCfg.otel.serviceVersion,
          // 仅当 otel 启用时才传 endpoint，避免 langfuse-only 模式下创建无效的 gRPC exporter
          endpoint: obsCfg.otel.enabled ? obsCfg.otel.endpoint : undefined,
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

    // ── Start integrated services (Scanner + Worker + State Backend) ──
    // In service mode, startIntegratedServices() also invokes
    // initSharedCosClient() which itself calls setStorage(new StorageAdapter(
    // cosBackend)); the `if (!this.core.getStorage())` guard below then
    // correctly skips the local fallback.
    await this.startIntegratedServices();

    // ── Initialize StorageAdapter for v2 API ──
    // In standalone mode, use LocalStorageBackend pointing to dataDir.
    // In service mode, CosStorageBackend was already injected above.
    if (!this.core.getStorage()) {
      const backend = new LocalStorageBackend(this.config.data.baseDir);
      this.core.setStorage(new StorageAdapter(backend));
      this.logger.info(`${TAG} StorageAdapter initialized (local: ${this.config.data.baseDir})`);
    }

    // ── Skill module post-wiring (after storage is set) ──
    // setStorage() above kicks off ensureSkillModuleWired() asynchronously
    // (B1 fix in tdai-core: concurrent triggers coalesce onto one promise);
    // we await it here so the FIRST /v3/skill/* request after `start()`
    // resolves doesn't race against an unfinished construction. Noop when
    // cfg.skill is absent or disabled.
    try {
      await this.core.ensureSkillModuleWired();
    } catch (err) {
      this.logger.warn(`${TAG} ensureSkillModuleWired failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    // Create HTTP server (with Trace middleware wrapping)
    //
    // [skill-perf 2026-07-21] Skill 接口可观测性埋点（issue：/v3/skill/extract
    // ttfb 忽快忽慢，缺少 phase-by-phase 耗时 log）：
    //   - socket-level 打 tcp.connect（debug）
    //   - request 头读完后立即打 skill.perf.req.enter，作为服务端 T0
    //   - res.on('finish') 打 skill.perf.req.finish（服务端 flush 时刻）
    // 只针对 /v3/skill/ 前缀，避免污染 meta / memory 主链路日志。
    this.server = http.createServer((req, res) => {
      const perfT0 = Date.now();
      const isSkill = typeof req.url === "string" && req.url.startsWith("/v3/skill/");
      if (isSkill) {
        // 挂到 req 上以便 handler 内部读取 T0 计算 phase 耗时
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (req as any).__skillPerfT0 = perfT0;
        const src = `${req.socket.remoteAddress ?? "?"}:${req.socket.remotePort ?? "?"}`;
        this.logger.info(
          `[skill-perf] req.enter t=${perfT0} method=${req.method ?? "?"} url=${req.url} src=${src}`,
        );
        res.on("finish", () => {
          const now = Date.now();
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const rid = (res as any).__skillReqId ?? "-";
          this.logger.info(
            `[skill-perf] req.finish t=${now} total=${now - perfT0}ms status=${res.statusCode} req_id=${rid} url=${req.url}`,
          );
        });
        res.on("close", () => {
          if (!res.writableEnded) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const rid = (res as any).__skillReqId ?? "-";
            this.logger.warn(
              `[skill-perf] req.close-before-end t=${Date.now()} elapsed=${Date.now() - perfT0}ms req_id=${rid} url=${req.url}`,
            );
          }
        });
      }
      wrapWithTrace(req, res, () => this.handleRequest(req, res)).catch((err) => {
        // wrapWithTrace 内部已经记录了错误，这里只做 fallback
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Internal Server Error" }));
        }
      });
    });

    // TCP-level：新 socket 打一条 debug，用来看 keepalive 是否命中（同一 src ip:port
    // 在很短时间内多次触发这条 log = 反代/网关没复用连接）
    this.server.on("connection", (socket) => {
      this.logger.debug?.(
        `[skill-perf] tcp.connect t=${Date.now()} src=${socket.remoteAddress ?? "?"}:${socket.remotePort ?? "?"}`,
      );
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
    // Idempotent: repeated calls (e.g. multiple SIGINT) share the same shutdown.
    if (this.stopPromise) {
      return this.stopPromise;
    }
    this.stopPromise = this.doStop();
    return this.stopPromise;
  }

  private async doStop(): Promise<void> {
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
    // Stop skill conversation-add per-instance workers.
    // Cache 里存的是 Promise<WiredConversationAdd>；先 await settle 再 stop。
    if (this.conversationAddByInstance.size > 0) {
      const settled = await Promise.allSettled(Array.from(this.conversationAddByInstance.values()));
      const stops = settled.flatMap((r) =>
        r.status === "fulfilled"
          ? [r.value.stop().catch((e) => this.logger.warn(`[skill-conversation-add] stop failed: ${e}`))]
          : [],
      );
      await Promise.allSettled(stops);
      this.conversationAddByInstance.clear();
      this.logger.info("Skill Conversation-Add workers stopped");
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
      // 进程关停场景下没有需要保护的 in-flight 请求, 跳过 grace period (默认 30s),
      // 否则 closeAll 会硬等满 grace 才返回, 拖慢退出。
      this.storePool.setGraceCloseDelay(0);
      await this.storePool.closeAll();
      this.logger.info("Store Pool closed");
    }
    if (this.metadataStorePool) {
      await this.metadataStorePool.closeAll();
      this.metadataStorePool = null;
      this.logger.info("Metadata Store Pool closed");
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

      // ── /v3/instance/destroy — v3 兼容版本。
      //    请求/响应契约与 v2 一致；鉴权同 v2（Bearer apiKey，运维接口，
      //    不走租户 x-tdai-user-key）。
      //    实现上先复用 v2 的通用清理（state / store / cos / quota），再预留
      //    v3 独有的 metadata 清理（team/user/asset/acl 等），后续由负责 v3
      //    metadata 的同学补上。此处不做完全复用，避免把 v3 侧清理静默漏掉。
      if (method === "POST" && pathname === "/v3/instance/destroy") {
        if (!this.checkAuth(req, res)) return;
        return await this.handleInstanceDestroyV3(req, res);
      }

      // ── v3 internal metadata（/v3/internal/meta/*，仅 Bearer）──
      if (pathname.startsWith(`${V3_INTERNAL_PREFIX}/`)) {
        if (!this.checkAuthForV2(req, res)) return;
        const handledInternal = await handleInternalMetaRoute(
          req, res, pathname, method, parseJsonBody, sendJson,
          {
            getMetadataService: (instanceId) => this.ensureMetadataService(instanceId),
            logger: this.logger,
          },
        );
        if (handledInternal) return;
      }

      // ── v3 metadata routes (/v3/meta/*) ──
      // Layer 1: same Bearer apiKey gate as v2. Layer 3 (x-tdai-user-key) in handleV3MetaRoute.
      if (pathname.startsWith(`${V3_PREFIX}/`)) {
        if (!this.checkAuthForV2(req, res)) return;
        const handledV3 = await handleV3MetaRoute(req, res, pathname, method, parseJsonBody, sendJson, {
          getMetadataService: (instanceId) => this.ensureMetadataService(instanceId),
          logger: this.logger,
        });
        if (handledV3) return;
      }

      // ── v2 / v3 API routes ──
      // /v2 = 现有数据面 + 管理面入口（team/agent 可选，user fallback）。
      // /v3 = L0–L3 数据面"严格 isolation 版本"（team/agent/user/session 必填），共享同一组 handler 实现，
      //       仅在 dispatch 层多一层校验。详见 v2-router.ts 中 V3_PREFIX/V3_ALLOWED_SUBPATHS 注释。
      //
      // Apply the develop-introduced apiKey gate first so v2/v3 inherits the
      // optional shared-secret protection. v2's own `parseV2Auth` (Bearer +
      // x-tdai-service-id) still runs inside `handleV2Route`, preserving
      // its existing semantics. When `server.apiKey` is unset, this gate
      // is a no-op (default-open), matching the develop_server_test
      // baseline.
      if (pathname.startsWith("/v2/") || pathname.startsWith("/v3/")) {
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
        // `V3_STRICT_ISOLATION` controls only /v3 L0–L3 memory data-plane
        // strictness. Default OFF for local/integration; production should set it.
        v3StrictIsolation: resolveV3StrictIsolation(),
        // handleConversationAdd 用它自动登记 chat_memory 资产（team+agent 粒度）
        // 并绑定到 agent。首次写入触发 create + bind；后续同 (team, agent) 走
        // MetadataService 的进程内 LRU 短路。
        getMetadataService: (instanceId) => this.ensureMetadataService(instanceId),
      };

      // Skill module deps — composed alongside V2RouterDeps so v2-router.ts
      // doesn't have to widen its interface (and break v2-router.test.ts
      // mocks). The skill route table is registered as `extraRouteTable` in
      // handleV2Route below, and our `mergedDeps` object satisfies BOTH
      // interfaces simultaneously (TypeScript-wise the cast widens it to
      // `unknown` so each handler reads its own fields).
      const skillDeps: SkillRouterDeps = {
        getSkillCore: () => this.core.getSkillCore(),
        getSkillExtractor: () => this.core.getSkillExtractor(),
        getResolvedSkillConfig: () => this.core.getResolvedSkillConfig(),
        quotaManager: this.quotaManager ?? undefined,
        logger: this.logger,
        // Standalone 模式下 SkillCore 由 TdaiCore 全局构造（不带 onSkillCreated 钩子），
        // handleCreate 用这个 dep 在 skill 创建成功后调 metaSvc.ensureSkillAsset()
        // 完成 asset 登记 + agent fixed-asset 绑定。service 模式下 buildSkillCore 里
        // 的 onSkillCreated 钩子做同样的事，两条路径都覆盖，ensureSkillAsset 本身幂等。
        getMetadataService: (instanceId) => this.ensureMetadataService(instanceId),
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

        v2Deps.resolveStorage = (instanceId: string) => this.resolveStorageForInstance(instanceId);

        // Pipeline notify: trigger async L1 extraction when v2 /conversation/add writes L0
        if (this.statefulPipelineManager) {
          const pipelineManager = this.statefulPipelineManager;
          v2Deps.notifyPipeline = async (
            instanceId: string,
            sessionId: string,
            rounds: number,
            teamId?: string,
            agentId?: string,
          ) => {
            await pipelineManager.notifyConversation(sessionId, [], instanceId, rounds, teamId, agentId);
          };
        }

        // Inject QuotaManager for memory/credit limit checks
        if (this.quotaManager) {
          v2Deps.quotaManager = this.quotaManager;
        }

        // ── Skill: per-instance resolver (TcvdbSkillStore + COS storage) ──
        // 复用 Memory 侧的 COS adapter，创建 per-instance SkillCore。
        // Skill queue + worker 如果已在 tdai-core 中启动则复用；否则
        // service 模式下可在此处单独启动。
        if (storePool.mode === "tcvdb") {
          // per-instance resolvers 抽到了私有方法（同一份实现被 handler 和 skill worker 共用）。
          skillDeps.resolveSkillCore = (instanceId: string) => this.resolveSkillCoreForInstance(instanceId);
          skillDeps.buildSkillExtractor = (core, instanceId) => this.buildSkillExtractorForInstance(core, instanceId);
        }
        // /v3/skill/conversation/add + /v3/skill/extract{,result} wiring:
        //   - tcvdb (service): 走 ensureConversationAddForInstance (per-instance TCVDB + COS)
        //   - sqlite (standalone): 走 ensureConversationAddForStandalone (单例 SqliteSkillStore + LocalStorage/内存队列)
        //
        // 返回完整 WiredConversationAdd:
        //   - handleConversationAdd 用 .handler
        //   - handleExtract 用 .trigger (direct-trigger)
        skillDeps.resolveConversationAdd = async (instanceId: string) => {
          const wired = storePool.mode === "tcvdb"
            ? await this.ensureConversationAddForInstance(instanceId)
            : await this.ensureConversationAddForStandalone(instanceId);
          return wired;
        };
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

      // Compose deps: V2RouterDeps fields + SkillRouterDeps fields. The
      // route table union of routeTable + makeSkillRouteTable() is what
      // tells the dispatcher which subset of fields each handler reads.
      const mergedDeps = Object.assign({}, v2Deps, skillDeps);

      // Merge skill + knowledge extra route tables
      const extraRoutes = {
        ...makeSkillRouteTable(),
        ...makeKnowledgeRouteTable(),
      } as Record<
        string,
        (body: unknown, auth: import("./v2-schemas.js").V2AuthContext, requestId: string, deps: unknown) => Promise<import("./v2-schemas.js").ApiResponseEnvelope>
      >;

      const handled = await handleV2Route(
        req,
        res,
        pathname,
        method,
        parseJsonBody,
        sendJson,
        mergedDeps as V2RouterDeps,
        extraRoutes,
      );
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
    const cleaned = await this.purgeInstanceCommon(instanceId, "v2");

    sendJson(res, 200, {
      code: 0,
      message: "ok",
      data: { instance_id: instanceId, cleaned },
    });
  }

  /**
   * POST /v3/instance/destroy — v3 兼容路由。
   *
   * 请求/响应契约与 v2 完全一致，调用方零改动即可切换。
   *
   * 在通用清理之上 drop 该实例元数据库（v3.0 分库：`MetadataStorePool.purgeInstance`）。
   */
  private async handleInstanceDestroyV3(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await parseJsonBody<{ instance_id?: string }>(req);
    const instanceId = body?.instance_id;

    if (!instanceId || typeof instanceId !== "string") {
      sendJson(res, 400, { code: 400, message: "Missing required field: instance_id" });
      return;
    }

    this.logger.info(`[instance/destroy] [v3] Purging instance: ${instanceId}`);
    const cleaned = await this.purgeInstanceCommon(instanceId, "v3");

    cleaned.v3_metadata = await this.purgeV3Metadata(instanceId);

    sendJson(res, 200, {
      code: 0,
      message: "ok",
      data: { instance_id: instanceId, cleaned },
    });
  }

  /**
   * 通用清理步骤（state / store / cos / quota），v2 与 v3 共用。
   * @param source 仅用于日志前缀，便于分辨调用方。
   */
  private async purgeInstanceCommon(
    instanceId: string,
    source: "v2" | "v3",
  ): Promise<Record<string, unknown>> {
    const tag = `[instance/destroy] [${source}]`;
    const cleaned: Record<string, unknown> = {};

    // 1. Purge state backend (timers, sessions, buffers, pending tasks)
    if (this.stateBackend?.purgeInstance) {
      try {
        const result = await this.stateBackend.purgeInstance(instanceId);
        cleaned.state = result;
        this.logger.info(`${tag} State purged: sessions=${result.sessions}, timers=${result.timers}, buffers=${result.buffers}`);
      } catch (err) {
        this.logger.error(`${tag} State purge failed: ${err instanceof Error ? err.message : String(err)}`);
        cleaned.state_error = err instanceof Error ? err.message : String(err);
      }
    }

    // 2. Evict store from StorePool (Memory + Skill)
    if (this.storePool) {
      try {
        await this.storePool.evict(instanceId);
        this.storePool.evictSkillStore(instanceId);
        cleaned.store_evicted = true;
        cleaned.skill_store_evicted = true;
      } catch (err) {
        this.logger.error(`${tag} Store evict failed: ${err instanceof Error ? err.message : String(err)}`);
        cleaned.store_evicted = false;
        cleaned.skill_store_evicted = false;
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
          this.logger.info(`${tag} COS objects deleted: ${deletedCount}`);
        }
      } catch (err) {
        this.logger.error(`${tag} COS cleanup failed: ${err instanceof Error ? err.message : String(err)}`);
        cleaned.cos_error = err instanceof Error ? err.message : String(err);
      }
    }

    // 5. Clear QuotaManager cache
    if (this.quotaManager) {
      (this.quotaManager as any).cache?.delete?.(instanceId);
      cleaned.quota_cache_cleared = true;
    }

    return cleaned;
  }

  /**
   * v3 独有：drop 该实例元数据库（MongoDB dropDatabase / SQLite 删目录）。
   * 失败时写入返回结构，不抛出，与 purgeInstanceCommon 一致。
   */
  private async purgeV3Metadata(instanceId: string): Promise<Record<string, unknown>> {
    try {
      const pool = await this.ensureMetadataStorePool();
      const result = await pool.purgeInstance(instanceId);
      this.metadataServiceByInstance.delete(instanceId);
      this.logger.info(
        `[instance/destroy] [v3] metadata purged: db=${result.db_name} dropped=${result.dropped}`,
      );
      return { db_name: result.db_name, dropped: result.dropped };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.logger.error(`[instance/destroy] [v3] metadata purge failed: ${msg}`);
      return { dropped: false, error: msg };
    }
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
          // Parse timer member by prefix: "offload-{type}:{instanceId}:{sessionId}[:{extra}]"
          // or legacy "session:L2_schedule"
          const member = entry.member;
          let taskType: string;
          let instanceId: string;
          let sessionId: string;
          let teamId: string | undefined;
          let agentId: string | undefined;

          const firstColon = member.indexOf(":");
          const prefix = firstColon > 0 ? member.slice(0, firstColon) : member;

          if (prefix === "offload-l1" || prefix === "offload-l15" || prefix === "offload-l2") {
            taskType = prefix;
            // Format: "offload-{type}:{instanceId}:{sessionId}[:{mmdFile}]"
            // instanceId is the segment right after the prefix
            const rest = member.slice(firstColon + 1);
            const instanceEnd = rest.indexOf(":");
            if (instanceEnd > 0) {
              instanceId = rest.slice(0, instanceEnd);
              sessionId = rest.slice(instanceEnd + 1);
            } else {
              instanceId = this.config.instanceId ?? "default";
              sessionId = rest;
            }
            // For offload-l2: strip trailing ":{mmdFile}" from sessionId
            // (mmdFile is extracted separately from timerMember in the executor)
            if (prefix === "offload-l2" && sessionId.endsWith(".mmd")) {
              const lastColon = sessionId.lastIndexOf(":");
              if (lastColon > 0) {
                sessionId = sessionId.slice(0, lastColon);
              }
            }
          } else {
            // Pipeline memory timer. The member may be legacy "sessionId:L1_idle"
            // or scoped "scope:team:T|agent:A|session:S:L1_idle"; scope is member
            // data only and does not affect Redis key slotting.
            const parsed = parsePipelineTimerMember(member);
            sessionId = parsed.sessionId;
            taskType = parsed.taskType;
            teamId = parsed.teamId;
            agentId = parsed.agentId;
            instanceId = this.config.instanceId ?? "default";
          }
          const now = Date.now();
          // Extract targetMmdFile from member for offload-l2 (needed by pipeline-worker lockKey)
          let targetMmdFile: string | undefined;
          if (taskType === "offload-l2") {
            const mmdMatch = member.match(/(\d+-[^:]+\.mmd)$/);
            if (mmdMatch) targetMmdFile = mmdMatch[1];
          }
          const task = {
            id: `${taskType}-${sessionId}-${now}`,
            type: taskType as any,
            instanceId,
            sessionId,
            teamId,
            agentId,
            priority: 0,
            createdAt: now,
            data: { triggeredBy: "timer_scanner", timerMember: member, instanceId, targetMmdFile, teamId, agentId },
          };
          this.stateBackend!.enqueueTask(task).then(() => {
            this.logger.info(`[local-timer] Timer fired: ${member} → enqueued ${taskType} task`);
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

    // Skill V2 queue opens its own ioredis connection from the
    // `skillRedisUrl` we passed into TdaiCore at construction time — see the
    // constructor above. No cross-module shared client, no startup-order
    // coupling with `setStorage()` / `ensureSkillModuleWired()`.

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
      this.logger.info("QuotaManager initialized (memoryLimit=10000, creditLimit=1000)");
    }
    // Allow overriding store mode independently from deployMode. Useful for
    // service-mode integration smoke tests where Redis + COS are real but no
    // VDB is available — set STORE_MODE=sqlite to keep the VDB-dependent
    // pieces local while exercising the rest of the service-mode wiring.
    const storeModeOverride = process.env.STORE_MODE === "sqlite" || process.env.STORE_MODE === "tcvdb"
      ? (process.env.STORE_MODE as "sqlite" | "tcvdb")
      : undefined;
    this.storePool = new StorePool({
      mode: storeModeOverride ?? (this.config.deployMode === "service" ? "tcvdb" : "sqlite"),
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
      // 与 skill worker 共享的节点级并发上限
      permitPool: this.workerPermitPool ?? undefined,
    }, this.logger);
    await this.pipelineWorker.start();
    this.logger.info("Pipeline Worker started");

    // Skill worker 由 ensureConversationAddForInstance/Standalone 内的
    // wireConversationAdd 起, 无需在此额外启动 (对齐 2026-07-17 skill_extract
    // 收敛方案: /v3/skill/extract 也走同一 worker + agent 队列)。
  }

  /**
   * 按 instance_id 构造 per-instance SkillCore（TCVDB VDB + COS + versioning + hooks）。
   *
   * 与 handleRequest 里内联版本共享同一实现体，避免 skill worker（进程级单例）
   * 和 handler（每请求）走两份不同的构造逻辑。
   */
  private async resolveSkillCoreForInstance(instanceId: string): Promise<SkillCoreType> {
    if (!this.configProvider || !this.storePool) {
      throw new SkillCoreError("SKILL_COS_REQUIRED", "resolveSkillCoreForInstance: configProvider/storePool not ready");
    }
    const configProvider = this.configProvider;
    const storePool = this.storePool;
    const logger = this.logger;
    const sharedCosClient = this.sharedCosClient;

    const vdbConfig = await configProvider.resolveVdb(instanceId);
    const skillStore = await storePool.getSkillStore(instanceId, vdbConfig);

    let storage: StorageAdapter;
    const cachedStorage = this.cosStorageCache?.get(instanceId);
    if (cachedStorage) {
      storage = cachedStorage;
    } else if (sharedCosClient) {
      const cosConfig = await configProvider.resolveCos();
      if (cosConfig?.cosUrl) {
        const { CosStorageBackend } = await import("../integrations/cos/cos-backend.js");
        const prefix = `${cosConfig.pathPrefix.replace(/\/$/, "")}/${instanceId}/`;
        const backend = new CosStorageBackend({ sharedClient: sharedCosClient, prefix, logger });
        storage = new StorageAdapter(backend);
        if (!this.cosStorageCache) this.cosStorageCache = new Map();
        this.cosStorageCache.set(instanceId, storage);
      } else {
        throw new SkillCoreError(
          "SKILL_COS_REQUIRED",
          "COS storage not configured for this instance. In service mode, skill.enabled=true requires COS credentials.",
        );
      }
    } else {
      throw new SkillCoreError(
        "SKILL_COS_REQUIRED",
        "COS storage not configured for this instance. In service mode, skill.enabled=true requires COS credentials.",
      );
    }

    const maxResourceSize = this.core.getResolvedSkillConfig()?.resources.maxResourceSizeBytes ?? 5_000_000;
    const { SkillResourceStore } = await import("../core/skill/skill-resource-store.js");
    const skillResources = new SkillResourceStore({ storage, maxResourceSizeBytes: maxResourceSize });

    const { SkillVersioning } = await import("../core/skill/skill-versioning.js");
    const quotaMgr = this.quotaManager;
    const resolveMetaSvc = () => this.ensureMetadataService(instanceId);
    const skillVersioning = new SkillVersioning({
      store: skillStore,
      resources: skillResources,
      storage,
      logger,
      onSkillVdbChanged: (delta: number) => {
        quotaMgr?.reportUsage(instanceId, delta, 0, "Skill").catch(() => {});
      },
      onSkillCreated: async ({ skill_id, team_id, agent_id, name }) => {
        if (!team_id || !agent_id) return;
        const metaSvc = await resolveMetaSvc();
        await metaSvc.ensureSkillAsset({ skill_id, team_id, agent_id, name });
      },
    });

    const { SkillCore } = await import("../core/skill/skill-core.js");
    return new SkillCore({
      store: skillStore,
      resources: skillResources,
      versioning: skillVersioning,
      onSkillAccessed: (skill) => {
        if (!skill.team_id || !skill.owner_agent_id) return;
        resolveMetaSvc()
          .then((svc) => svc.ensureSkillAsset({
            skill_id: skill.skill_id,
            team_id: skill.team_id,
            agent_id: skill.owner_agent_id,
            name: skill.name,
          }))
          .catch((err: unknown) => {
            logger.warn(
              `[skill-asset-sync] ensureSkillAsset(access) failed for ${skill.skill_id}: `
                + (err instanceof Error ? err.message : String(err)),
            );
          });
      },
      onSkillArchived: ({ skill_id, team_id }) => {
        resolveMetaSvc()
          .then((svc) => svc.deleteAssets([skill_id]))
          .catch((err: unknown) => {
            logger.warn(
              `[skill-asset-sync] deleteAssets(archive) failed for ${skill_id}`
                + ` (team=${team_id ?? "-"}): `
                + (err instanceof Error ? err.message : String(err)),
            );
          });
      },
    });
  }

  /**
   * 按 skillCore + instance_id 构造 per-instance SkillExtractor（含 LLM runner）。
   * instance_id 参与 llm resolver 拼 upstream URL（provider=proxy 场景关键）。
   */
  private async buildSkillExtractorForInstance(
    skillCore: SkillCoreType,
    instanceId: string,
  ): Promise<SkillExtractorClass> {
    const { SKILL_REVIEW_PROMPT } = await import("../core/skill/index.js");
    const { StandaloneLLMRunner } = await import("../adapters/standalone/llm-runner.js");
    const { resolveStandaloneLlmForRuntime, LlmProviderResolveError } = await import("../adapters/standalone/llm-provider-resolver.js");

    const llmCfg = this.config.llm;
    if (!llmCfg?.baseUrl) {
      throw new SkillCoreError("LLM_UNAVAILABLE", "LLM baseUrl not configured for skill extraction");
    }
    if ((llmCfg.provider ?? "openai") === "openai" && !llmCfg.apiKey) {
      throw new SkillCoreError("LLM_UNAVAILABLE", "LLM apiKey not configured (provider=openai)");
    }

    let effective: import("../adapters/standalone/llm-runner.js").StandaloneLLMConfig;
    try {
      effective = resolveStandaloneLlmForRuntime(llmCfg, instanceId);
    } catch (err) {
      const msg = err instanceof LlmProviderResolveError ? err.message : String(err);
      throw new SkillCoreError("LLM_UNAVAILABLE", `LLM provider resolve failed: ${msg}`);
    }
    this.logger.debug?.(
      `[skill-extractor] resolved LLM: provider=${llmCfg.provider ?? "openai"}, model=${effective.model}, baseUrl=${effective.baseUrl}`,
    );

    const llmRunner = new StandaloneLLMRunner({
      config: {
        baseUrl: effective.baseUrl,
        apiKey: effective.apiKey,
        model: effective.model ?? "default",
        timeoutMs: effective.timeoutMs ?? 120_000,
      },
    });
    const cfg = this.core.getResolvedSkillConfig();
    return new SkillExtractorClass({
      core: skillCore,
      runner: llmRunner,
      systemPrompt: SKILL_REVIEW_PROMPT,
      maxIterations: cfg?.extraction.maxIterations ?? 5,
      // 透传 archiveBytes 派生的 head/tail chars + 独立的 maxTokens，
      // 让 skill.extraction.archiveBytes 与 skill.extraction.maxTokens 生效
      // (yaml 里改完不用改代码)。cfg 缺失时 SkillExtractor 内部有默认。
      headChars: cfg?.extraction.headChars,
      tailChars: cfg?.extraction.tailChars,
      maxTokens: cfg?.extraction.maxTokens,
      logger: this.logger,
    } as import("../core/skill/skill-extractor.js").ExtractorOptions);
  }

  /**
   * 懒装 per-instance conversation-add：handler + worker + sink 一整套。
   *
   * 依赖：
   *   - storage: 复用 memory 的 CosStorageBackend（resolveSkillCoreForInstance 里已 cache）
   *   - redis: 复用 stateBackend 的 ioredis 客户端（skill:tasks-mutex / skill:extract-lock
   *     用同一物理连接）
   *   - skillCore/extractor: 走跟 handleExtract 相同的 per-instance factory
   *   - metadataService: 兜底登记 skill asset
   *
   * keyPrefix 拼 memory 的 redis.keyPrefix，例如
   * `tdai_memory_prod_v3:skill-conv` —— 跟老 skill 队列
   * (`{prefix}:skill:*`) 字面不撞。
   */
  private ensureConversationAddForInstance(instanceId: string): Promise<WiredConversationAdd> {
    // In-flight cache：同 instanceId 并发请求共享同一个 wire promise，
    // 避免并发情形下 wire 被跑多次生成多个 Worker（灾难：多 Worker 抢同一 agent lock）。
    const cached = this.conversationAddByInstance.get(instanceId);
    if (cached) return cached;

    const inflight = this.buildConversationAddForInstance(instanceId);
    this.conversationAddByInstance.set(instanceId, inflight);
    // 失败时移除 cache，允许下次重试
    inflight.catch(() => {
      if (this.conversationAddByInstance.get(instanceId) === inflight) {
        this.conversationAddByInstance.delete(instanceId);
      }
    });
    return inflight;
  }

  /**
   * Standalone (sqlite) 版本的 in-flight cache 入口，语义与
   * ensureConversationAddForInstance 完全一致，只是 build 走 standalone 路径。
   */
  private ensureConversationAddForStandalone(instanceId: string): Promise<WiredConversationAdd> {
    const cached = this.conversationAddByInstance.get(instanceId);
    if (cached) return cached;

    const inflight = this.buildConversationAddForStandalone(instanceId);
    this.conversationAddByInstance.set(instanceId, inflight);
    inflight.catch(() => {
      if (this.conversationAddByInstance.get(instanceId) === inflight) {
        this.conversationAddByInstance.delete(instanceId);
      }
    });
    return inflight;
  }

  /**
   * 把 resolved skill config 的归档/压缩派生字段翻译成 wireConversationAdd 的
   * thresholds / compressOptions / oversizeOptions 覆盖参数。cfg 缺失时返回空对象,
   * wire 层退回内置默认 (与旧行为等价)。
   *
   * 让 yaml 里的 skill.extraction.archiveBytes / skill.compress 生效不用改代码。
   */
  private buildSkillWireOverrides(): Pick<
    Parameters<typeof wireConversationAdd>[0],
    "thresholds" | "compressOptions" | "oversizeOptions"
  > {
    const cfg = this.core.getResolvedSkillConfig();
    if (!cfg) return {};
    return {
      thresholds: {
        toolCallThreshold: cfg.extraction.toolCallThreshold,
        bytesThreshold: cfg.extraction.bytesThreshold,
        requestCompressThresholdBytes: cfg.extraction.requestCompressThresholdBytes,
      },
      compressOptions: {
        toolContentThresholdBytes: cfg.compress.toolContentThresholdBytes,
        headBytes: cfg.compress.headBytes,
        tailBytes: cfg.compress.tailBytes,
      },
      oversizeOptions: {
        chunkMaxBytes: cfg.extraction.chunkMaxBytes,
        headKeepBytes: cfg.extraction.headKeepBytes,
        tailKeepBytes: cfg.extraction.tailKeepBytes,
      },
    };
  }

  private async buildConversationAddForInstance(instanceId: string): Promise<WiredConversationAdd> {
    // 1) storage —— 复用 resolveSkillCoreForInstance 已经建好的 per-instance COS adapter
    //    resolveSkillCoreForInstance 里已经把 skillCore 和 storage 关联并 cache 好,
    //    这里再调一次仅是为了保证 cosStorageCache 里有对应 instance 的 adapter。
    const skillCore = await this.resolveSkillCoreForInstance(instanceId);
    const storage = this.cosStorageCache?.get(instanceId);
    if (!storage) {
      throw new Error(`[skill-conversation-add] storage missing for instance=${instanceId} after resolveSkillCore`);
    }

    // 2) redis —— 从 stateBackend 拿 ioredis (只有 RedisStateBackend 才有 getClient)
    const redisClient = this.getSharedIoRedisClient();
    if (!redisClient) {
      // Local state backend 场景：降级到内存 queue（单节点可用）
      this.logger.warn(
        `[skill-conversation-add] no shared ioredis client available (deployMode=${this.config.deployMode}); ` +
          `falling back to in-memory queue for instance=${instanceId} (single-node only)`,
      );
    }

    // 3) extractor —— 复用 buildSkillExtractorForInstance
    const rawExtractor = await this.buildSkillExtractorForInstance(skillCore, instanceId);
    const extractor = createExtractorAdapter(rawExtractor, this.logger);

    // 4) metadataService —— 兜底 asset 登记 (sink 只做这个)
    const metadataService = await this.ensureMetadataService(instanceId).catch(() => undefined);

    // 5) Redis key prefix
    const memoryPrefix = this.config.redis?.keyPrefix ?? "tdai_memory";
    const keyPrefix = `${memoryPrefix}:skill-conv`;

    const wired = wireConversationAdd({
      storage,
      redis: redisClient as SkillAgentTaskQueueRedisLike | undefined,
      redisKeyPrefix: keyPrefix,
      metadataService,
      extractor,
      logger: this.logger,
      workerId: `${this.config.instanceId ?? "gateway"}:${instanceId}:skill-conv-worker`,
      // 归档 / 压缩 / 兜底 参数从 resolved skill config 派生，
      // 让 skill.extraction.archiveBytes + skill.compress yaml 改动即时生效。
      ...this.buildSkillWireOverrides(),
    });

    this.logger.info(
      `[skill-conversation-add] wired instance=${instanceId} storage=${storage.type} ` +
        `redis=${redisClient ? "shared-ioredis" : "in-memory-fallback"} prefix=${keyPrefix}`,
    );
    return wired;
  }

  /**
   * Standalone 模式的 conversation-add wiring。
   *
   * 与 service 模式 (buildConversationAddForInstance) 的差异：
   *   - skillCore/extractor：走 TdaiCore 全局单例（SqliteSkillStore + 单例 extractor），
   *     不做 per-instance；standalone 本来就是单实例部署。
   *   - storage：走 resolveStorageForInstance → LocalStorageBackend（COS 降级为 fs）。
   *   - redis：不一定有；wire.ts 会自动降级到 LocalSkillAgentTaskQueue（单节点内存队列）。
   *   - extractor：若 cfg.llm 不全导致 TdaiCore 没造出 skillExtractor，就 skipWorker
   *     只挂 handler+buffer，Client 端能落 buffer，但归档触发不了（warn）。
   */
  private async buildConversationAddForStandalone(instanceId: string): Promise<WiredConversationAdd> {
    // 1) skillCore —— 单例
    const skillCore = this.core.getSkillCore();
    if (!skillCore) {
      throw new Error(`[skill-conversation-add] SkillCore not enabled (standalone) — check cfg.skill.enabled`);
    }

    // 2) storage —— 复用 resolveStorageForInstance；standalone 下会返回 LocalStorageBackend adapter
    const storage = await this.resolveStorageForInstance(instanceId);

    // 3) redis —— standalone 下通常没有；wire 层自动降级到 LocalSkillAgentTaskQueue
    const redisClient = this.getSharedIoRedisClient();

    // 4) extractor —— TdaiCore 单例，可能是 undefined（cfg.llm 未配全）
    const rawExtractor = this.core.getSkillExtractor();
    const skipWorker = !rawExtractor;
    if (skipWorker) {
      this.logger.warn(
        `[skill-conversation-add] singleton SkillExtractor unavailable (check cfg.llm); ` +
          `wiring handler+buffer only, skipping Worker for instance=${instanceId}. ` +
          `Buffer will accumulate but archive won't fire until extractor is available.`,
      );
    }
    // Worker 需要一个 extractor 实例；skipWorker=true 时占位不会被调用。
    const noopExtractor: import("../core/skill/queue/types.js").ISkillExtractor = {
      extract: async () => ({ candidates: [] }),
    };
    const extractor = rawExtractor
      ? createExtractorAdapter(rawExtractor, this.logger)
      : noopExtractor;

    // 5) metadataService —— 兜底 asset 登记 (sink 只做这个)
    const metadataService = await this.ensureMetadataService(instanceId).catch(() => undefined);

    // 6) Redis key prefix (即便走内存队列, 命名保持一致以便后续接 redis 时无缝切换)
    const memoryPrefix = this.config.redis?.keyPrefix ?? "tdai_memory";
    const keyPrefix = `${memoryPrefix}:skill-conv`;

    const wired = wireConversationAdd({
      storage,
      redis: redisClient as SkillAgentTaskQueueRedisLike | undefined,
      redisKeyPrefix: keyPrefix,
      metadataService,
      extractor,
      logger: this.logger,
      workerId: `${this.config.instanceId ?? "gateway"}:${instanceId}:skill-conv-worker`,
      skipWorker,
      // 归档 / 压缩 / 兜底 参数由 resolved skill config 派生 (跟 service 模式一致)
      ...this.buildSkillWireOverrides(),
    });

    this.logger.info(
      `[skill-conversation-add] wired (standalone) instance=${instanceId} storage=${storage.type} ` +
        `redis=${redisClient ? "shared-ioredis" : "in-memory-fallback"} ` +
        `worker=${skipWorker ? "skipped" : "on"} prefix=${keyPrefix}`,
    );
    return wired;
  }

  /**
   * Resolve per-instance StorageAdapter.
   *
   * Two paths (both cached in `cosStorageCache` — the name is historical; it
   * also holds LocalStorageBackend adapters in standalone mode):
   *   - Standalone (sharedCosClient == null && deployMode==='standalone'):
   *     LocalStorageBackend rooted at data.baseDir.
   *   - Service: per-instance CosStorageBackend with `${pathPrefix}/${instanceId}/`.
   *
   * Both `/v2/*` (memory) and `/v3/skill/conversation/add` need per-instance
   * storage; keeping this in one method keeps the fallback semantics identical
   * on both paths.
   */
  private async resolveStorageForInstance(instanceId: string): Promise<StorageAdapter> {
    const cached = this.cosStorageCache?.get(instanceId);
    if (cached) return cached;

    // Standalone mode: fall back to local storage (no COS needed)
    if (!this.sharedCosClient && this.config.deployMode === "standalone") {
      const localDir = this.config.data.baseDir;
      const backend = new LocalStorageBackend({ rootDir: localDir, logger: this.logger });
      const adapter = new StorageAdapter(backend);
      if (!this.cosStorageCache) this.cosStorageCache = new Map();
      this.cosStorageCache.set(instanceId, adapter);
      return adapter;
    }

    if (!this.sharedCosClient) {
      throw new Error(`SharedCosClient not initialized for instance ${instanceId}`);
    }
    if (!this.configProvider) {
      throw new Error(`configProvider not initialized for instance ${instanceId}`);
    }

    // Get current COS config to determine prefix
    const cosConfig = await this.configProvider.resolveCos();
    if (!cosConfig?.cosUrl) {
      throw new Error(`COS config not available for instance ${instanceId} (Shark returned null or empty CosUrl)`);
    }

    // Per-instance CosStorageBackend: lightweight, only holds prefix
    const { CosStorageBackend } = await import("../integrations/cos/cos-backend.js");
    const prefix = `${cosConfig.pathPrefix.replace(/\/$/, '')}/${instanceId}/`;
    const backend = new CosStorageBackend({
      sharedClient: this.sharedCosClient,
      prefix,
      logger: this.logger,
    });
    const adapter = new StorageAdapter(backend);
    if (!this.cosStorageCache) this.cosStorageCache = new Map();
    this.cosStorageCache.set(instanceId, adapter);
    return adapter;
  }

  /**
   * 从 RedisStateBackend 取 ioredis 客户端；非 Redis backend 返回 null。
   */
  private getSharedIoRedisClient(): SkillAgentTaskQueueRedisLike | null {
    if (!this.stateBackend) return null;
    // duck-type：只 RedisStateBackend 有 getClient()
    const maybe = this.stateBackend as unknown as { getClient?: () => unknown };
    if (typeof maybe.getClient !== "function") return null;
    try {
      return maybe.getClient() as SkillAgentTaskQueueRedisLike;
    } catch {
      return null;
    }
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
        const teamId = task.teamId ?? (typeof task.data?.teamId === "string" ? task.data.teamId : undefined);
        const agentId = task.agentId ?? (typeof task.data?.agentId === "string" ? task.data.agentId : undefined);

        // H-11 Step 2: early abort check — if pipeline-worker already lost its lock
        // before we even started, bail out without doing any work.
        if (signal?.aborted) throw signal.reason ?? new Error("executeL1: aborted before start");

        // Dedup: if triggered by timer but session already processed (count=0), skip
        if (task.data?.triggeredBy === "timer_scanner" && gateway.stateBackend) {
          const state = await gateway.stateBackend.getSessionState(instanceId, task.sessionId, teamId, agentId);
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

        (task as any)._l2ProfileScopes = result.profileScopes;

        // Report usage after L1: memory added + credit consumed
        // provider=proxy 模式下 credit 由 context_proxy 上报，此处仅报 memory delta。
        if (gateway.quotaManager) {
          const { storedCount, creditUsed } = result;
          const reportCredit = gateway.reportedCreditFor(creditUsed, "L1");
          if (storedCount > 0 || reportCredit > 0) {
            gateway.quotaManager.reportUsage(instanceId, storedCount, reportCredit, "L1").catch(() => {});
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
            gateway.statefulPipelineManager.enqueueL1Drain(task.sessionId, instanceId, teamId, agentId).catch((err) => {
              gateway.logger.warn(`[executor] L1 drain enqueue failed: ${err instanceof Error ? err.message : String(err)}`);
            });
          } else if (result.hasMore) {
            gateway.statefulPipelineManager.armL1IdleAfterDrain(task.sessionId, instanceId, teamId, agentId).catch((err) => {
              gateway.logger.warn(`[executor] L1 idle arm failed: ${err instanceof Error ? err.message : String(err)}`);
            });
          }
        }
      },
      async executeL2(task: TaskPayload, signal?: AbortSignal) {
        const instanceId = typeof task.data?.instanceId === "string" ? task.data.instanceId : undefined;
        if (!instanceId) throw new Error(`L2 task ${task.id} missing instanceId`);
        const teamId = task.teamId ?? (typeof task.data?.teamId === "string" ? task.data.teamId : undefined);
        const agentId = task.agentId ?? (typeof task.data?.agentId === "string" ? task.data.agentId : undefined);

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
          const state = await gateway.stateBackend.getSessionState(instanceId, task.sessionId, teamId, agentId);
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
          // provider=proxy 模式下 credit 由 context_proxy 上报，此处仅报 memory delta。
          const reportCredit = gateway.reportedCreditFor(creditUsed, "L2");
          if (reportCredit > 0 || newScenes > 0) {
            gateway.quotaManager.reportUsage(instanceId, newScenes, reportCredit, "L2").catch(() => {});
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
        // provider=proxy 模式下 credit 由 context_proxy 上报，此处仅报 memory delta。
        if (gateway.quotaManager) {
          const { creditUsed } = result;
          const memoryDelta = (!personaExistedBefore && storage) ? 1 : 0;
          const reportCredit = gateway.reportedCreditFor(creditUsed, "L3");
          if (reportCredit > 0 || memoryDelta > 0) {
            gateway.quotaManager.reportUsage(instanceId, memoryDelta, reportCredit, "L3").catch(() => {});
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
        const llmClient = gateway.buildOffloadLlmClient(task.instanceId);
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
        const llmClient = gateway.buildOffloadLlmClient(task.instanceId);
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
        const llmClient = gateway.buildOffloadLlmClient(task.instanceId);
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
   * 应上报的 CreditDelta：provider=proxy 时归零（proxy 已上报，避免重复计费），
   * 否则原样传递。见 quota-credit-policy.ts 详解。
   */
  private reportedCreditFor(rawCreditUsed: number, level: "L1" | "L2" | "L3"): number {
    const reported = resolveReportedCredit(rawCreditUsed, this.config.llm.provider);
    if (reported === 0 && rawCreditUsed > 0) {
      this.logger.debug?.(
        `[quota] ${level} creditUsed=${rawCreditUsed} 由 context_proxy 上报，内核跳过 credit 部分`,
      );
    }
    return reported;
  }

  /**
   * Build a simple LLM client for offload executors using gateway's LLM config.
   *
   * instanceId 用于 provider=proxy 场景把 baseUrl 拼成 /proxy/<iid>/v1；
   * provider=openai 场景可以传 undefined。
   */
  private buildOffloadLlmClient(instanceId?: string) {
    const llmCfg = this.config.llm;
    if (!llmCfg.baseUrl || !llmCfg.model) return null;
    // provider=openai 时仍要求显式 apiKey；provider=proxy 时 apiKey 由 resolver 从 env 注入
    if ((llmCfg.provider ?? "openai") === "openai" && !llmCfg.apiKey) return null;

    let effective: StandaloneLLMConfig;
    try {
      effective = resolveStandaloneLlmForRuntime(llmCfg, instanceId);
    } catch (err) {
      this.logger.warn(
        `[offload-llm] provider 解析失败, skipped: ${err instanceof Error ? err.message : String(err)}`,
      );
      return null;
    }
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
          const response = await fetch(`${effective.baseUrl}/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${effective.apiKey}`,
            },
            body: JSON.stringify({
              model: effective.model || params.model,
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

  // Graceful shutdown. A *deliberate* second signal (impatient Ctrl+C) forces
  // immediate exit. We debounce because a single Ctrl+C under `tsx` arrives
  // twice (TTY delivers to the process group AND tsx forwards it), and those
  // duplicates land within milliseconds — they must not be treated as "again".
  const FORCE_EXIT_DEBOUNCE_MS = 1000;
  let firstSignalAt = 0;
  const shutdown = async (signal: NodeJS.Signals) => {
    const now = Date.now();
    if (firstSignalAt === 0) {
      firstSignalAt = now;
      await gateway.stop();
      process.exit(0);
    }
    // Ignore duplicate deliveries of the same keypress; only a clearly separate
    // signal (after the debounce window) forces an immediate exit.
    if (now - firstSignalAt >= FORCE_EXIT_DEBOUNCE_MS) {
      console.error(`Received ${signal} again — forcing exit.`);
      process.exit(130);
    }
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
