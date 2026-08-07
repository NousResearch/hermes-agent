/**
 * TDAI Gateway — Configuration management.
 *
 * Reads gateway configuration from:
 * 1. `tdai-gateway.yaml` (or JSON) in CWD or data dir
 * 2. Environment variables (override individual fields)
 *
 * Minimal config: just LLM API credentials. Everything else has sensible defaults.
 */

import fs from "node:fs";
import path from "node:path";
import YAML from "yaml";
import { getEnv } from "../utils/env.js";
import { parseConfig as parseMemoryConfig } from "../config.js";
import type { MemoryTdaiConfig } from "../config.js";
import type { StandaloneLLMConfig } from "../adapters/standalone/llm-runner.js";

// ============================
// Gateway config types
// ============================

/**
 * Deployment mode determines how the system manages state and coordination:
 *
 * - "standalone": Open-source single-node mode.
 *     Pipeline state lives in-process (Map/setTimeout/SerialQueue).
 *     No external dependencies beyond LLM API and optional VDB.
 *     Suitable for single-machine / sidecar / developer setups.
 *
 * - "service": Cloud service (multi-tenant) mode.
 *     Pipeline state is externalized through IStateBackend.
 *     Timer Scanner + Pipeline Worker run inside the gateway process.
 *     Supports multi-replica coordination and HA.
 *     May require deployment-specific remote backends.
 */
export type DeployMode = "standalone" | "service";

export interface RedisConfig {
  host: string;
  port: number;
  password?: string;
  db: number;
  keyPrefix: string;
}

export interface SharkConfig {
  baseUrl?: string;
  vdbTtlMs: number;
  cosBufferMs: number;
  maxInstances: number;
}

export interface ScannerConfig {
  instances: string;
  instancesSharkUrl?: string;
  intervalMs: number;
  nodeId?: string;
}

export interface WorkerConfig {
  pollMs: number;
  /** 并发消费协程数 (default: 60) */
  concurrency: number;
}

export interface CosExtraConfig {
  domain?: string;
}

export interface KafkaConfig {
  /** 是否启用 Kafka (默认: false) */
  enabled: boolean;
  /** Kafka Broker 列表 (逗号分隔) */
  brokers: string;
  /** Topic 名称 (默认: "memory_monitor") */
  topic: string;
  /** 消费者组 ID（仅 Consumer 使用，如 Monitor） */
  groupId?: string;
  /** 分区总数（仅 Producer 使用，用于 hash 分区） */
  totalPartitions?: number;
}

export interface OTelConfig {
  /** 是否启用 OTel SDK (默认: false) */
  enabled: boolean;
  /** Collector endpoint (默认: http://localhost:4317) */
  endpoint: string;
  /** 协议: "grpc" | "http/protobuf" (默认: "grpc") */
  protocol: "grpc" | "http/protobuf";
  /** OTLP 请求头，用于鉴权等，格式 key=value 逗号分隔 */
  headers?: string;
  /** 服务名 (默认: "core") */
  serviceName: string;
  /** 服务版本 */
  serviceVersion: string;
  /** 实例标识 */
  instanceId?: string;
  /** 智研 APM 租户 ID */
  tenantId: string;
  /** Metric 导出间隔 (秒, 默认: 60) */
  metricExportInterval?: number;
  /** Log 导出间隔 (秒, 默认: 5) */
  logExportInterval: number;
}

export interface ClickHouseConfig {
  /** 是否启用 ClickHouse 双写 (默认: false) */
  enabled: boolean;
  /** ClickHouse HTTP endpoint */
  endpoint: string;
  /** 用户名 */
  username: string;
  /** 密码 */
  password: string;
  /** 数据库名 */
  database: string;
  /** 写入的目标表名（Monitor 使用，其他组件可留空） */
  table?: string;
  /** 批量写入最大条数 */
  maxBatchSize?: number;
  /** 刷新间隔（秒） */
  flushInterval?: number;
  /** 缓冲队列最大长度，超出时丢弃数据 */
  maxQueueSize?: number;
}

export interface LangfuseConfig {
  /** 是否启用 Langfuse LLM trace 上报 (默认: false) */
  enabled: boolean;
  /** Langfuse 实例地址 (如 http://langfuse.example.local:3000) */
  host: string;
  /** Langfuse 公钥 */
  publicKey: string;
  /** Langfuse 私钥 */
  secretKey: string;
}

/**
 * 可观测性配置（统一格式，四组件共用）。
 * 各组件按需启用对应子配置，未使用的子配置保持 enabled=false 即可。
 *
 * 各子配置按部署需要启用；未使用的子配置保持 enabled=false 即可。
 *
 * yaml 示例:
 * ```yaml
 * observability:
 *   otel:
 *     enabled: true
 *     endpoint: "http://otel.example.com:4317"
 *     protocol: "grpc"
 *     serviceName: "core"
 *     serviceVersion: "1.0.0"
 *     tenantId: "<APM_TENANT_ID>"
 *     metricExportInterval: 60
 *     logExportInterval: 5
 *   clickhouse:
 *     enabled: true
 *     endpoint: "http://clickhouse.example.local:8123"
 *     username: "default"
 *     password: "xxx"
 *     database: "tdai_eval"
 *     maxBatchSize: 1000
 *     flushInterval: 5
 *     maxQueueSize: 10000
 *   kafka:
 *     enabled: true
 *     brokers: "kafka.example.local:9092"
 *     topic: "memory_monitor"
 *     totalPartitions: 32
 *   langfuse:
 *     enabled: true
 *     host: "http://langfuse.example.local:3000"
 *     publicKey: "pk-lf-xxx"
 *     secretKey: "sk-lf-yyy"
 * ```
 */
export interface ObservabilityConfig {
  /** OTel SDK 配置 (Trace + Log)。 */
  otel: OTelConfig;
  /** ClickHouse 双写配置。 */
  clickhouse: ClickHouseConfig;
  /** Kafka 配置。 */
  kafka: KafkaConfig;
  /** Barad 云监控上报配置。 */
  barad?: BaradConfig;
  /** 智研监控宝 Metric 上报配置。 */
  zhiyan?: ZhiYanConfig;
  /** Langfuse LLM trace 上报配置。 */
  langfuse: LangfuseConfig;
}

/** Barad 云监控上报配置。 */
export interface BaradConfig {
  /** 是否启用 Barad 上报 (默认: false) */
  enabled: boolean;
  /** 上报地域，如 ap-guangzhou */
  region: string;
  /** 命名空间 (默认: "qce/memory") */
  namespace: string;
  /** 上报频率（秒）(默认: 60) */
  freq: number;
  /** 测试环境上报地址（覆盖默认的 region 拼接地址） */
  testEndpoint?: string;
  /** 测试环境查询地址（用于集成测试验证数据） */
  testQueryEndpoint?: string;
  /** 采集间隔（秒）(默认: 10) */
  collectInterval?: number;
}

/** 智研监控宝 Metric 上报配置。使用组件: Monitor ✓ */
export interface ZhiYanConfig {
  /** 是否启用智研 Metric 上报 (默认: false) */
  enabled: boolean;
  /** 智研监控宝上报地址 */
  endpoint: string;
  /** 应用标识（智研监控宝必需），格式: {业务ID}_{应用ID}_{应用名} */
  appMark: string;
  /** 分组名称 (默认: "default") */
  group: string;
  /** 环境标识，如 dev/test/prod */
  env: string;
  /** 指标命名空间前缀 (默认: "memory") */
  namespace: string;
  /** 上报间隔（秒）(默认: 60) */
  exportInterval: number;
}

export interface GatewayConfig {
  /**
   * Deployment mode. Default: "standalone".
   *
   * env: TDAI_DEPLOY_MODE=standalone|service
   * yaml: deployMode: service
   */
  deployMode: DeployMode;
  server: {
    port: number;
    host: string;
    /**
     * Optional API token for HTTP authentication.
     *
     * When set (non-empty string), every route except `GET /health` and CORS
     * preflight (`OPTIONS *`) requires an `Authorization: Bearer <apiKey>`
     * header. Requests without a valid token receive HTTP 401.
     *
     * **Default: undefined** — authentication is disabled, all routes are
     * open (preserves legacy behaviour). A WARN is emitted at startup if the
     * gateway binds to a non-loopback host without an API key set, to avoid
     * silently exposing an unauthenticated endpoint to the network.
     *
     * env: `TDAI_GATEWAY_API_KEY`
     * yaml: `server.apiKey`
     */
    apiKey?: string;
    /**
     * Optional CORS allow-list.
     *
     * When empty (default), the gateway sends **no** `Access-Control-Allow-*`
     * headers and rejects CORS preflight (`OPTIONS`) with 403 if an `Origin`
     * header is present — browsers will then block all cross-origin requests
     * via same-origin policy.
     *
     * When set, each request's `Origin` is matched against this list and
     * `Access-Control-Allow-Origin` is echoed back only on match. Use the
     * single entry `"*"` to restore the legacy permissive behaviour (only
     * appropriate for local development).
     *
     * env: `TDAI_CORS_ORIGINS` (comma-separated)
     * yaml: `server.corsOrigins` (string[])
     */
    corsOrigins: string[];
  };
  data: {
    /** Base directory for TDAI data storage. */
    baseDir: string;
  };
  llm: StandaloneLLMConfig;
  /** Parsed memory-tdai plugin config (recall, capture, extraction, pipeline, etc.). */
  memory: MemoryTdaiConfig;

  /**
   * Optional Skill module config — passed through to MemoryTdaiConfig.skill
   * at load time. Kept at the top level here so gateway-only deployments
   * can configure skills without nesting under `memory.skill`. The loader
   * shallow-merges this onto `memory.skill` (yaml top-level wins).
   */
  skill?: import("../core/skill/types.js").SkillConfigInput;

  // ── Service-mode config (also settable via env vars, env takes priority) ──

  /** State backend type. env: STATE_BACKEND. yaml: stateBackend */
  stateBackend?: "redis" | "local";
  /** Default instance ID for standalone pipeline. env: TDAI_INSTANCE_ID. yaml: instanceId */
  instanceId: string;
  redis: RedisConfig;
  shark: SharkConfig;
  scanner: ScannerConfig;
  worker: WorkerConfig;
  cos: CosExtraConfig;
  /** 可观测性配置 (yaml: observability, env: KAFKA_METRIC_*) */
  observability: ObservabilityConfig;
  /**
   * 元数据模块配置（env 优先于 yaml；见 applyMetadataEnvFromGatewayConfig）。
   * yaml: metadata.*
   * env: TDAI_METADATA_*
   */
  metadata: GatewayMetadataConfig;
  /** Offload server executor 配置 (yaml: offload) */
  offload: {
    forceTriggerThreshold: number;
    pendingMaxAgeSeconds: number;
    l1Temperature: number;
    l1MaxTokens: number;
    l1TimeoutMs: number;
    l15Temperature: number;
    l15MaxTokens: number;
    l15TimeoutMs: number;
    l2Temperature: number;
    l2MaxTokens: number;
    l2TimeoutMs: number;
    l2NullThreshold: number;
    mildOffloadRatio: number;
    aggressiveCompressRatio: number;
    emergencyCompressRatio: number;
    maxRetries: number;
  };
}

/** v3 元数据 Gateway yaml 配置（§6.3）。 */
export interface GatewayMetadataStoreConfig {
  sqliteBaseDir?: string;
  mongoUri?: string;
  mongoTransactions?: boolean;
  storeCacheMaxInstances?: number;
  /** 元数据库名前缀，默认 tdai_metadata；库名 {prefix}_{instance_id} */
  mongoDbPrefix?: string;
}

export interface GatewayMemorySystemUserConfig {
  userId?: string;
  displayName?: string;
  userKey?: string;
}

export interface GatewayMetadataSystemUserConfig {
  memory?: GatewayMemorySystemUserConfig;
}

export interface GatewayMetadataConfig {
  maxUsersPerInstance: number;
  maxTeamsPerInstance: number;
  configParamsFile?: string;
  store?: GatewayMetadataStoreConfig;
  /** 内部静态系统用户（仅 auth/verify；不落库）。 */
  systemUser?: GatewayMetadataSystemUserConfig;
}

// ============================
// Config loading
// ============================

// ============================
// Utility Functions
// ============================

/**
 * 将 Kafka brokers 配置值转换为 string[] 数组。
 *
 * 背景：gateway config 中 KafkaConfig.brokers 类型为 string（逗号分隔），
 * 但 MetricBackendConfig.brokers 期望 string[]。如果直接传入字符串，
 * KafkaJS 会按字符解析导致端口号变成 NaN。
 *
 * @param brokers - 逗号分隔的 broker 地址字符串，或已经是 string[] 数组
 * @returns broker 地址数组（已 trim，已过滤空元素）
 */
export function parseBrokers(brokers: string | string[]): string[] {
  if (Array.isArray(brokers)) return brokers;
  if (!brokers) return [];
  return brokers.split(",").map(s => s.trim()).filter(Boolean);
}

/**
 * Load gateway config from file + environment variables.
 *
 * Resolution order for config file:
 * 1. `TDAI_GATEWAY_CONFIG` env var (explicit path)
 * 2. `./tdai-gateway.yaml` or `./tdai-gateway.json` in CWD
 * 3. `<dataDir>/tdai-gateway.yaml` or `<dataDir>/tdai-gateway.json`
 * 4. Pure environment-variable config (no file)
 */
export function loadGatewayConfig(overrides?: Partial<GatewayConfig>): GatewayConfig {
  let fileConfig: Record<string, unknown> = {};

  // Try to load config file
  const configPath = resolveConfigPath();
  if (configPath) {
    try {
      const raw = fs.readFileSync(configPath, "utf-8");
      if (configPath.endsWith(".json")) {
        fileConfig = JSON.parse(raw);
      } else {
      // Full YAML support (arbitrary nesting, anchors, lists, multi-line).
        // We still postprocess ${VAR} env-var interpolation on string leaves
        // below so existing configs that relied on the previous simple parser
        // keep working.
        const parsed = YAML.parse(raw);
        fileConfig = (parsed && typeof parsed === "object" && !Array.isArray(parsed))
          ? parsed as Record<string, unknown>
          : {};
      }
      fileConfig = expandEnvVars(fileConfig) as Record<string, unknown>;
    } catch {
      // Config file is optional — malformed files fall back to env-only config.
    }
  }

  // Server config
  const serverConfig = obj(fileConfig, "server");
  const port = envInt("TDAI_GATEWAY_PORT") ?? num(serverConfig, "port") ?? 8420;
  const host = env("TDAI_GATEWAY_HOST") ?? str(serverConfig, "host") ?? "127.0.0.1";

  // Optional auth / CORS — both default to "disabled" so existing setups keep
  // working unchanged. When unset the gateway behaves exactly like before this
  // change (open v1 routes, permissive CORS *will not* be re-introduced — see
  // resolveCorsOrigins below: empty list means "send no CORS headers").
  const apiKey = env("TDAI_GATEWAY_API_KEY") ?? str(serverConfig, "apiKey");
  const corsOrigins = resolveCorsOrigins(serverConfig);

  // Data config (expand leading ~ to $HOME so Node.js fs/path can resolve it)
  const dataConfig = obj(fileConfig, "data");
  const rawBaseDir = env("TDAI_DATA_DIR") ?? str(dataConfig, "baseDir") ?? resolveDefaultDataDir();
  const home = getEnv("HOME") ?? getEnv("USERPROFILE") ?? "/tmp";
  const baseDir = rawBaseDir.startsWith("~/") ? path.join(home, rawBaseDir.slice(2)) : rawBaseDir;

  // LLM config
  //
  // provider 支持两种模式：
  //   - "openai"（默认）：直连 OpenAI 兼容服务
  //   - "proxy"：走 context_proxy，运行期把 baseUrl 拼成 `${baseUrl}/proxy/<iid>/v1`，
  //             Authorization 用 memory 系统用户 key。gateway 层负责最终解析，
  //             见 src/gateway/llm-resolver.ts 的 resolveEffectiveLlmConfig。
  const llmConfig = obj(fileConfig, "llm");
  const llmProxyConfig = obj(llmConfig, "proxy");
  const rawLlmProvider = env("TDAI_LLM_PROVIDER") ?? str(llmConfig, "provider");
  const llmProvider: "openai" | "proxy" =
    rawLlmProvider === "proxy" ? "proxy" : "openai";
  const llm: StandaloneLLMConfig = {
    baseUrl: env("TDAI_LLM_BASE_URL") ?? str(llmConfig, "baseUrl") ?? "https://api.openai.com/v1",
    apiKey: env("TDAI_LLM_API_KEY") ?? str(llmConfig, "apiKey") ?? "",
    model: env("TDAI_LLM_MODEL") ?? str(llmConfig, "model") ?? "gpt-4o",
    maxTokens: envInt("TDAI_LLM_MAX_TOKENS") ?? num(llmConfig, "maxTokens") ?? 4096,
    timeoutMs: envInt("TDAI_LLM_TIMEOUT_MS") ?? num(llmConfig, "timeoutMs") ?? 120_000,
    provider: llmProvider,
    proxy: {
      useMemorySystemUserKey: bool(llmProxyConfig, "useMemorySystemUserKey") ?? true,
    },
  };

  // Memory config (reuse the plugin's parseConfig for full compatibility)
  const memoryRaw = obj(fileConfig, "memory");
  const topLevelPromptMode = str(fileConfig, "promptMode") ?? str(obj(fileConfig, "prompts"), "mode");
  const memoryCompatRaw: Record<string, unknown> = { ...memoryRaw };
  if (!str(memoryCompatRaw, "promptMode") && topLevelPromptMode) {
    memoryCompatRaw.promptMode = topLevelPromptMode;
  }
  const memory = parseMemoryConfig(memoryCompatRaw);

  // ── Standalone LLM override pass-through to MemoryTdaiConfig.llm ──
  // The gateway has its own top-level `llm` block (read above into the
  // `llm` variable), and `parseMemoryConfig` ALSO has a `memory.llm` block
  // that defaults to enabled=false. When the gateway's top-level llm is
  // configured (baseUrl + apiKey), splice it onto memory.llm so the
  // SkillExtractor / L1 / L2 / L3 runners see a usable runner without
  // requiring the user to duplicate the block under `memory.llm`.
  // provider=proxy 模式下即使 apiKey 为空也要 splice —— 因为最终 apiKey 由
  // memory systemUser.userKey 提供，llm.apiKey 只是 provider=openai 时的显式配置。
  const shouldSpliceLlm =
    !memory.llm.enabled && llm.baseUrl && (llm.apiKey || llm.provider === "proxy");
  if (shouldSpliceLlm) {
    memory.llm = {
      enabled: true,
      baseUrl: llm.baseUrl,
      apiKey: llm.apiKey,
      model: llm.model,
      maxTokens: llm.maxTokens ?? 4096,
      timeoutMs: llm.timeoutMs ?? 120_000,
      provider: llm.provider,
      proxy: {
        useMemorySystemUserKey: llm.proxy?.useMemorySystemUserKey ?? true,
      },
    };
  }

  // ── Standalone embedding override pass-through ──
  // Some existing gateway configs place `embedding` at the top level next to
  // `llm`. The memory parser expects it under `memory.embedding`, so splice the
  // top-level block into MemoryTdaiConfig for backward compatibility.
  const topLevelEmbedding = obj(fileConfig, "embedding");
  if (topLevelEmbedding) {
    memory.embedding = {
      ...memory.embedding,
      enabled: bool(topLevelEmbedding, "enabled") ?? memory.embedding.enabled,
      provider: str(topLevelEmbedding, "provider") ?? memory.embedding.provider,
      baseUrl: str(topLevelEmbedding, "baseUrl") ?? memory.embedding.baseUrl,
      apiKey: str(topLevelEmbedding, "apiKey") ?? memory.embedding.apiKey,
      model: str(topLevelEmbedding, "model") ?? memory.embedding.model,
      dimensions: num(topLevelEmbedding, "dimensions") ?? memory.embedding.dimensions,
      sendDimensions: bool(topLevelEmbedding, "sendDimensions") ?? memory.embedding.sendDimensions,
      conflictRecallTopK: num(topLevelEmbedding, "conflictRecallTopK") ?? memory.embedding.conflictRecallTopK,
      maxInputChars: num(topLevelEmbedding, "maxInputChars") ?? memory.embedding.maxInputChars,
      timeoutMs: num(topLevelEmbedding, "timeoutMs") ?? memory.embedding.timeoutMs,
      recallTimeoutMs: num(topLevelEmbedding, "recallTimeoutMs") ?? memory.embedding.recallTimeoutMs,
      captureTimeoutMs: num(topLevelEmbedding, "captureTimeoutMs") ?? memory.embedding.captureTimeoutMs,
      proxyUrl: str(topLevelEmbedding, "proxyUrl") ?? memory.embedding.proxyUrl,
    };
  }

  // ── Skill module config ──
  // Resolution order:
  //   1. Top-level `skill` block in yaml (preferred — skill is a top-level
  //      gateway feature, not buried under `memory`).
  //   2. `memory.skill` (if user nested it under memory by accident).
  //   3. env var TDAI_SKILL_ENABLED forces enabled=true so a one-line
  //      shell flag opts in without touching yaml.
  // The result is *also* spliced onto memory.skill so TdaiCore (which reads
  // from cfg.skill in its config) sees it regardless of which path wired it.
  const topLevelSkill = (fileConfig.skill && typeof fileConfig.skill === "object" && !Array.isArray(fileConfig.skill))
    ? (fileConfig.skill as import("../core/skill/types.js").SkillConfigInput)
    : undefined;
  const envSkillEnabled = env("TDAI_SKILL_ENABLED");
  const skillFromAnywhere: import("../core/skill/types.js").SkillConfigInput | undefined =
    topLevelSkill ?? memory.skill ?? (envSkillEnabled === "true" || envSkillEnabled === "1" ? { enabled: true } : undefined);
  if (skillFromAnywhere) {
    // Last write wins: env-only override forces enabled=true on top of yaml.
    if (envSkillEnabled === "true" || envSkillEnabled === "1") {
      skillFromAnywhere.enabled = true;
    }
    memory.skill = skillFromAnywhere;
  }

  // Deploy mode: "standalone" (open-source single-node) or "service" (cloud multi-tenant)
  const rawMode = env("TDAI_DEPLOY_MODE") ?? str(fileConfig, "deployMode") ?? "standalone";
  const deployMode: DeployMode = rawMode === "service" ? "service" : "standalone";

  // State backend (env > yaml > auto from deployMode)
  const rawBackend = env("STATE_BACKEND") ?? str(fileConfig, "stateBackend");
  const stateBackend = rawBackend === "redis" || rawBackend === "local" ? rawBackend : undefined;

  // Instance ID: service mode requires explicit instanceId from request headers (x-tdai-service-id),
  // standalone mode uses configured or defaults to "default".
  const instanceId = env("TDAI_INSTANCE_ID") ?? str(fileConfig, "instanceId")
    ?? (deployMode === "standalone" ? "default" : undefined);

  // Remote state backend config
  const redisConfig = obj(fileConfig, "redis");
  const redis: RedisConfig = {
    host: env("REDIS_HOST") ?? str(redisConfig, "host") ?? "127.0.0.1",
    port: envInt("REDIS_PORT") ?? num(redisConfig, "port") ?? 6379,
    password: env("REDIS_PASSWORD") ?? str(redisConfig, "password"),
    db: envInt("REDIS_DB") ?? num(redisConfig, "db") ?? 0,
    // v2 默认 prefix：与升级前的 "tdai_memory" 物理隔离。
    // 升级原因：sk/bk 的 hash tag 从 {p:inst} 升级到 {p:inst:tid:aid}，
    // 旧 key 与新 key 不互斥，混跑会破坏锁互斥语义；切 prefix 直接物理隔离，
    // 老数据丢弃。升级时若需保留 redis 中 pending 任务，先停服务 + 等队列空。
    keyPrefix: env("REDIS_KEY_PREFIX") ?? str(redisConfig, "keyPrefix") ?? "tdai_memory_v2",
  };

  // Remote config source settings
  const sharkConfig = obj(fileConfig, "shark");
  const shark: SharkConfig = {
    baseUrl: env("SHARK_BASE_URL") ?? str(sharkConfig, "baseUrl"),
    vdbTtlMs: envInt("CONFIG_VDB_TTL_MS") ?? num(sharkConfig, "vdbTtlMs") ?? 300_000,
    cosBufferMs: envInt("CONFIG_COS_BUFFER_MS") ?? num(sharkConfig, "cosBufferMs") ?? 120_000,
    maxInstances: envInt("CONFIG_MAX_INSTANCES") ?? num(sharkConfig, "maxInstances") ?? 1000,
  };

  // Scanner config
  const scannerConfig = obj(fileConfig, "scanner");
  const scanner: ScannerConfig = {
    instances: env("SCANNER_INSTANCES") ?? str(scannerConfig, "instances") ?? "default",
    instancesSharkUrl: env("SCANNER_INSTANCES_SHARK_URL") ?? str(scannerConfig, "instancesSharkUrl"),
    intervalMs: envInt("SCANNER_INTERVAL_MS") ?? num(scannerConfig, "intervalMs") ?? 500,
    nodeId: env("SCANNER_NODE_ID") ?? str(scannerConfig, "nodeId"),
  };

  // Worker config
  const workerConfig = obj(fileConfig, "worker");
  const worker: WorkerConfig = {
    pollMs: envInt("WORKER_POLL_MS") ?? num(workerConfig, "pollMs") ?? 200,
    concurrency: envInt("WORKER_CONCURRENCY") ?? num(workerConfig, "concurrency") ?? 60,
  };

  // COS extra config
  const cosConfig = obj(fileConfig, "cos");
  const cos: CosExtraConfig = {
    domain: env("COS_DOMAIN") ?? str(cosConfig, "domain"),
  };

  // Observability config (yaml: observability.{otel,clickhouse,kafka}, env 兜底)
  const observabilityConfig = obj(fileConfig, "observability");

  // OTel config
  const otelConfig = obj(observabilityConfig, "otel");
  const otel: OTelConfig = {
    enabled: otelConfig.enabled !== undefined
      ? Boolean(otelConfig.enabled)
      : env("TDAI_OTEL_ENABLED") === "true",
    endpoint: str(otelConfig, "endpoint") ?? env("OTEL_EXPORTER_OTLP_ENDPOINT") ?? "http://localhost:4317",
    protocol: (str(otelConfig, "protocol") ?? env("OTEL_EXPORTER_OTLP_PROTOCOL") ?? "grpc") as "grpc" | "http/protobuf",
    serviceName: str(otelConfig, "serviceName") ?? env("OTEL_SERVICE_NAME") ?? "core",
    serviceVersion: str(otelConfig, "serviceVersion") ?? "1.0.0",
    tenantId: str(otelConfig, "tenantId") ?? env("OTEL_TENANT_ID") ?? "",
    logExportInterval: num(otelConfig, "logExportInterval") ?? envInt("OTEL_LOG_EXPORT_INTERVAL") ?? 5,
  };

  // ClickHouse config
  const chConfig = obj(observabilityConfig, "clickhouse");
  const clickhouse: ClickHouseConfig = {
    enabled: chConfig.enabled !== undefined
      ? Boolean(chConfig.enabled)
      : env("CLICKHOUSE_ENABLED") === "true",
    endpoint: str(chConfig, "endpoint") ?? env("CLICKHOUSE_ENDPOINT") ?? "",
    username: str(chConfig, "username") ?? env("CLICKHOUSE_USERNAME") ?? "default",
    password: str(chConfig, "password") ?? env("CLICKHOUSE_PASSWORD") ?? "",
    database: str(chConfig, "database") ?? env("CLICKHOUSE_DATABASE") ?? "tdai_eval",
  };

  // Kafka config
  const kafkaConfig = obj(observabilityConfig, "kafka");
  const kafka: KafkaConfig = {
    brokers: str(kafkaConfig, "brokers") ?? env("KAFKA_METRIC_BROKERS") ?? "",
    topic: str(kafkaConfig, "topic") ?? env("KAFKA_METRIC_TOPIC") ?? "memory_monitor",
    enabled: kafkaConfig.enabled !== undefined
      ? Boolean(kafkaConfig.enabled)
      : (env("KAFKA_METRIC_ENABLED") === "true" || Boolean(str(kafkaConfig, "brokers") ?? env("KAFKA_METRIC_BROKERS"))),
  };

  // Langfuse config
  const langfuseConfig = obj(observabilityConfig, "langfuse");
  const langfuse: LangfuseConfig = {
    enabled: langfuseConfig.enabled !== undefined
      ? Boolean(langfuseConfig.enabled)
      : env("LANGFUSE_ENABLED") === "true",
    host: str(langfuseConfig, "host") ?? env("LANGFUSE_HOST") ?? "",
    publicKey: str(langfuseConfig, "publicKey") ?? env("LANGFUSE_PUBLIC_KEY") ?? "",
    secretKey: str(langfuseConfig, "secretKey") ?? env("LANGFUSE_SECRET_KEY") ?? "",
  };

  const observability: ObservabilityConfig = { otel, clickhouse, kafka, langfuse };

  // Offload executor config (yaml: offload)
  const metadataConfig = obj(fileConfig, "metadata");
  const storeYaml = obj(metadataConfig, "store");
  const systemUserYaml = obj(metadataConfig, "systemUser");
  const memorySystemUserYaml = obj(systemUserYaml, "memory");
  const metadataStore: GatewayMetadataStoreConfig | undefined =
    str(storeYaml, "sqliteBaseDir") || str(storeYaml, "mongoUri") ||
    str(storeYaml, "mongoDbPrefix") ||
    bool(storeYaml, "mongoTransactions") !== undefined || num(storeYaml, "storeCacheMaxInstances") !== undefined
      ? {
          sqliteBaseDir: str(storeYaml, "sqliteBaseDir"),
          mongoUri: str(storeYaml, "mongoUri"),
          mongoTransactions: bool(storeYaml, "mongoTransactions") ?? true,
          storeCacheMaxInstances: num(storeYaml, "storeCacheMaxInstances"),
          mongoDbPrefix: str(storeYaml, "mongoDbPrefix"),
        }
      : undefined;
  const metadata: GatewayMetadataConfig = {
    maxUsersPerInstance: positiveQuotaLimit(
      envInt("TDAI_METADATA_MAX_USERS") ?? num(metadataConfig, "maxUsersPerInstance"),
      500,
    ),
    maxTeamsPerInstance: positiveQuotaLimit(
      envInt("TDAI_METADATA_MAX_TEAMS") ?? num(metadataConfig, "maxTeamsPerInstance"),
      100,
    ),
    configParamsFile: process.env.TDAI_METADATA_CONFIG_PARAMS_FILE || str(metadataConfig, "configParamsFile") || undefined,
    store: metadataStore,
    systemUser: str(memorySystemUserYaml, "userId") || str(memorySystemUserYaml, "userKey") ||
      str(memorySystemUserYaml, "displayName")
      ? {
          memory: {
            userId: str(memorySystemUserYaml, "userId"),
            displayName: str(memorySystemUserYaml, "displayName"),
            userKey: str(memorySystemUserYaml, "userKey"),
          },
        }
      : undefined,
  };

  const offloadConfig = obj(fileConfig, "offload");
  const offload = {
    forceTriggerThreshold: num(offloadConfig, "forceTriggerThreshold") ?? 4,
    pendingMaxAgeSeconds: num(offloadConfig, "pendingMaxAgeSeconds") ?? 30,
    l1Temperature: num(offloadConfig, "l1Temperature") ?? 0.3,
    l1MaxTokens: num(offloadConfig, "l1MaxTokens") ?? 8000,
    l1TimeoutMs: num(offloadConfig, "l1TimeoutMs") ?? 120_000,
    l15Temperature: num(offloadConfig, "l15Temperature") ?? 0.2,
    l15MaxTokens: num(offloadConfig, "l15MaxTokens") ?? 3000,
    l15TimeoutMs: num(offloadConfig, "l15TimeoutMs") ?? 120_000,
    l2Temperature: num(offloadConfig, "l2Temperature") ?? 0.4,
    l2MaxTokens: num(offloadConfig, "l2MaxTokens") ?? 16000,
    l2TimeoutMs: num(offloadConfig, "l2TimeoutMs") ?? 120_000,
    l2NullThreshold: num(offloadConfig, "l2NullThreshold") ?? 6,
    mildOffloadRatio: num(offloadConfig, "mildOffloadRatio") ?? 0.5,
    aggressiveCompressRatio: num(offloadConfig, "aggressiveCompressRatio") ?? 0.85,
    emergencyCompressRatio: num(offloadConfig, "emergencyCompressRatio") ?? 0.95,
    maxRetries: num(offloadConfig, "maxRetries") ?? 3,
  };

  const base: GatewayConfig = {
    deployMode,
    stateBackend,
    instanceId,
    server: { port, host, apiKey, corsOrigins },
    data: { baseDir },
    llm,
    memory,
    redis,
    shark,
    scanner,
    worker,
    cos,
    observability,
    metadata,
    offload,
    skill: skillFromAnywhere,
  };

  // Merge overrides one level deep so partial `server`/`data`/`llm` patches
  // (frequently used by e2e tests) don't accidentally drop sibling fields
  // such as `corsOrigins` introduced after they were written.
  if (!overrides) return base;
  return {
    ...base,
    ...overrides,
    server: { ...base.server, ...(overrides.server ?? {}) },
    data: { ...base.data, ...(overrides.data ?? {}) },
    llm: { ...base.llm, ...(overrides.llm ?? {}) },
  };
}

// ============================
// Helpers
// ============================

function resolveConfigPath(): string | null {
  // 1. Explicit env var
  const explicit = getEnv("TDAI_GATEWAY_CONFIG")?.trim();
  if (explicit && fs.existsSync(explicit)) return explicit;

  // 2. CWD
  for (const name of ["tdai-gateway.yaml", "tdai-gateway.json"]) {
    const p = path.join(process.cwd(), name);
    if (fs.existsSync(p)) return p;
  }

  // 3. Default data dir
  const dataDir = resolveDefaultDataDir();
  for (const name of ["tdai-gateway.yaml", "tdai-gateway.json"]) {
    const p = path.join(dataDir, name);
    if (fs.existsSync(p)) return p;
  }

  return null;
}

function resolveDefaultDataDir(): string {
  const home = getEnv("HOME") ?? getEnv("USERPROFILE") ?? "/tmp";

  // New canonical location: everything related to standalone/Hermes-mode TDAI
  // is collected under ~/.memory-tencentdb/ to avoid scattering top-level dirs
  // in $HOME. The Gateway data dir lives at:
  //
  //   ~/.memory-tencentdb/memory-tdai/
  //
  // Note: this only governs the standalone/Hermes fallback. Under the openclaw
  // host the plugin data dir is decided by `resolveStateDir() + "memory-tdai"`
  // (typically ~/.openclaw/memory-tdai/) which is intentionally NOT changed.
  const root = getEnv("MEMORY_TENCENTDB_ROOT") ?? path.join(home, ".memory-tencentdb");
  const newDefault = path.join(root, "memory-tdai");

  // Backward compatibility: if the new location does not yet exist but the
  // legacy ~/memory-tdai still has data, keep using the legacy dir so existing
  // users don't silently lose their memory store. The install script
  // (install_hermes_memory_tencentdb.sh, Step 0) will migrate it on next run.
  try {
    if (!fs.existsSync(newDefault)) {
      const legacy = path.join(home, "memory-tdai");
      if (fs.existsSync(legacy)) {
        // Stderr-only deprecation hint; doesn't pollute structured logs.
        process.stderr.write(
          `[tdai-gateway] DEPRECATED: using legacy data dir ${legacy}; ` +
          `move it to ${newDefault} (or set TDAI_DATA_DIR / MEMORY_TENCENTDB_ROOT) to silence this warning.\n`,
        );
        return legacy;
      }
    }
  } catch {
    // existsSync should not throw, but guard anyway.
  }

  return newDefault;
}

function env(key: string): string | undefined {
  const v = getEnv(key)?.trim();
  return v || undefined;
}

function envInt(key: string): number | undefined {
  const v = env(key);
  if (!v) return undefined;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : undefined;
}

function obj(c: Record<string, unknown>, key: string): Record<string, unknown> {
  const v = c[key];
  return v && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : {};
}

function str(src: Record<string, unknown>, key: string): string | undefined {
  const v = src[key];
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

function num(src: Record<string, unknown>, key: string): number | undefined {
  const v = src[key];
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function positiveQuotaLimit(value: number | undefined, fallback: number): number {
  return value !== undefined && value > 0 ? value : fallback;
}

function bool(src: Record<string, unknown>, key: string): boolean | undefined {
  const v = src[key];
  return typeof v === "boolean" ? v : undefined;
}

/**
 * Read `server.corsOrigins` from yaml or `TDAI_CORS_ORIGINS` from env.
 *
 * Accepted yaml shapes (yaml has precedence over env):
 *   server:
 *     corsOrigins: []                              # explicit empty → no CORS
 *     corsOrigins: ["https://app.example.com"]     # array of allowed origins
 *     corsOrigins: "https://a,https://b"           # comma-separated string
 *
 * Env: `TDAI_CORS_ORIGINS="https://a,https://b"`
 *
 * Returns `[]` when nothing is set — the server interprets that as
 * "do not emit any CORS headers" (most restrictive default).
 */
function resolveCorsOrigins(serverConfig: Record<string, unknown>): string[] {
  // 1. YAML takes precedence so an explicit `corsOrigins: []` can mean
  //    "I want CORS off" even when the env var leaks in from the shell.
  const raw = serverConfig["corsOrigins"];
  if (Array.isArray(raw)) {
    return raw.filter((s): s is string => typeof s === "string" && s.trim().length > 0).map(s => s.trim());
  }
  if (typeof raw === "string" && raw.trim()) {
    return raw.split(",").map(s => s.trim()).filter(Boolean);
  }

  // 2. Fall back to env. Empty string from env is treated as "not set".
  const envValue = env("TDAI_CORS_ORIGINS");
  if (!envValue) return [];
  return envValue.split(",").map(s => s.trim()).filter(Boolean);
}

/**
 * Recursively replace ``${VAR_NAME}`` placeholders in string leaves with
 * the corresponding ``process.env`` value. Missing variables expand to an
 * empty string, matching the behaviour of the previous simple YAML parser
 * so existing configs keep working after the switch to the full YAML lib.
 *
 * - Only whole-string matches (``"${VAR}"``) are substituted, preserving
 *   types: numbers/booleans/null pass through unchanged.
 * - Arrays and nested objects are walked in-place (new arrays/objects are
 *   returned; the input is not mutated).
 */
function expandEnvVars(value: unknown): unknown {
  if (typeof value === "string") {
    const m = value.match(/^\$\{(\w+)\}$/);
    if (m) {
      return process.env[m[1]!] ?? "";
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(expandEnvVars);
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = expandEnvVars(v);
    }
    return out;
  }
  return value;
}
