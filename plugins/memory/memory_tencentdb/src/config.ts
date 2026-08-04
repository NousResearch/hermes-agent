/**
 * Plugin configuration types and parser (v3).
 *
 * Config is organized into flat functional groups:
 *   capture, extraction, persona, pipeline, recall, embedding
 *
 * Minimal config (zero config): {} — all fields have sensible defaults.
 */

// ============================
// Type definitions
// ============================

/** Prompt family for L1-L3 memory pipeline. */
export type MemoryPromptMode = "chat" | "code";

/** Capture settings — controls L0 conversation recording. */
export interface CaptureConfig {
  /** Enable auto-capture (default: true) */
  enabled: boolean;
  /** Glob patterns to exclude agents (e.g. "bench-judge-*"); matched agents are fully ignored */
  excludeAgents: string[];
  /**
   * L0/L1 local file retention days used as TTL switch.
   * 0 means cleanup disabled.(default: 0)
   */
  l0l1RetentionDays: number;

  /**
   * Allow dangerous low retention (1 or 2 days).
   * Default false: when disabled, non-zero retention must be >= 3.
   */
  allowAggressiveCleanup: boolean;
}

/** Extraction settings (L1) — controls memory extraction from conversations. */
export interface ExtractionConfig {
  /** Enable background extraction (default: true) */
  enabled: boolean;
  /** Enable L1 smart dedup (default: true) */
  enableDedup: boolean;
  /** Max memories per session (default: 20) */
  maxMemoriesPerSession: number;
  /** LLM model for extraction, format: "provider/model" (falls back to OpenClaw default model when omitted) */
  model?: string;
  /** Prompt family for L1 extraction (default: chat). */
  promptMode: MemoryPromptMode;
}

/** Persona (L2/L3) settings — controls scene extraction (L2) and user profile generation (L3). */
export interface PersonaConfig {
  /** Trigger persona generation every N new memories (default: 50) */
  triggerEveryN: number;
  /** Max scene blocks (default: 20) */
  maxScenes: number;
  /** Persona backup count (default: 3) */
  backupCount: number;
  /** Scene blocks backup count (default: 10) */
  sceneBackupCount: number;
  /** LLM model for persona generation, format: "provider/model" (falls back to OpenClaw default model when omitted) */
  model?: string;
  /** Prompt family for L2/L3 prompts (default: chat; code prompts will be wired as they are added). */
  promptMode: MemoryPromptMode;
}

/** Pipeline trigger settings (L1→L2→L3 scheduling). */
export interface PipelineTriggerConfig {
  /** Trigger L1 after every N conversation rounds (default: 5) */
  everyNConversations: number;
  /** Enable warm-up: start threshold at 1, double after each L1 (1→2→4→...→everyN) (default: true) */
  enableWarmup: boolean;
  /** L1 idle timeout: trigger L1 after this many seconds of inactivity (default: 30) */
  l1IdleTimeoutSeconds: number;
  /** L2 delay after L1: wait this many seconds after L1 completes before triggering L2 (default: 90) */
  l2DelayAfterL1Seconds: number;
  /** L2 min interval: minimum seconds between L2 runs per session (default: 900 = 15 min) */
  l2MinIntervalSeconds: number;
  /** L2 max interval: even without new conversations, trigger L2 at most this often per session (default: 3600 = 60 min) */
  l2MaxIntervalSeconds: number;
  /** Sessions inactive longer than this (hours) stop L2 polling (default: 24) */
  sessionActiveWindowHours: number;
}

/** Recall settings — controls memory retrieval for context injection. */
export interface RecallConfig {
  /** Enable auto-recall (default: true) */
  enabled: boolean;
  /** Max results to return (default: 5) */
  maxResults: number;
  /** Max characters injected for a single recalled L1 memory. 0 disables the per-memory limit. */
  maxCharsPerMemory: number;
  /** Max total characters injected for all recalled L1 memories. 0 disables the total limit. */
  maxTotalRecallChars: number;
  /** Minimum score threshold (default: 0.3) */
  scoreThreshold: number;
  /** Search strategy (default: "hybrid") */
  strategy: "embedding" | "keyword" | "hybrid";
  /** Overall recall timeout in milliseconds (default: 5000). When exceeded, recall is skipped with a warning. */
  timeoutMs: number;
}

/** Embedding service configuration for vector search. */
export interface EmbeddingConfig {
  /** User-facing default is true in schema, but provider="none" still disables embedding effectively. */
  enabled: boolean;
  /** Embedding provider: default "none" disables vector search; other values (e.g. "openai", "deepseek") are treated as OpenAI-compatible remote providers. */
  provider: string;
  /** API Base URL (required for remote provider). */
  baseUrl: string;
  /** API Key (required for remote provider). */
  apiKey: string;
  /** Model name (required for remote provider). */
  model: string;
  /** Vector dimensions (required for remote provider, must match model). */
  dimensions: number;
  /**
   * Whether to send the `dimensions` field in the embeddings request body.
   * Default true (compatible with OpenAI text-embedding-3-* Matryoshka models).
   * Set to false for self-hosted / OSS models that reject unknown `dimensions`
   * (e.g. BGE-M3, which returns HTTP 400 "does not support matryoshka representation").
   */
  sendDimensions: boolean;
  /** Top-K candidates to recall during conflict detection (default: 5) */
  conflictRecallTopK: number;
  /** Proxy URL for qclaw provider — when provider="qclaw", requests are forwarded through this local proxy */
  proxyUrl?: string;
  /** Max input text length in characters before truncation (default: 5000). Texts exceeding this limit are truncated with a warning. */
  maxInputChars: number;
  /** Timeout per embedding API call in milliseconds (default: 10000). */
  timeoutMs: number;
  /** Override timeoutMs for recall-path embedding calls (user-facing, should be shorter). Falls back to timeoutMs. */
  recallTimeoutMs?: number;
  /** Override timeoutMs for capture-path embedding calls (background L1 dedup, can be longer). Falls back to timeoutMs. */
  captureTimeoutMs?: number;
  /** Internal-only local model cache directory, not exposed in plugin schema. */
  modelCacheDir?: string;
  /** If set, contains an error message about invalid remote config (embedding is disabled) */
  configError?: string;
}

/** Daily cleaner settings for local JSONL data (L0/L1). */
export interface MemoryCleanupConfig {
  /** TTL switch from capture.l0l1RetentionDays. Undefined means disabled. */
  retentionDays?: number;

  /** Whether cleanup is enabled. True only when retentionDays is a valid positive number. */
  enabled: boolean;
  /** Daily execution time in HH:mm format (default: 03:00). */
  cleanTime: string;
}

/** BM25 sparse vector encoding configuration (local @tencentdb-agent-memory/tcvdb-text). */
export interface BM25Config {
  /** Whether BM25 sparse encoding is enabled (default: true) */
  enabled: boolean;
  /** Language for BM25 pre-trained params: "zh" or "en" (default: "zh") */
  language: "zh" | "en";
}

/** Tencent Cloud VectorDB configuration. */
export interface TcvdbConfig {
  /** Instance URL (e.g. "http://10.0.1.1:80" or external domain) */
  url: string;
  /** Account name (default: "root") */
  username: string;
  /** API Key */
  apiKey: string;
  /** Database name (auto-generated from instance_id if empty) */
  database: string;
  /** User-friendly alias for this database (optional, for identification in database.json) */
  alias: string;
  /** Whether to enable VectorDB server-side dense embedding/vector index. Default false: BM25 sparse only. */
  embeddingEnabled?: boolean;
  /** Built-in embedding model (default: "bge-large-zh"; used only when embeddingEnabled=true) */
  embeddingModel: string;
  /** Request timeout in ms (default: 10000) */
  timeout: number;
  /** Path to CA certificate PEM file (for HTTPS connections) */
  caPemPath?: string;
}

/** Storage backend type. */
export type StoreBackend = "sqlite" | "tcvdb";

/** Report settings — controls metric/event reporting. */
export interface ReportConfig {
  /** Enable reporting (default: true) */
  enabled: boolean;
  /** Reporter type: "local" logs structured JSON via logger (default: "local") */
  type: string;
}

/**
 * Standalone LLM configuration — when set, TDAI uses direct API calls
 * instead of the host's built-in LLM runner (e.g. OpenClaw's runEmbeddedPiAgent).
 *
 * This allows using a different (often cheaper/faster) model for memory
 * extraction while the main agent uses a premium model.
 *
 * Leave undefined (default) to use the host's native LLM mechanism.
 */
export interface StandaloneLLMOverrideConfig {
  /** Enable standalone LLM mode (default: false). When false, uses host LLM. */
  enabled: boolean;
  /** OpenAI-compatible API base URL (e.g. "https://api.openai.com/v1"). */
  baseUrl: string;
  /** API key for authentication. */
  apiKey: string;
  /** Model name (e.g. "gpt-4o", "deepseek-v3", "claude-sonnet-4-6"). */
  model: string;
  /** Max output tokens (default: 4096). */
  maxTokens: number;
  /** Request timeout in milliseconds (default: 120000). */
  timeoutMs: number;
  /**
   * LLM 访问模式：
   *   - "openai": 直连 OpenAI 兼容服务（默认）
   *   - "proxy":  走 context_proxy，运行时把 baseUrl 拼成
   *               `${baseUrl}/proxy/<instanceId>/v1`，Authorization 用 memory 系统用户 key
   * gateway 层负责在构造 runner 前把 baseUrl / apiKey 换成解析后的最终值，
   * 因此 runner 无需感知 provider 字段。
   */
  provider?: "openai" | "proxy";
  /** provider=proxy 时的可选配置。 */
  proxy?: {
    /** 是否用 memory systemUser.userKey 作为 Authorization（默认 true）。 */
    useMemorySystemUserKey?: boolean;
  };
}

/** Context Offload settings — controls multi-layer context compression. */
export interface OffloadConfig {
  /** Enable context offload (default: false) */
  enabled: boolean;
  /**
   * LLM execution mode for L1/L1.5/L2 tasks.
   * - "local": call LLM directly via AI SDK (uses offload.model or main agent model)
   * - "backend": route through remote backend service (requires backendUrl)
   * - "client": stateless client mode — all compression delegated to offload server v2
   * - "collect": data collection only — runs L1/L1.5/L2 asynchronously but disables
   *   L3 compression and does NOT occupy the contextEngine slot (uses legacy compaction)
   * Default: "local" (auto-detects based on backendUrl presence for backward compat)
   */
  mode: "local" | "backend" | "client" | "collect";
  /** LLM model for offload tasks, format: "provider/model-id". Falls back to agents.defaults.model when omitted. */
  model?: string;
  /** LLM temperature (default: 0.2) */
  temperature: number;
  /** Force-trigger L1 when pending tool pairs >= this threshold (default: 4) */
  forceTriggerThreshold: number;
  /** Custom data directory (absolute path). Default: ~/.openclaw/context-offload */
  dataDir?: string;
  /** Default context window size (default: 200000) */
  defaultContextWindow: number;
  /** Max tool pairs per L1 batch (default: 20) */
  maxPairsPerBatch: number;
  /** Trigger L2 when node_id=null entries >= this count (default: 4) */
  l2NullThreshold: number;
  /** Trigger L2 if hasn't run for this many seconds (default: 300) */
  l2TimeoutSeconds: number;
  /** Mild compression ratio threshold (default: 0.5) */
  mildOffloadRatio: number;
  /** Aggressive compression ratio threshold (default: 0.85) */
  aggressiveCompressRatio: number;
  /** MMD injection token budget ratio (default: 0.2) */
  mmdMaxTokenRatio: number;
  /** Backend service URL. When set, L1/L1.5/L2/L4 LLM calls go through the backend. */
  backendUrl?: string;
  /** Backend API authentication token */
  backendApiKey?: string;
  /** Backend call timeout in milliseconds (default: 10000) */
  backendTimeoutMs: number;
  /**
   * Offload data retention days. Sessions/refs/mmds older than this are cleaned up.
   * 0 = disabled (default). Values in (0, 3) are treated as invalid and forced to 0.
   * Minimum effective value: 3.
   */
  offloadRetentionDays: number;
  /**
   * Max total size in MB for offload debug log files (*.log in dataRoot).
   * When exceeded, the largest logs are truncated to zero.
   * 0 = disabled. Default: 50.
   */
  logMaxSizeMb: number;
  /**
   * User identifier sent as `X-User-Id` on backend requests. This is the
   * primary key used by the backend `/offload/v1/store` endpoint to upsert
   * per-user state. When omitted the plugin falls back to the machine's
   * primary non-loopback IPv4 address.
   */
  userId?: string;

  // ── Client mode fields (used when mode === "client") ──────────────
  /** Offload server v2 base URL (e.g. "http://localhost:9100"). */
  serverUrl?: string;
  /** Bearer token for offload server v2 Authorization header. */
  apiKey?: string;
  /** X-TDAI-Service-Id header value. */
  serviceId?: string;
  /** Agent name used in storage path (default: "default"). */
  agentName?: string;
  /** Client-side threshold: skip compaction when ratio < this value (default: 0.5). */
  compactionRatio?: number;
  /** Ingest request timeout in ms (default: 5000). */
  ingestTimeoutMs?: number;
  /** Compaction request timeout in ms (default: 30000). */
  compactionTimeoutMs?: number;
}

/** Fully resolved plugin configuration (v3). */
export interface MemoryTdaiConfig {
  /** Global prompt family; group-level promptMode can override it. */
  promptMode: MemoryPromptMode;
  capture: CaptureConfig;
  extraction: ExtractionConfig;
  persona: PersonaConfig;
  pipeline: PipelineTriggerConfig;
  recall: RecallConfig;
  embedding: EmbeddingConfig;
  /** Storage backend: "sqlite" (default) or "tcvdb" */
  storeBackend: StoreBackend;
  /** Tencent Cloud VectorDB configuration (required when storeBackend = "tcvdb") */
  tcvdb: TcvdbConfig;
  /** BM25 sparse vector encoding (local @tencentdb-agent-memory/tcvdb-text) */
  bm25: BM25Config;
  /** Local JSONL cleanup settings */
  memoryCleanup: MemoryCleanupConfig;
  report: ReportConfig;
  /**
   * Standalone LLM override — when enabled, TDAI bypasses the host's LLM
   * (e.g. OpenClaw's runEmbeddedPiAgent) and uses direct OpenAI-compatible
   * API calls for L1/L2/L3 extraction.
   *
   * Default: disabled (uses host LLM).
   */
  llm: StandaloneLLMOverrideConfig;
  offload: OffloadConfig;
  /**
   * Optional Skill module config. Pass-through to `resolveSkillConfig()` at
   * the host wiring layer; defaults are applied there. When absent, the
   * Skill module is not constructed (host-side gate). Shape: SkillConfigInput.
   */
  skill?: import("./core/skill/types.js").SkillConfigInput;
}

// ============================
// Parser
// ============================

/**
 * Parse plugin config from raw user input.
 * All fields have sensible defaults — minimal config is just {}.
 */
export function parseConfig(raw: Record<string, unknown> | undefined): MemoryTdaiConfig {
  const c = raw ?? {};

  // --- Prompt mode (L1-L3) ---
  const promptsGroup = obj(c, "prompts");
  const globalPromptMode = normalizePromptMode(str(c, "promptMode") ?? str(promptsGroup, "mode"));

  // --- Capture (L0) ---
  const captureGroup = obj(c, "capture");

  // --- Retention days validation (from capture.l0l1RetentionDays) ---
  const rawRetentionDays = num(captureGroup, "l0l1RetentionDays") ?? 0;
  const allowAggressiveCleanup = bool(captureGroup, "allowAggressiveCleanup") ?? false;

  let retentionDays: number | undefined;
  if (rawRetentionDays <= 0) {
    retentionDays = undefined;
  } else if (rawRetentionDays >= 3) {
    retentionDays = rawRetentionDays;
  } else if (allowAggressiveCleanup) {
    retentionDays = rawRetentionDays;
  } else {
    retentionDays = undefined;
  }

  // --- Extraction (L1) ---
  const extractionGroup = obj(c, "extraction");

  // --- Persona (L2/L3) ---
  const personaGroup = obj(c, "persona");

  // --- Pipeline ---
  const pipelineGroup = obj(c, "pipeline");

  // --- Recall ---
  const recallGroup = obj(c, "recall");

  // --- Embedding ---
  const embeddingGroup = obj(c, "embedding");
  let embeddingConfigError: string | undefined;

  // Embedding config: determine provider based on user input and apiKey availability
  const embeddingApiKey = str(embeddingGroup, "apiKey") ?? "";
  const embeddingBaseUrl = str(embeddingGroup, "baseUrl") ?? "";
  const embeddingProviderRaw = str(embeddingGroup, "provider") ?? "none";
  const embeddingModelRaw = str(embeddingGroup, "model") ?? "";
  const embeddingDimensionsRaw = num(embeddingGroup, "dimensions");
  const embeddingProxyUrl = str(embeddingGroup, "proxyUrl");

  // provider="none" → embedding disabled (default for zero-config users)
  // provider="local" → no longer exposed to users; treated as disabled at entry level
  // provider="qclaw" → requires proxyUrl for local proxy forwarding
  // Any other value → remote mode (requires apiKey, baseUrl, model, dimensions)
  let embeddingProvider: string;
  let embeddingEnabled = bool(embeddingGroup, "enabled") ?? true;

  if (embeddingProviderRaw === "none") {
    // Explicitly disabled (default): no embedding, no vector search
    embeddingProvider = "none";
    embeddingEnabled = false;
  } else if (embeddingProviderRaw === "local") {
    // Local embedding is not exposed to users; treat as disabled at entry level.
    // Internal LocalEmbeddingService code is preserved but not reachable from config.
    embeddingProvider = "none";
    embeddingEnabled = false;
    embeddingConfigError =
      "Local embedding provider is not available in user config. " +
      "Please configure a remote embedding provider (e.g. openai, deepseek). Embedding has been disabled.";
  } else if (embeddingProviderRaw === "qclaw") {
    // qclaw provider: requires proxyUrl for local proxy forwarding
    const missingFields: string[] = [];
    if (!embeddingProxyUrl) missingFields.push("proxyUrl");
    if (!embeddingBaseUrl) missingFields.push("baseUrl");
    if (!embeddingApiKey) missingFields.push("apiKey");
    if (!embeddingModelRaw) missingFields.push("model");
    if (embeddingDimensionsRaw == null || embeddingDimensionsRaw <= 0) missingFields.push("dimensions");

    if (missingFields.length > 0) {
      const errorMsg =
        `Embedding provider 'qclaw' requires 'proxyUrl', 'baseUrl', 'apiKey', 'model', and 'dimensions' to be set. ` +
        `Missing: ${missingFields.join(", ")}. Embedding has been disabled.`;
      embeddingConfigError = errorMsg;
      embeddingEnabled = false;
      embeddingProvider = embeddingProviderRaw;
    } else {
      embeddingProvider = embeddingProviderRaw;
    }
  } else {
    // Remote mode — validate all required fields
    const missingFields: string[] = [];
    if (!embeddingApiKey) missingFields.push("apiKey");
    if (!embeddingBaseUrl) missingFields.push("baseUrl");
    if (!embeddingModelRaw) missingFields.push("model");
    if (embeddingDimensionsRaw == null || embeddingDimensionsRaw <= 0) missingFields.push("dimensions");

    if (missingFields.length > 0) {
      // Configuration error: disable embedding and log detailed error
      // This does NOT throw — the plugin continues running without vector search
      const errorMsg =
        `Remote embedding provider '${embeddingProviderRaw}' requires 'apiKey', 'baseUrl', 'model', and 'dimensions' to be set. ` +
        `Missing: ${missingFields.join(", ")}. Embedding has been disabled.`;
      // We store the error message so the caller (index.ts) can log it
      embeddingConfigError = errorMsg;
      embeddingEnabled = false;
      embeddingProvider = embeddingProviderRaw; // preserve original for error context
    } else {
      embeddingProvider = embeddingProviderRaw;
    }
  }

  // When provider="none", dimensions=0 signals VectorStore to skip vec0 table
  // creation entirely (deferred until a real embedding provider is configured).
  // This avoids creating vec0 tables with a placeholder dimension that would
  // mismatch if the user later enables a different-dimensional provider.
  const defaultDimensions =
    embeddingProvider === "none" ? 0 :
    embeddingDimensionsRaw ?? 0;
  const defaultModel = embeddingProvider === "none" ? "" : embeddingModelRaw;

  const cleanTime = normalizeCleanTime(str(captureGroup, "cleanTime")) ?? "03:00";

  // --- BM25 (local @tencentdb-agent-memory/tcvdb-text encoder) ---
  const bm25Group = obj(c, "bm25");

  // --- Store backend ---
  const storeBackendRaw = str(c, "storeBackend") ?? "sqlite";
  const storeBackend: StoreBackend = storeBackendRaw === "tcvdb" ? "tcvdb" : "sqlite";

  // --- TCVDB config ---
  const tcvdbGroup = obj(c, "tcvdb");

  const memoryCleanup: MemoryCleanupConfig = {
    retentionDays,
    enabled: retentionDays != null,
    cleanTime,
  };

  // --- Offload ---
  const offloadGroup = obj(c, "offload");

  // Auto-derive offload serverUrl/apiKey/serviceId from top-level server config
  // when offload client fields are not explicitly set.
  const serverGroup = obj(c, "server");
  const serverUrl = optStr(serverGroup, "url");
  const serverApiKey = optStr(serverGroup, "apiKey");
  const serverInstanceId = optStr(serverGroup, "instanceId");

  const offloadMode: "local" | "backend" | "client" | "collect" = (() => {
    const raw = optStr(offloadGroup, "mode");
    if (raw === "local" || raw === "backend" || raw === "client" || raw === "collect") return raw;
    // Auto-derive: if backendUrl is set → "backend"; if server.url is set → "client"; else "local"
    if (optStr(offloadGroup, "backendUrl")) return "backend";
    if (serverUrl) return "client";
    return "local";
  })();

  const offload: OffloadConfig = {
    enabled: bool(offloadGroup, "enabled") ?? false,
    mode: offloadMode,
    model: optStr(offloadGroup, "model"),
    temperature: num(offloadGroup, "temperature") ?? 0.2,
    forceTriggerThreshold: num(offloadGroup, "forceTriggerThreshold") ?? 4,
    dataDir: optStr(offloadGroup, "dataDir"),
    defaultContextWindow: num(offloadGroup, "defaultContextWindow") ?? 200000,
    maxPairsPerBatch: num(offloadGroup, "maxPairsPerBatch") ?? 20,
    l2NullThreshold: num(offloadGroup, "l2NullThreshold") ?? 4,
    l2TimeoutSeconds: num(offloadGroup, "l2TimeoutSeconds") ?? 300,
    mildOffloadRatio: num(offloadGroup, "mildOffloadRatio") ?? 0.5,
    aggressiveCompressRatio: num(offloadGroup, "aggressiveCompressRatio") ?? 0.85,
    mmdMaxTokenRatio: num(offloadGroup, "mmdMaxTokenRatio") ?? 0.2,
    backendUrl: optStr(offloadGroup, "backendUrl"),
    backendApiKey: optStr(offloadGroup, "backendApiKey"),
    backendTimeoutMs: num(offloadGroup, "backendTimeoutMs") ?? 120000,
    offloadRetentionDays: normalizeOffloadRetentionDays(num(offloadGroup, "offloadRetentionDays") ?? 0),
    logMaxSizeMb: num(offloadGroup, "logMaxSizeMb") ?? 50,
    userId: optStr(offloadGroup, "userId"),
    // Client mode fields — fall back to top-level server config when not explicitly set
    serverUrl: optStr(offloadGroup, "serverUrl") ?? serverUrl,
    apiKey: optStr(offloadGroup, "apiKey") ?? serverApiKey,
    serviceId: optStr(offloadGroup, "serviceId") ?? serverInstanceId,
    compactionRatio: num(offloadGroup, "compactionRatio") ?? 0.5,
    ingestTimeoutMs: num(offloadGroup, "ingestTimeoutMs") ?? 5000,
    compactionTimeoutMs: num(offloadGroup, "compactionTimeoutMs") ?? 30000,
  };

  return {
    promptMode: globalPromptMode,
    capture: {
      enabled: bool(captureGroup, "enabled") ?? true,
      excludeAgents: strArray(captureGroup, "excludeAgents") ?? [],
      l0l1RetentionDays: retentionDays ?? 0,
      allowAggressiveCleanup,
    },
    extraction: {
      enabled: bool(extractionGroup, "enabled") ?? true,
      enableDedup: bool(extractionGroup, "enableDedup") ?? true,
      maxMemoriesPerSession: num(extractionGroup, "maxMemoriesPerSession") ?? 20,
      model: optStr(extractionGroup, "model"),
      promptMode: normalizePromptMode(str(extractionGroup, "promptMode"), globalPromptMode),
    },
    persona: {
      triggerEveryN: num(personaGroup, "triggerEveryN") ?? 50,
      maxScenes: num(personaGroup, "maxScenes") ?? 15,
      backupCount: num(personaGroup, "backupCount") ?? 3,
      sceneBackupCount: num(personaGroup, "sceneBackupCount") ?? 10,
      model: optStr(personaGroup, "model"),
      promptMode: normalizePromptMode(str(personaGroup, "promptMode"), globalPromptMode),
    },
    pipeline: {
      everyNConversations: num(pipelineGroup, "everyNConversations") ?? 5,
      enableWarmup: bool(pipelineGroup, "enableWarmup") ?? true,
      l1IdleTimeoutSeconds: num(pipelineGroup, "l1IdleTimeoutSeconds") ?? 600,
      l2DelayAfterL1Seconds: num(pipelineGroup, "l2DelayAfterL1Seconds") ?? 10,
      l2MinIntervalSeconds: num(pipelineGroup, "l2MinIntervalSeconds") ?? 900,
      l2MaxIntervalSeconds: num(pipelineGroup, "l2MaxIntervalSeconds") ?? 3600,
      sessionActiveWindowHours: num(pipelineGroup, "sessionActiveWindowHours") ?? 24,
    },
    recall: {
      enabled: bool(recallGroup, "enabled") ?? true,
      maxResults: num(recallGroup, "maxResults") ?? 5,
      maxCharsPerMemory: num(recallGroup, "maxCharsPerMemory") ?? 0,
      maxTotalRecallChars: num(recallGroup, "maxTotalRecallChars") ?? 0,
      scoreThreshold: num(recallGroup, "scoreThreshold") ?? 0.3,
      strategy: validateStrategy(str(recallGroup, "strategy")) ?? "hybrid",
      timeoutMs: num(recallGroup, "timeoutMs") ?? 5000,
    },
    embedding: {
      enabled: embeddingEnabled,
      provider: embeddingProvider,
      baseUrl: embeddingBaseUrl,
      apiKey: embeddingApiKey,
      model: str(embeddingGroup, "model") ?? defaultModel,
      dimensions: num(embeddingGroup, "dimensions") ?? defaultDimensions,
      sendDimensions: bool(embeddingGroup, "sendDimensions") ?? true,
      conflictRecallTopK: num(embeddingGroup, "conflictRecallTopK") ?? 5,
      proxyUrl: embeddingProxyUrl,
      maxInputChars: num(embeddingGroup, "maxInputChars") ?? 5000,
      timeoutMs: num(embeddingGroup, "timeoutMs") ?? 10_000,
      recallTimeoutMs: num(embeddingGroup, "recallTimeoutMs") ?? undefined,
      captureTimeoutMs: num(embeddingGroup, "captureTimeoutMs") ?? undefined,
      modelCacheDir: optStr(embeddingGroup, "modelCacheDir"),
      configError: embeddingConfigError,
    },
    storeBackend,
    tcvdb: {
      url: str(tcvdbGroup, "url") ?? "",
      username: str(tcvdbGroup, "username") ?? "root",
      apiKey: str(tcvdbGroup, "apiKey") ?? "",
      database: str(tcvdbGroup, "database") ?? "",
      alias: str(tcvdbGroup, "alias") ?? "",
      embeddingEnabled: bool(tcvdbGroup, "embeddingEnabled") ?? false,
      embeddingModel: str(tcvdbGroup, "embeddingModel") ?? "bge-large-zh",
      timeout: num(tcvdbGroup, "timeout") ?? 10000,
      caPemPath: str(tcvdbGroup, "caPemPath") || undefined,
    },
    bm25: {
      enabled: bool(bm25Group, "enabled") ?? true,
      language: (str(bm25Group, "language") === "en" ? "en" : "zh") as "zh" | "en",
    },
    memoryCleanup,
    report: {
      enabled: bool(obj(c, "report"), "enabled") ?? false,
      type: str(obj(c, "report"), "type") ?? "local",
    },
    llm: (() => {
      const llmGroup = obj(c, "llm");
      const rawProvider = str(llmGroup, "provider");
      const provider: "openai" | "proxy" =
        rawProvider === "proxy" ? "proxy" : "openai";
      const proxyGroup = obj(llmGroup, "proxy");
      return {
        enabled: bool(llmGroup, "enabled") ?? false,
        baseUrl: str(llmGroup, "baseUrl") ?? "https://api.openai.com/v1",
        apiKey: str(llmGroup, "apiKey") ?? "",
        model: str(llmGroup, "model") ?? "gpt-4o",
        maxTokens: num(llmGroup, "maxTokens") ?? 4096,
        timeoutMs: num(llmGroup, "timeoutMs") ?? 120_000,
        provider,
        proxy: {
          // 默认 true：走 proxy 时用 memory 系统用户 key 作为 Authorization。
          useMemorySystemUserKey: bool(proxyGroup, "useMemorySystemUserKey") ?? true,
        },
      };
    })(),
    offload,
    // Skill: passthrough — let the host wiring call resolveSkillConfig() with
    // ambient probes (TCVDB / COS / embedding / LLMRunner). We don't apply
    // defaults here so the resolver remains the single source of truth.
    skill: (c.skill && typeof c.skill === "object" && !Array.isArray(c.skill))
      ? (c.skill as import("./core/skill/types.js").SkillConfigInput)
      : undefined,
  };
}

// ============================
// Helper functions
// ============================

/** Get sub-object by key, or empty object if missing. */
function obj(c: Record<string, unknown>, key: string): Record<string, unknown> {
  const v = c[key];
  return v && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : {};
}

function str(src: Record<string, unknown>, key: string): string | undefined {
  const v = src[key];
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

function normalizePromptMode(value: string | undefined, fallback: MemoryPromptMode = "chat"): MemoryPromptMode {
  if (value === "code" || value === "chat") return value;
  return fallback;
}

function optStr(src: Record<string, unknown>, key: string): string | undefined {
  const v = src[key];
  return typeof v === "string" ? v : undefined;
}

function num(src: Record<string, unknown>, key: string): number | undefined {
  const v = src[key];
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function bool(src: Record<string, unknown>, key: string): boolean | undefined {
  const v = src[key];
  return typeof v === "boolean" ? v : undefined;
}

function strArray(src: Record<string, unknown>, key: string): string[] | undefined {
  const v = src[key];
  if (!Array.isArray(v)) return undefined;
  return v.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

const VALID_STRATEGIES: RecallConfig["strategy"][] = ["embedding", "keyword", "hybrid"];

/**
 * Validate recall strategy against whitelist.
 * Returns the strategy if valid, undefined otherwise (caller falls back to default).
 */
function validateStrategy(value: string | undefined): RecallConfig["strategy"] | undefined {
  if (!value) return undefined;
  return VALID_STRATEGIES.includes(value as RecallConfig["strategy"])
    ? (value as RecallConfig["strategy"])
    : undefined;
}

/**
 * Normalize a cleanup time string.
 *
 * The input must follow "HH:MM" or "H:MM" format (24-hour clock).
 * If the time is valid, it returns the normalized format "HH:MM"
 * with leading zeros added when necessary.
 * If the format is invalid or the time is out of range
 * (hour: 0–23, minute: 0–59), it returns undefined.
 *
 * Examples:
 * normalizeCleanTime("3:05")  -> "03:05"
 * normalizeCleanTime("03:05") -> "03:05"
 * normalizeCleanTime("23:59") -> "23:59"
 *
 * normalizeCleanTime("24:00") -> undefined   // hour out of range
 * normalizeCleanTime("12:60") -> undefined   // minute out of range
 * normalizeCleanTime("3:5")   -> undefined   // minute must have two digits
 * normalizeCleanTime("abc")   -> undefined   // invalid format
 */
function normalizeCleanTime(input: string | undefined): string | undefined {
  if (!input) return undefined;
  const trimmed = input.trim();
  const m = /^(\d{1,2}):(\d{2})$/.exec(trimmed);
  if (!m) return undefined;

  const hh = Number(m[1]);
  const mm = Number(m[2]);
  if (!Number.isInteger(hh) || !Number.isInteger(mm)) return undefined;
  if (hh < 0 || hh > 23 || mm < 0 || mm > 59) return undefined;

  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

/**
 * Normalize offload retention days.
 *
 * - `<= 0` → 0 (disabled)
 * - `(0, 3)` → 0 (invalid, force disabled)
 * - `>= 3` → as-is
 */
function normalizeOffloadRetentionDays(value: number): number {
  if (value <= 0) return 0;
  if (value < 3) return 0;
  return value;
}
