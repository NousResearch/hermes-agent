/**
 * Type definitions for the Skill module — v2 redesign (2026-06-17).
 *
 * Host-neutral. No imports from openclaw / hermes / automation projects.
 * 设计文档：docs/design/2026-06-17-skill-redesign-v2.md
 */

// ============================
// Configuration types (input from user / openclaw.json)
// ============================

/**
 * User-facing skill configuration. All fields optional; defaults applied
 * by `resolveSkillConfig`.
 */
export interface SkillConfigInput {
  enabled?: boolean;

  /** Override for skill metadata + vector store backend. Falls back to outer storeBackend, then 'sqlite'. */
  storeBackend?: "sqlite" | "tcvdb";

  /** Override for skill content (SKILL.md + resources) backend. Falls back to env probe → 'local'. */
  contentBackend?: "local" | "cos";

  routing?: {
    mode?: "bm25" | "embedding" | "hybrid";
    hybridAlpha?: number;
    searchTopK?: number;
    charBudgetPercent?: number;
    fastPathMinNameLength?: number;
  };

  extraction?: {
    enabled?: boolean;
    toolCallThreshold?: number;
    model?: string;
    maxIterations?: number;
    /**
     * 单一"归档尺寸"旋钮（字节）。默认 40960 (40KB)。派生 7 个内部字段:
     *   • Handler 的 bytesThreshold / requestCompressThresholdBytes = archiveBytes
     *   • Oversize 兜底的 chunkMaxBytes = 2 × archiveBytes
     *   • Oversize 兜底的 headKeepBytes / tailKeepBytes = archiveBytes
     *   • Extractor transcript 截断的 headChars / tailChars = archiveBytes
     * 语义: 归档 payload 目标大小 = archiveBytes，上限 = 2 × archiveBytes。
     */
    archiveBytes?: number;
    /** Skill review 单次 LLM 调用输出 token 上限。不填 → 继承顶层 llm.maxTokens。 */
    maxTokens?: number;
  };

  /**
   * 单条大 tool 消息头尾压缩规则。只影响 tool_call / tool_result 单条 content
   * 超阈值时的头尾切分；user/assistant/system 永不压缩。
   */
  compress?: {
    /** 单条 tool 消息 content 超过多少字节才压缩。默认 2048 (2KB)。 */
    toolContentThresholdBytes?: number;
    /** 压缩后保留的头字节。默认 1024 (1KB)。 */
    headBytes?: number;
    /** 压缩后保留的尾字节。默认 1024 (1KB)。 */
    tailBytes?: number;
  };

  resources?: {
    maxResourceSizeBytes?: number;
    downloadDir?: string;
    allowExecutable?: boolean;
  };

  /** 旧版本 TTL 天数。默认 0（关闭）。设 7 = 非 head 版本创建 7 天后过期。 */
  versionTtlDays?: number;
}

// ============================
// Resolved configuration (after defaults + downgrade decisions)
// ============================

export interface ResolvedSkillConfig {
  enabled: true; // when this object exists, skill is enabled
  storeBackend: "sqlite" | "tcvdb";
  contentBackend: "local" | "cos";

  routing: {
    mode: "bm25" | "embedding" | "hybrid";
    hybridAlpha: number;
    searchTopK: number;
    charBudgetPercent: number;
    fastPathMinNameLength: number;
  };

  extraction: {
    enabled: boolean;
    toolCallThreshold: number;
    model?: string;
    maxIterations: number;
    /** 归档尺寸旋钮 (字节)。用户可见配置源；下面 7 个字段由它派生。 */
    archiveBytes: number;
    /** Skill review 单次 LLM 调用输出 token 上限；不填 → 由 runner 继承 llm.maxTokens。 */
    maxTokens?: number;
    // ↓↓↓ 以下 7 个由 archiveBytes 派生，用户不直接配 ↓↓↓
    /** Handler: buffer 累计字节 ≥ 触发归档。= archiveBytes。 */
    bytesThreshold: number;
    /** Handler: 单次 add 请求 ≥ 强制走压缩路径。= archiveBytes。 */
    requestCompressThresholdBytes: number;
    /** Oversize 兜底: 归档 payload > 触发切分。= 2 × archiveBytes。 */
    chunkMaxBytes: number;
    /** Oversize 兜底: 切完保留的头字节。= archiveBytes。 */
    headKeepBytes: number;
    /** Oversize 兜底: 切完保留的尾字节。= archiveBytes。 */
    tailKeepBytes: number;
    /** Extractor transcript 截断: 保留头字符。= archiveBytes (字节数近似当字符数)。 */
    headChars: number;
    /** Extractor transcript 截断: 保留尾字符。= archiveBytes。 */
    tailChars: number;
  };

  /** 单条大 tool 消息头尾压缩，参数与 CompressOptions 对齐。 */
  compress: {
    toolContentThresholdBytes: number;
    headBytes: number;
    tailBytes: number;
  };

  resources: {
    maxResourceSizeBytes: number;
    downloadDir: string;
    allowExecutable: boolean;
  };

  /** 旧版本 TTL 秒数。0 = 关闭。 */
  versionTtlSeconds: number;

  /** Records of automatic downgrades made during resolution. */
  degradations: SkillDegradation[];
}

export interface SkillDegradation {
  field: string;
  from: string;
  to: string;
  reason: string;
  level: "info" | "warn";
}

// ============================
// Probe inputs to resolveSkillConfig
// ============================

/**
 * Information about ambient capabilities that resolveSkillConfig uses
 * to make downgrade decisions. Keep this minimal and explicit; no
 * implicit env/process reads inside resolveSkillConfig itself.
 */
export interface SkillEnvProbe {
  /** Outer storeBackend from MemoryTdaiConfig. */
  outerStoreBackend?: "sqlite" | "tcvdb";

  /** TCVDB credentials present (url + apiKey + database all set). */
  hasTcvdbCredentials: boolean;

  /** COS credentials present (secretId + secretKey + bucket all set). */
  hasCosCredentials: boolean;

  /** Embedding subsystem usable (enabled + provider valid + dimensions > 0). */
  embeddingAvailable: boolean;

  /**
   * Whether the host provides an LLMRunnerFactory. When false and
   * extraction.enabled=true, we mark extraction as degraded (it stays
   * "enabled" but will return [] at runtime).
   */
  llmRunnerAvailable: boolean;
}

// ════════════════════════════════════════════════════════════════════════
//  v2 数据面契约（2026-06-17 redesign）
// ════════════════════════════════════════════════════════════════════════

/**
 * 业务身份四元组。全部可选。
 * team_id 和 agent_id 要么都传要么都不传（由 gateway schema 层 cross-field 校验保证）。
 */
export interface IdFields {
  user_id?: string;
  team_id?: string;
  agent_id?: string;
  task_id?: string;
}

/** skill 状态。与 interface.yaml 对齐：active 或 archived。 */
export type SkillStatus = "active" | "archived";

/** manifest_json 列里的单个资源元信息。字节不在此类型中。 */
export interface SkillManifestEntry {
  /** 相对 `files/` 的路径，UNIX 风格，禁 `..` / 绝对路径。例 "scripts/run.sh" */
  path: string;
  size_bytes: number;
  mime_type: string;
  is_executable: boolean;
}

// ============================
// Skill dedup / propose types (M13 — two-step confirmation)
// ============================

export interface SkillSimilarityResult {
  name: string;
  description: string;
  similarity: number;
}

export interface SkillProposeResult {
  propose_id: string;
  proposed: {
    name: string;
    description: string;
  };
  similar_skills: SkillSimilarityResult[];
}

/**
 * skill 主表的一行。每行 = (skill_id, version) 一个不可变快照。
 *
 * - 字段对应 `skills` 表（见 skill-store-ddl.ts SKILLS_DDL）
 * - `manifest` 是 `manifest_json` 列反序列化后的结构化形式
 * - `is_head` 是 boolean（DB 中是 0/1 INTEGER）
 */
export interface Skill {
  row_id: string;
  skill_id: string;
  version: number;
  is_head: boolean;

  user_id: string;
  owner_agent_id: string;
  team_id: string;
  task_id: string;

  name: string;
  description: string;
  content: string;
  content_hash: string;
  manifest: SkillManifestEntry[];
  storage_dir: string;

  status: SkillStatus;
  metadata_json: string;
  created_at_ms: number;
  updated_at_ms: number;
}

/** `appendVersion` 的入参。store 内部基于 head 推导 version+1。 */
export interface AppendVersionInput {
  /** 写入身份。不传则写 "default"。 */
  user_id?: string;
  team_id?: string;
  agent_id?: string;
  task_id?: string;

  /** 业务主键 — 首次创建由调用方生成；同 skill 的后续版本同 skill_id。 */
  skill_id: string;

  name: string;
  description: string;
  content: string;
  content_hash: string;
  manifest: SkillManifestEntry[];
  storage_dir: string;

  /** 仅 create 时由调用方指定为 owner_agent_id；后续版本由 store 校验后从 head 继承。 */
  owner_agent_id?: string;

  metadata_json?: string;
}

/** `listSkills` 的查询参数。仅返回 head + (status 满足) 的行。四个 ID 全部可选，传了就过滤。 */
export interface ListSkillsOptions {
  team_id?: string;
  owner_agent_id?: string;
  user_id?: string;
  task_id?: string;
  name_prefix?: string;
  status?: SkillStatus[];
  limit?: number;
  offset?: number;
}

/** `searchSkills` 的查询参数。仅命中 head + active 行。四个 ID 全部可选，传了就过滤。 */
export interface SearchSkillsOptions {
  team_id?: string;
  query: string;
  queryEmbedding?: Float32Array;
  topK?: number;
  /**
   * 检索模式（设计 §3.5.7）。
   *   - 'bm25'      : 仅 FTS5 BM25
   *   - 'embedding' : 仅 vec0 KNN（需 queryEmbedding）
   *   - 'hybrid'    : BM25 + KNN RRF 融合（需 queryEmbedding）
   * 不传 / 未配置 embedding 时降级为 bm25。
   */
  mode?: "bm25" | "embedding" | "hybrid";
  /** 可选：按 owner agent 过滤搜索结果。 */
  agent_id?: string;
  /** 可选：按 task 过滤搜索结果。 */
  task_id?: string;
  /** 可选：按 user 过滤搜索结果。 */
  user_id?: string;
}

/** 抽取接口的结构化对话消息。对齐 interface.yaml §SkillImportMessage。 */
export interface ExtractMessage {
  role: "user" | "assistant" | "tool_call" | "tool_result";
  content: string;
  timestamp?: string;
}

/**
 * Skill Review Agent 用的 LLM runner 形状。boot 端构造后注入 SkillExtractor。
 * 与 src/adapters/standalone/llm-runner.ts 的 StandaloneLLMRunner.run 兼容。
 */
export interface ExtractorLLMRunner {
  run(params: {
    prompt: string;
    systemPrompt?: string;
    /** Tool dict (Vercel AI SDK shape). 当 enableTools=true 时驱动 tool-call 循环。 */
    tools?: Record<string, unknown>;
    enableTools?: boolean;
    maxIterations?: number;
    taskId: string;
    timeoutMs?: number;
    /** Worker 在锁失效时通过 abort 信号取消 LLM 调用。 */
    signal?: AbortSignal;
  }): Promise<string>;
}
