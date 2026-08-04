/**
 * TDAI Core — Host-neutral type definitions and abstract interfaces.
 *
 * These types define the boundary between TDAI Core (memory algorithms)
 * and the host environment (OpenClaw, Hermes, standalone Gateway, etc.).
 *
 * Design principles:
 * 1. TDAI Core depends ONLY on these interfaces — never on a specific host.
 * 2. Each host provides its own implementation of HostAdapter + LLMRunnerFactory.
 * 3. RuntimeContext is the single source of truth for session/user identity.
 */

// ============================
// Logger (unified across all layers)
// ============================

/**
 * Canonical logger interface used across all TDAI modules.
 *
 * Named variants (StoreLogger, PluginLogger, etc.) are type aliases
 * of this interface, kept for backward compatibility.
 */
export interface Logger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

// ============================
// RuntimeContext
// ============================

/**
 * Unified runtime context — provides identity, scoping, and path information.
 *
 * In OpenClaw: populated from `pluginConfig`, `sessionKey`, `resolveStateDir()`.
 * In Hermes:   populated from `MemoryProvider.initialize()` kwargs.
 * In Gateway:  populated from HTTP request parameters.
 */
export interface RuntimeContext {
  /** User identifier (e.g. "default_user" for CLI, platform user ID for gateway). */
  userId: string;
  /** Session identifier (unique per conversation session). */
  sessionId: string;
  /** Session key (stable across reconnects, used for L0/L1 grouping). */
  sessionKey: string;
  /** Host platform identifier. */
  platform: "openclaw" | "hermes" | "cli" | "gateway" | string;
  /** Agent identity / profile name (optional). */
  agentIdentity?: string;
  /** Agent execution context — primary agent, subagent, cron job, or flush task. */
  agentContext?: "primary" | "subagent" | "cron" | "flush";
  /** Workspace directory (for tool sandbox, if applicable). */
  workspaceDir: string;
  /** Plugin/provider data directory (L0, records, scene_blocks, etc.). */
  dataDir: string;
}

// ============================
// LLMRunner
// ============================

/** Parameters for a single LLM execution. */
export interface LLMRunParams {
  /** User-facing prompt (or combined prompt if no systemPrompt). */
  prompt: string;
  /** Optional system prompt. When provided, `prompt` is used as the user message. */
  systemPrompt?: string;
  /** Unique task identifier for logging and metrics. */
  taskId: string;
  /** Execution timeout in milliseconds (default: 120_000). */
  timeoutMs?: number;
  /** Max output tokens (optional — defaults to model catalog value). */
  maxTokens?: number;
  /**
   * Caller-provided tool dict (Vercel AI SDK shape). When set, REPLACES
   * the runner's default sandbox tools for this single call. Used by
   * SkillExtractor to inject skill_list / skill_view / skill_manage
   * without polluting the runner's permanent configuration.
   */
  tools?: Record<string, unknown>;
  /**
   * Per-call override of the runner's `enableTools` setting. When true
   * and `tools` is provided, the runner drives an automatic tool-call
   * loop via the AI SDK. When false, tools are ignored.
   */
  enableTools?: boolean;
  /** Cap iterations of the per-call tool-call loop. Default 20 in standalone runner. */
  maxIterations?: number;
  /**
   * Working directory for tool-enabled runs.
   * When `enableTools` is true, the LLM's file tools resolve paths relative to this dir.
   * When omitted, a clean empty workspace is used.
   */
  workspaceDir?: string;
  /**
   * Storage adapter for service mode (COS). When provided, LLM file tools
   * (read/write/edit) operate via StorageAdapter instead of local filesystem.
   * `storagePrefix` defines the sandbox key prefix (e.g. "scene_blocks/").
   */
  storage?: import("./storage/adapter.js").StorageAdapter;
  /** Key prefix for storage-backed tools (sandbox boundary). Default: "" */
  storagePrefix?: string;
  /** Plugin instance ID for metric reporting (optional). */
  instanceId?: string;
  /**
   * H-11 Step 2: external abort signal (in addition to the internal timeout).
   * When this aborts (e.g. pipeline-worker lost its lock), the LLM call
   * tears down immediately to save tokens and avoid late writes.
   */
  abortSignal?: AbortSignal;
  /**
   * Langfuse trace name — 决定 Langfuse UI 上 trace 的 "Name" 字段。
   * 未传时保持向后兼容（Langfuse 会显示 Unnamed trace）。
   * 建议传业务语义值，如 "skill.extract" / "memory.l1-extract"。
   */
  traceName?: string;
  /**
   * Langfuse trace 标签，用于 UI 筛选。空数组等价于未传。
   * 建议格式：["<domain>", "team:<id>", "agent:<id>"]。
   */
  tags?: string[];
  /**
   * Langfuse 顶级 sessionId 字段。空字符串等价于未传。
   */
  sessionId?: string;
  /**
   * Langfuse 顶级 userId 字段。空字符串等价于未传。
   */
  userId?: string;
}

/**
 * TraceContext —— 记忆/技能抽取链路给 langfuse 上报用的身份四元组。
 * 由 caller 向下透传到 llmRunner.run，最终填充 LLMRunParams 的 userId/sessionId/tags。
 * 好处：langfuse UI 上按 user_id / session_id 独立列过滤，不用把身份塞进 name。
 */
export interface TraceContext {
  teamId?: string;
  userId?: string;
  agentId?: string;
  sessionId?: string;
}

/**
 * 把 TraceContext 展平进 LLMRunParams 的 langfuse 三字段。
 *   - traceName: 传业务锚点（如 "memory.l1-extract"）
 *   - userId:    langfuse 顶级 userId 列
 *   - sessionId: langfuse 顶级 sessionId 列
 *   - tags:      ["team:<id>", "agent:<id>"] 便于侧栏筛选
 * 三者结合后 langfuse UI 上一眼能定位某个 (user, agent, session) 的 trace 组。
 */
export function buildTraceParams(traceName: string, ctx?: TraceContext): {
  traceName: string;
  userId?: string;
  sessionId?: string;
  tags?: string[];
} {
  const out: { traceName: string; userId?: string; sessionId?: string; tags?: string[] } = { traceName };
  if (ctx?.userId) out.userId = ctx.userId;
  if (ctx?.sessionId) out.sessionId = ctx.sessionId;
  const tags: string[] = [];
  if (ctx?.teamId) tags.push(`team:${ctx.teamId}`);
  if (ctx?.agentId) tags.push(`agent:${ctx.agentId}`);
  if (tags.length > 0) out.tags = tags;
  return out;
}

/**
 * Unified LLM execution interface.
 *
 * Replaces direct usage of `CleanContextRunner` throughout TDAI Core.
 *
 * Implementations:
 * - `OpenClawLLMRunner`: wraps `CleanContextRunner` / `runEmbeddedPiAgent` (OpenClaw host)
 * - `StandaloneLLMRunner`: direct OpenAI-compatible HTTP calls (Gateway / Hermes host)
 */
export interface LLMRunner {
  /**
   * Execute a prompt and return the LLM's text output.
   *
   * Behavior depends on the factory configuration:
   * - `enableTools: false` → pure text output (used by L1 extraction, L1 dedup)
   * - `enableTools: true`  → LLM may call file tools (used by L2 scene, L3 persona)
   *
   * @returns The LLM's text response. Empty string if the LLM produces no output.
   * @throws On timeout, network errors, or unrecoverable LLM failures.
   */
  run(params: LLMRunParams): Promise<string>;
}

// ============================
// LLMRunnerFactory
// ============================

/** Options for creating an LLMRunner instance. */
export interface LLMRunnerCreateOptions {
  /**
   * Full "provider/model" string (e.g. "openai/gpt-4o").
   * Takes precedence over host default model.
   */
  modelRef?: string;
  /**
   * Whether the runner should allow tool calls (read_file, write_to_file, etc.).
   * Default: false (text-only output).
   */
  enableTools?: boolean;
}

/**
 * Factory for creating LLMRunner instances.
 *
 * Each host provides its own factory implementation that knows how to
 * configure runners with the correct model, API keys, and tool sandbox.
 */
export interface LLMRunnerFactory {
  createRunner(opts?: LLMRunnerCreateOptions): LLMRunner;
}

// ============================
// HostAdapter
// ============================

/**
 * Host adapter — translates host-specific events, context, and capabilities
 * into TDAI Core's unified interface.
 *
 * Each host environment provides exactly one HostAdapter implementation:
 * - OpenClaw:    `OpenClawHostAdapter` — wraps `OpenClawPluginApi`
 * - Hermes/GW:   `StandaloneHostAdapter` — wraps Gateway HTTP request context
 *
 * HostAdapter answers these questions for TDAI Core:
 * - "Who is the current user/session?" → `getRuntimeContext()`
 * - "How do I call an LLM?"           → `getLLMRunnerFactory()`
 * - "Where do I log?"                 → `getLogger()`
 */
export interface HostAdapter {
  /** Identifies the host type for conditional behavior (should be rare). */
  readonly hostType: "openclaw" | "hermes" | "standalone";

  /** Get the unified runtime context for the current session. */
  getRuntimeContext(): RuntimeContext;

  /** Get the logger instance provided by the host. */
  getLogger(): Logger;

  /** Get the LLM runner factory configured for this host. */
  getLLMRunnerFactory(): LLMRunnerFactory;
}

// ============================
// CompletedTurn — represents a finished conversation turn
// ============================

/** A completed conversation turn, ready for capture/storage. */
export interface CompletedTurn {
  /** The user's original message text. */
  userText: string;
  /** The assistant's response text. */
  assistantText: string;
  /** All messages in the turn (may include tool call results, etc.). */
  messages: unknown[];
  /** Session key for this turn. */
  sessionKey: string;
  /** Session ID within the session key (optional, for sub-session grouping). */
  sessionId?: string;
  /** Epoch ms when this turn started. */
  startedAt?: number;
  /**
   * Number of messages in the session at before_prompt_build time.
   * Used by l0-recorder to locate the exact user message that was
   * polluted by prependContext injection.
   */
  originalUserMessageCount?: number;
}

// ============================
// Core service result types
// ============================

/** Result from a recall (prefetch) operation. */
export interface RecallResult {
  /** L1 relevant memories — prepended to user prompt text (dynamic, per-turn). */
  prependContext?: string;
  /** Stable recall context appended to system prompt (persona, scene nav, tools guide). */
  appendSystemContext?: string;
  /** Recalled L1 memories with scores (for metrics). */
  recalledL1Memories?: Array<{ content: string; score: number; type: string }>;
  /** L3 Persona content (for metrics). */
  recalledL3Persona?: string | null;
  /** Search strategy used. */
  recallStrategy?: string;
  /**
   * H-15: structured failure signal. When recall fails (config error / dependency timeout /
   * storage error / etc), this is populated with a RecallError; success leaves it undefined.
   * Gateway handlers should surface this in the response envelope (e.g. v2 envelope.code).
   */
  error?: import("./hooks/recall-errors.js").RecallError;
  /** Partial success: some steps succeeded, others failed. */
  partial?: boolean;
}

/** Result from a capture (sync_turn) operation. */
export interface CaptureResult {
  /** Number of L0 messages recorded. */
  l0RecordedCount: number;
  /** Whether the pipeline scheduler was notified. */
  schedulerNotified: boolean;
  /** Number of L0 vectors written. */
  l0VectorsWritten: number;
  /** Filtered messages that were captured. */
  filteredMessages: Array<{
    role: string;
    content: string;
    timestamp: number;
  }>;
}

/** Search parameters for L1 memory search. */
export interface MemorySearchParams {
  query: string;
  limit?: number;
  type?: string;
  scene?: string;
}

/** Search parameters for L0 conversation search. */
export interface ConversationSearchParams {
  query: string;
  limit?: number;
  sessionKey?: string;
}
