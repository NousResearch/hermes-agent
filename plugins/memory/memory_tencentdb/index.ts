/**
 * memory-tdai v3: Four-layer memory system plugin for OpenClaw.
 *
 * Provides:
 * - L0: Automatic conversation recording (local JSONL)
 * - L1: Structured memory extraction (LLM + dedup)
 * - L2: Scene block management (LLM scene extraction)
 * - L3: Persona generation (LLM persona synthesis)
 *
 * All processing is local, zero external API dependencies.
 *
 * v3.1: Refactored to use TdaiCore + OpenClawHostAdapter.
 * index.ts is now a thin shell that:
 * - Registers tools and hooks with OpenClaw
 * - Translates OpenClaw events into TdaiCore calls
 * - Manages prompt caching and metric reporting
 *
 * Core memory logic lives in src/core/tdai-core.ts (host-neutral).
 */

import path from "node:path";
import fs from "node:fs";
import { createRequire } from "node:module";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import registerClientOpenClawPlugin from "./openclaw-plugin/index.js";
import { parseConfig } from "./src/config.js";
import type { MemoryTdaiConfig } from "./src/config.js";
import { executeReadCos, READ_COS_TOOL_SCHEMA, READ_COS_TOOL_NAME, READ_COS_TOOL_DESCRIPTION } from "./src/core/tools/read-cos.js";
import { LocalStorageBackend } from "./src/core/storage/local-backend.js";
import { StaticCredentialProvider } from "./src/core/storage/credential-provider.js";
import type { IStorageBackend } from "./src/core/storage/types.js";
import { registerOffload } from "./src/offload/index.js";
import { registerOffloadClient } from "./src/offload-client/index.js";
import {
  setPreferredEmbeddedAgentRuntime,
  prewarmEmbeddedAgent,
} from "./src/utils/clean-context-runner.js";
import { SessionFilter } from "./src/utils/session-filter.js";
import { LocalMemoryCleaner } from "./src/utils/memory-cleaner.js";
import { readCosToolEnvConfig } from "./src/utils/env-config.js";
import { registerMemoryTdaiCli } from "./src/cli/index.js";
import { initDataDirectories, resetStores } from "./src/utils/pipeline-factory.js";
import { getOrCreateInstanceId, initReporter, report, resetReporter } from "./src/core/report/reporter.js";
import { ensureL2L3Local } from "./src/core/profile/profile-sync.js";

// Core abstractions (host-neutral)
import { OpenClawHostAdapter } from "./src/adapters/openclaw/host-adapter.js";
import { TdaiCore } from "./src/core/tdai-core.js";
import {
  ensurePluginHookPolicy,
  decideHookPolicy,
} from "./src/utils/ensure-hook-policy.js";
import { resolveOpenClawStateDir } from "./src/utils/openclaw-state-dir.js";

const TAG = "[memory-tdai]";

type OpenClawAdapterMode = "local" | "client";

function resolveOpenClawAdapterMode(rawPluginConfig: Record<string, unknown> | undefined): OpenClawAdapterMode {
  const rawMode = typeof rawPluginConfig?.mode === "string"
    ? rawPluginConfig.mode.trim().toLowerCase()
    : "";

  if (rawMode === "client" || rawMode === "gateway" || rawMode === "remote") {
    return "client";
  }

  // Default to local/function mode: OpenClaw calls this plugin in-process and
  // memory processing uses the host LLM runner (no standalone Gateway needed).
  return "local";
}

/**
 * Epoch ms when the plugin was registered (cold-start timestamp).
 * Used as a fallback cursor in performAutoCapture when no checkpoint
 * exists yet — prevents the first agent_end from dumping the entire
 * session history into L0.
 */
let pluginStartTimestamp = 0;

/**
 * Cache original user prompts and message counts across hooks.
 * - text: clean user prompt before prependContext injection
 * - ts: cache creation time (for TTL sweep)
 * - messageCount: session message count at before_prompt_build time,
 *   used as fallback slice offset if timestamp cursor is unreliable
 */
const pendingOriginalPrompts = new Map<string, { text: string; ts: number; messageCount: number }>();
const PROMPT_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
const PROMPT_CACHE_MAX_SIZE = 10_000; // Hard limit to prevent unbounded growth in high-concurrency scenarios

/**
 * Cache recall results (L1 memories + L3 Persona) from before_prompt_build
 * for retrieval at agent_end, enabling the agent_turn metric event.
 *
 * Keyed by sessionKey — same correlation pattern as pendingOriginalPrompts.
 */
const pendingRecallCache = new Map<string, {
  l1Memories: Array<{ content: string; score: number; type: string }>;
  l3Persona: string | null;
  strategy: string;
  durationMs: number;
  ts: number;
}>();

/**
 * Cache recall completion timestamps per session.
 * Used in agent_end to estimate LLM reasoning time:
 *   llmEstimatedMs ≈ agent_end_start - recall_end_ts
 * Entries are cleaned up in agent_end after use; stale entries swept alongside prompt cache.
 */
const pendingRecallEndTimestamps = new Map<string, number>();

// 进程级单例，避免同一进程重复启动清理器导致并发清理竞态
let sharedMemoryCleaner: LocalMemoryCleaner | undefined;

/**
 * Sweep both pendingOriginalPrompts and pendingRecallCache for stale entries.
 * Unified from the original sweepStalePromptCache() to cover both Maps
 * with identical TTL + hard-cap logic.
 */
function sweepStaleCaches(): void {
  const now = Date.now();
  // Clean pendingOriginalPrompts
  for (const [key, entry] of pendingOriginalPrompts) {
    if (now - entry.ts > PROMPT_CACHE_TTL_MS) {
      pendingOriginalPrompts.delete(key);
      pendingRecallEndTimestamps.delete(key);
    }
  }
  // Clean pendingRecallCache
  for (const [key, entry] of pendingRecallCache) {
    if (now - entry.ts > PROMPT_CACHE_TTL_MS) {
      pendingRecallCache.delete(key);
    }
  }
  // Hard limit: evict oldest entries if either Map exceeds cap
  if (pendingOriginalPrompts.size > PROMPT_CACHE_MAX_SIZE) {
    const entries = [...pendingOriginalPrompts.entries()].sort((a, b) => a[1].ts - b[1].ts);
    const toEvict = entries.slice(0, entries.length - PROMPT_CACHE_MAX_SIZE);
    for (const [key] of toEvict) {
      pendingOriginalPrompts.delete(key);
      pendingRecallEndTimestamps.delete(key);
    }
  }
  if (pendingRecallCache.size > PROMPT_CACHE_MAX_SIZE) {
    const entries = [...pendingRecallCache.entries()].sort((a, b) => a[1].ts - b[1].ts);
    const toEvict = entries.slice(0, entries.length - PROMPT_CACHE_MAX_SIZE);
    for (const [key] of toEvict) {
      pendingRecallCache.delete(key);
    }
  }
}

export default function register(api: OpenClawPluginApi) {
  // ─── CLI metadata mode: register CLI commands only, skip all runtime init ───
  // In this mode, runtime is `{} as PluginRuntime` (empty object).
  // OpenClaw calls this to discover CLI subcommands without starting the full plugin.
  if (api.registrationMode === "cli-metadata") {
    api.registerCli(
      ({ program, config, logger: cliLogger }) => {
        const memoryTdai = program
          .command("memory-tdai")
          .description("memory-tdai plugin commands (seed, query, stats)");

        registerMemoryTdaiCli(memoryTdai, {
          config,
          pluginConfig: api.pluginConfig,
          stateDir: resolveOpenClawStateDir((api.runtime as any)?.state),
          logger: cliLogger,
        });
      },
      { commands: ["memory-tdai"] },
    );
    return;
  }

  // ─── Full / discovery mode: complete runtime initialization ───
  const rawPluginConfigForMode = api.pluginConfig as Record<string, unknown> | undefined;
  const adapterMode = resolveOpenClawAdapterMode(rawPluginConfigForMode);
  if (adapterMode === "client") {
    api.logger.info?.(`${TAG} mode=client: delegating to memory-tencentdb-client adapter`);
    return registerClientOpenClawPlugin(api as any);
  }

  pluginStartTimestamp = Date.now();
  setPreferredEmbeddedAgentRuntime(api.runtime.agent);
  // Reset reporter singleton so config changes take effect on hot-reload.
  resetReporter();
  const _require = createRequire(import.meta.url);
  const pluginVersion = (() => { try { return (_require("./package.json") as { version?: string }).version ?? "unknown"; } catch { return "unknown"; } })();
  api.logger.debug?.(
    `${TAG} Registering plugin ... ` +
    `startTimestamp=${pluginStartTimestamp} (${new Date(pluginStartTimestamp).toISOString()})`,
  );

  let cfg: MemoryTdaiConfig;
  try {
    // OpenClaw calls register() N times (plugin scan → gateway start →
    // per-channel bootstrap → config reload). Each call receives the full
    // pluginConfig from openclaw.json, so we parse it directly every time.
    const rawPluginConfig = api.pluginConfig as Record<string, unknown> | undefined;
    const rawKeys = rawPluginConfig ? Object.keys(rawPluginConfig) : [];
    api.logger.debug?.(
      `${TAG} pluginConfig received (${rawKeys.length} keys)`,
    );

    cfg = parseConfig(rawPluginConfig);
    api.logger.debug?.(
      `${TAG} Config parsed: ` +
      `capture=${cfg.capture.enabled}, ` +
      `recall=${cfg.recall.enabled}(maxResults=${cfg.recall.maxResults}), ` +
      `extraction=${cfg.extraction.enabled}(dedup=${cfg.extraction.enableDedup}, maxMem=${cfg.extraction.maxMemoriesPerSession}), ` +
      `pipeline=(everyN=${cfg.pipeline.everyNConversations}, warmup=${cfg.pipeline.enableWarmup}, l1Idle=${cfg.pipeline.l1IdleTimeoutSeconds}s, l2DelayAfterL1=${cfg.pipeline.l2DelayAfterL1Seconds}s, l2Min=${cfg.pipeline.l2MinIntervalSeconds}s, l2Max=${cfg.pipeline.l2MaxIntervalSeconds}s, activeWindow=${cfg.pipeline.sessionActiveWindowHours}h), ` +
      `persona(triggerEvery=${cfg.persona.triggerEveryN}, backupCount=${cfg.persona.backupCount}, sceneBackupCount=${cfg.persona.sceneBackupCount}), ` +
      `memoryCleanup(enabled=${cfg.memoryCleanup.enabled}, retentionDays=${cfg.memoryCleanup.retentionDays ?? "(disabled)"}, cleanTime=${cfg.memoryCleanup.cleanTime}), ` +
      `offload(enabled=${cfg.offload.enabled}, mode=${cfg.offload.mode}, ${cfg.offload.mode === "client" ? `serverUrl=${cfg.offload.serverUrl ?? "(none)"}` : `backendUrl=${cfg.offload.backendUrl ?? "(none)"}, mildRatio=${cfg.offload.mildOffloadRatio}, aggressiveRatio=${cfg.offload.aggressiveCompressRatio}`}, retentionDays=${cfg.offload.offloadRetentionDays})`,
    );
  } catch (err) {
    api.logger.error(`${TAG} Config parsing failed: ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }

  // ============================
  // Hook policy auto-patch (v2026.4.24+ compat)
  // ============================
  // `allowConversationAccess` hook policy was introduced in v2026.4.23;
  // the zod schema fix landed in v2026.4.24. Older hosts don't understand
  // the field and don't need it patched in.
  //
  // Note: `api.runtime.version` is only exposed on v2026.4.15+. On older
  // hosts it is `undefined`; we MUST treat that as "does not need the
  // patch" (old hosts have no gate), otherwise we would silently mutate
  // the user's openclaw.json on every gateway start.
  {
    // Gate: only apply the auto-patch when host version >= 2026.4.24.
    // decideHookPolicy() parses the leading x.y.z prefix numerically
    // (ignoring `-beta.N`, `-N`, etc.) and returns apply=false for any
    // version we cannot parse — which is the safe default on old hosts
    // that don't expose `api.runtime.version`. See ensure-hook-policy.ts
    // for the full policy + co-located unit tests.
    const rawVersion = (api.runtime as any)?.version;
    const decision = decideHookPolicy(rawVersion);
    const parsedStr = decision.parsedXYZ ? decision.parsedXYZ.join(".") : "<unparsable>";
    const minStr = decision.minXYZ.join(".");

    if (!decision.apply) {
      api.logger.debug?.(
        `${TAG} Hook policy auto-patch skipped: ` +
        `original=${JSON.stringify(rawVersion)}, parsed=${parsedStr}, min=${minStr}`,
      );
    } else {
      api.logger.debug?.(
        `${TAG} Hook policy auto-patch applying: ` +
        `original=${JSON.stringify(rawVersion)}, parsed=${parsedStr} >= min=${minStr}`,
      );
      try {
        ensurePluginHookPolicy({
          rootConfig: api.config,
          runtimeConfig: (api.runtime as any)?.config,
          logger: api.logger,
        });
      } catch (err) {
        api.logger.warn(`${TAG} Hook policy check failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  // If remote embedding config is incomplete, log a prominent error so the user knows
  if (cfg.embedding.configError) {
    api.logger.error(`${TAG} [EMBEDDING CONFIG ERROR] ${cfg.embedding.configError}`);
  }

  // Resolve plugin data directory via runtime API (avoid importing internal paths directly)
  const openclawStateDir = resolveOpenClawStateDir((api.runtime as any)?.state);
  const pluginDataDir = path.join(openclawStateDir, "memory-tdai");
  initDataDirectories(pluginDataDir);
  api.logger.debug?.(`${TAG} Data dir: ${pluginDataDir} (all subdirectories initialized)`);

  // ============================
  // Create OpenClawHostAdapter + TdaiCore
  // ============================
  const hostAdapter = new OpenClawHostAdapter({
    api,
    pluginDataDir,
    openclawConfig: api.config,
  });

  const sessionFilter = new SessionFilter(cfg.capture.excludeAgents);
  if (cfg.capture.excludeAgents.length > 0) {
    api.logger.debug?.(`${TAG} Agent exclude patterns: ${cfg.capture.excludeAgents.join(", ")}`);
  }

  const core = new TdaiCore({
    hostAdapter,
    config: cfg,
    sessionFilter,
  });

  // Initialize TdaiCore (async — store init, pipeline wiring)
  const coreReady = core.initialize().then(() => {
    // Keep cleaner's SQLite handle updated after store init
    memoryCleaner?.setVectorStore(core.getVectorStore());

    // Pull L2/L3 profiles if remote store supports it
    const vs = core.getVectorStore();
    if (vs?.pullProfiles) {
      ensureL2L3Local(pluginDataDir, vs, api.logger).catch((err) => {
        api.logger.warn(`${TAG} Startup L2/L3 pull failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
      });
    }
  }).catch((err) => {
    api.logger.error(`${TAG} Core init failed: ${err instanceof Error ? err.message : String(err)}`);
  });

  // Kick off instanceId resolution immediately after data dir is ready.
  let instanceId: string | undefined;
  getOrCreateInstanceId(pluginDataDir).then((id) => {
    instanceId = id;
    core.setInstanceId(id);
    initReporter({ enabled: cfg.report.enabled, type: cfg.report.type, logger: api.logger, instanceId: id, pluginVersion });
  }).catch((err) => {
    api.logger.warn(`${TAG} Failed to initialize instanceId for metrics: ${err instanceof Error ? err.message : String(err)}`);
  });

  // Daily local JSONL cleaner (L0/L1), enabled only when retentionDays is configured.
  let memoryCleaner: LocalMemoryCleaner | undefined;
  if (cfg.memoryCleanup.enabled && cfg.memoryCleanup.retentionDays != null) {
    if (!sharedMemoryCleaner) {
      sharedMemoryCleaner = new LocalMemoryCleaner({
        baseDir: pluginDataDir,
        retentionDays: cfg.memoryCleanup.retentionDays,
        cleanTime: cfg.memoryCleanup.cleanTime,
        logger: api.logger,
      });
      sharedMemoryCleaner.start();
      api.logger.debug?.(`${TAG} Memory cleaner started (singleton)`);
    } else {
      api.logger.debug?.(`${TAG} Memory cleaner already started in this process, reusing existing instance`);
    }
    memoryCleaner = sharedMemoryCleaner;
  } else {
    api.logger.debug?.(`${TAG} Memory cleaner disabled (retentionDays not configured)`);
  }

  const resolveSessionKey = (sessionKey?: string): string | undefined => {
    if (sessionKey) return sessionKey;
    api.logger.warn(`${TAG} sessionKey is empty, skipping capture/recall to avoid unstable fallback key`);
    return undefined;
  };

  /**
   * Whether embedding warmup has been triggered.
   * Deferred until first real conversation to avoid model downloads during CLI commands.
   */
  let embeddingWarmupTriggered = false;
  const ensureEmbeddingWarmup = (): void => {
    const svc = core.getEmbeddingService();
    if (!svc) return;
    if (!embeddingWarmupTriggered) {
      embeddingWarmupTriggered = true;
      api.logger.debug?.(`${TAG} Triggering lazy embedding warmup on first conversation`);
      svc.startWarmup();
      return;
    }
    if (!svc.isReady()) {
      api.logger.debug?.(`${TAG} Embedding not ready, re-triggering warmup (retry)`);
      svc.startWarmup();
    }
  };

  // ============================
  // Tool registration — delegate to TdaiCore
  // ============================

  // tdai_memory_search — Agent-callable L1 memory search tool
  // TODO: implement hard per-turn call limit via before_tool_call hook + execute early-return (方案 D)
  if (cfg.recall.enabled || cfg.capture.enabled) {
  api.registerTool(
    {
      name: "tdai_memory_search",
      label: "Memory Search",
      description:
        "Search through the user's long-term memories. Use this when you need to recall specific information about the user's preferences, past events, instructions, or context from previous conversations. Returns relevant memory records ranked by relevance. " +
        "Limit: tdai_memory_search and tdai_conversation_search share a combined limit of 3 calls per turn. Stop searching after 3 total attempts.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query describing what you want to recall about the user",
          },
          limit: {
            type: "number",
            description: "Maximum number of results to return (default: 5, max: 20)",
          },
          type: {
            type: "string",
            enum: ["persona", "episodic", "instruction"],
            description: "Optional filter by memory type: persona (identity/preferences), episodic (events/activities), instruction (user rules/commands)",
          },
          scene: {
            type: "string",
            description: "Optional filter by scene name",
          },
        },
        required: ["query"],
      },
      async execute(_toolCallId: string, params: Record<string, unknown>) {
        const startMs = Date.now();
        const query = String(params.query ?? "");
        const limit = Math.min(Math.max(Number(params.limit) || 5, 1), 20);
        const typeFilter = typeof params.type === "string" ? params.type : undefined;
        const sceneFilter = typeof params.scene === "string" ? params.scene : undefined;

        api.logger.debug?.(
          `${TAG} [tool] tdai_memory_search called: ` +
          `query="${query.length > 80 ? query.slice(0, 80) + "…" : query}", ` +
          `limit=${limit}, type=${typeFilter ?? "(all)"}, scene=${sceneFilter ?? "(all)"}`,
        );

        try {
          const result = await core.searchMemories({ query, limit, type: typeFilter, scene: sceneFilter });

          const elapsedMs = Date.now() - startMs;
          api.logger.debug?.(
            `${TAG} [tool] tdai_memory_search completed (${elapsedMs}ms): ` +
            `total=${result.total}, strategy=${result.strategy}, ` +
            `responseLength=${result.text.length} chars`,
          );
          report("tool_call", {
            tool: "tdai_memory_search",
            query, limit, typeFilter, sceneFilter,
            resultCount: result.total,
            strategy: result.strategy,
            durationMs: elapsedMs,
            success: true,
          });
          return {
            content: [{ type: "text" as const, text: result.text }],
            details: { count: result.total, strategy: result.strategy },
          };
        } catch (err) {
          const elapsedMs = Date.now() - startMs;
          const errMsg = err instanceof Error ? err.message : String(err);
          api.logger.error(`${TAG} [tool] tdai_memory_search failed (${elapsedMs}ms): ${errMsg}`);
          report("tool_call", {
            tool: "tdai_memory_search",
            query, limit, typeFilter, sceneFilter,
            durationMs: elapsedMs,
            success: false,
            error: errMsg,
          });
          return {
            content: [{ type: "text" as const, text: `Memory search failed: ${errMsg}` }],
            details: { error: errMsg },
          };
        }
      },
    },
    { name: "tdai_memory_search" },
  );

  // tdai_conversation_search — Agent-callable L0 conversation search tool
  // TODO: implement hard per-turn call limit via before_tool_call hook + execute early-return (方案 D)
  api.registerTool(
    {
      name: "tdai_conversation_search",
      label: "Conversation Search",
      description:
        "Search through past conversation history (raw dialogue records). " +
        "Use this when tdai_memory_search (structured memories) doesn't have the information you need, " +
        "or when you want to find specific past conversations, dialogue context, or exact words " +
        "the user said before. Returns relevant individual messages ranked by relevance. " +
        "Limit: tdai_memory_search and tdai_conversation_search share a combined limit of 3 calls per turn. Stop searching after 3 total attempts.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query describing what conversation content you want to find",
          },
          limit: {
            type: "number",
            description: "Maximum number of messages to return (default: 5, max: 20)",
          },
          session_key: {
            type: "string",
            description: "Optional: filter results to a specific session",
          },
        },
        required: ["query"],
      },
      async execute(_toolCallId: string, params: Record<string, unknown>) {
        const startMs = Date.now();
        const query = String(params.query ?? "");
        const limit = Math.min(Math.max(Number(params.limit) || 5, 1), 20);
        const sessionKeyFilter = typeof params.session_key === "string" ? params.session_key : undefined;

        api.logger.debug?.(
          `${TAG} [tool] tdai_conversation_search called: ` +
          `query="${query.length > 80 ? query.slice(0, 80) + "…" : query}", ` +
          `limit=${limit}, session_key=${sessionKeyFilter ?? "(all)"}`,
        );

        try {
          const result = await core.searchConversations({ query, limit, sessionKey: sessionKeyFilter });

          const elapsedMs = Date.now() - startMs;
          api.logger.debug?.(
            `${TAG} [tool] tdai_conversation_search completed (${elapsedMs}ms): ` +
            `total=${result.total}, responseLength=${result.text.length} chars`,
          );
          report("tool_call", {
            tool: "tdai_conversation_search",
            query, limit, sessionKeyFilter,
            resultCount: result.total,
            durationMs: elapsedMs,
            success: true,
          });
          return {
            content: [{ type: "text" as const, text: result.text }],
            details: { count: result.total },
          };
        } catch (err) {
          const elapsedMs = Date.now() - startMs;
          const errMsg = err instanceof Error ? err.message : String(err);
          api.logger.error(`${TAG} [tool] tdai_conversation_search failed (${elapsedMs}ms): ${errMsg}`);
          report("tool_call", {
            tool: "tdai_conversation_search",
            query, limit, sessionKeyFilter,
            durationMs: elapsedMs,
            success: false,
            error: errMsg,
          });
          return {
            content: [{ type: "text" as const, text: `Conversation search failed: ${errMsg}` }],
            details: { error: errMsg },
          };
        }
      },
    },
    { name: "tdai_conversation_search" },
  );

  // tdai_read_cos — Agent-callable tool for reading files from COS/local storage
  {
    // Read COS config from {pluginDataDir}/cos.env
    // Format: KEY=VALUE per line (standard .env)
    const cosEnvPath = path.join(pluginDataDir, "cos.env");
    const cosEnv: Record<string, string> = {};

    if (fs.existsSync(cosEnvPath)) {
      const lines = fs.readFileSync(cosEnvPath, "utf-8").split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const eqIdx = trimmed.indexOf("=");
        if (eqIdx > 0) {
          cosEnv[trimmed.slice(0, eqIdx).trim()] = trimmed.slice(eqIdx + 1).trim();
        }
      }
    }

    // Priority: env vars > cos.env file
    const {
      cosSecretId,
      cosSecretKey,
      cosBucket,
      cosRegion,
      cosPrefix,
      cosDomain,
    } = readCosToolEnvConfig(cosEnv);

    // Lazily resolve the storage backend on first call. Optional COS support
    // is loaded dynamically; when unavailable, the tool falls back to local
    // storage. The resolution is cached so subsequent calls are zero-cost.
    let storagePromise: Promise<IStorageBackend> | null = null;
    const resolveReadCosStorage = (): Promise<IStorageBackend> => {
      if (storagePromise) return storagePromise;
      storagePromise = (async (): Promise<IStorageBackend> => {
        if (cosSecretId && cosSecretKey && cosBucket) {
          try {
            const cosBackendModulePath = ["./src/integrations/cos", "cos-backend.js"].join("/");
            const { CosStorageBackend } = await import(cosBackendModulePath) as {
              CosStorageBackend: new (options: any) => IStorageBackend;
            };
            const backend = new CosStorageBackend({
              credentialProvider: new StaticCredentialProvider({
                secretId: cosSecretId,
                secretKey: cosSecretKey,
                bucket: cosBucket,
                region: cosRegion,
                prefix: cosPrefix,
              }),
              logger: api.logger,
              cosEndpointDomain: cosDomain,
            });
            api.logger.info(`${TAG} read_cos: using COS backend (bucket=${cosBucket}, prefix=${cosPrefix})`);
            return backend;
          } catch (err) {
            const storageDataDir = path.join(pluginDataDir, "cos-storage");
            const backend = new LocalStorageBackend({ rootDir: storageDataDir, logger: api.logger });
            api.logger.warn(
              `${TAG} read_cos: COS integration not available (${err instanceof Error ? err.message : String(err)}), ` +
              `falling back to local backend (${storageDataDir}).`,
            );
            return backend;
          }
        }
        const storageDataDir = path.join(pluginDataDir, "cos-storage");
        const backend = new LocalStorageBackend({ rootDir: storageDataDir, logger: api.logger });
        api.logger.info(`${TAG} read_cos: no COS credentials, using local backend (${storageDataDir}). Place cos.env in ${pluginDataDir} to enable COS.`);
        return backend;
      })();
      return storagePromise;
    };

    api.registerTool(
      {
        name: READ_COS_TOOL_NAME,
        label: "Read COS File",
        description: READ_COS_TOOL_DESCRIPTION,
        parameters: READ_COS_TOOL_SCHEMA,
        async execute(_toolCallId: string, params: Record<string, unknown>) {
          const startMs = Date.now();
          const filePath = String(params.path ?? "");

          api.logger.debug?.(`${TAG} [tool] tdai_read_cos called: path="${filePath}"`);

          const readCosStorage = await resolveReadCosStorage();
          const result = await executeReadCos(
            { path: filePath },
            readCosStorage,
            api.logger,
          );

          const elapsedMs = Date.now() - startMs;
          api.logger.debug?.(
            `${TAG} [tool] tdai_read_cos completed (${elapsedMs}ms): ` +
            `success=${result.success}, size=${result.size}`,
          );
          report("tool_call", {
            tool: "tdai_read_cos",
            path: filePath,
            durationMs: elapsedMs,
            success: result.success,
            size: result.size,
            error: result.error,
          });

          if (result.success) {
            return {
              content: [{ type: "text" as const, text: result.content }],
              details: { path: result.path, size: result.size },
            };
          } else {
            return {
              content: [{ type: "text" as const, text: `Read failed: ${result.error}` }],
              details: { path: result.path, error: result.error },
            };
          }
        },
      },
      { name: READ_COS_TOOL_NAME },
    );

    api.logger.info(`${TAG} Registered tool: tdai_read_cos (storage backend resolved lazily on first call)`);
  }
  } else {
    api.logger.debug?.(`${TAG} Memory tools (tdai_memory_search, tdai_conversation_search) not registered — memory features disabled`);
  }

  // ============================
  // Lifecycle hooks — delegate to TdaiCore
  // ============================

  // Before prompt build: auto-recall relevant memories
  if (cfg.recall.enabled) {
    api.logger.debug?.(`${TAG} Registering before_prompt_build hook (auto-recall)`);
    api.on("before_prompt_build", async (event, ctx) => {
      const startMs = Date.now();
      api.logger.debug?.(`${TAG} [before_prompt_build] Hook triggered`);

      const sessionKey = ctx.sessionKey;

      if (sessionFilter.shouldSkipCtx(ctx)) {
        api.logger.debug?.(`${TAG} [before_prompt_build] Skipping filtered session`);
        return;
      }

      ensureEmbeddingWarmup();

      // Cache original user prompt for agent_end
      const rawPrompt = event.prompt;
      const messages = Array.isArray(event.messages) ? event.messages : undefined;
      if (sessionKey && rawPrompt) {
        const messageCount = messages?.length ?? 0;
        pendingOriginalPrompts.set(sessionKey, { text: rawPrompt, ts: Date.now(), messageCount });
        api.logger.debug?.(`${TAG} [before_prompt_build] Cached original prompt (${rawPrompt.length} chars, msgCount=${messageCount})`);
      }
      sweepStaleCaches();

      const userText = rawPrompt;
      api.logger.debug?.(`${TAG} [before_prompt_build] userText length: ${userText?.length}`);
      if (!userText) {
        api.logger.debug?.(`${TAG} [before_prompt_build] No user text found, skipping recall`);
        return;
      }

      const resolvedSessionKey = resolveSessionKey(sessionKey);
      if (!resolvedSessionKey) {
        return;
      }

      try {
        await coreReady;
        const recallStartMs = Date.now();
        const result = await core.handleBeforeRecall(userText, resolvedSessionKey);
        const elapsedMs = Date.now() - startMs;
        const recallDurationMs = Date.now() - recallStartMs;

        // Cache recall results for agent_turn metric (retrieved at agent_end)
        if (sessionKey && result) {
          pendingRecallCache.set(sessionKey, {
            l1Memories: result.recalledL1Memories ?? [],
            l3Persona: result.recalledL3Persona ?? null,
            strategy: result.recallStrategy ?? "unknown",
            durationMs: recallDurationMs,
            ts: Date.now(),
          });
        }

        // Record recall completion timestamp for LLM timing estimation in agent_end
        if (resolvedSessionKey) {
          pendingRecallEndTimestamps.set(resolvedSessionKey, Date.now());
        }

        if (result?.appendSystemContext || result?.prependContext) {
          const appendLen = result.appendSystemContext?.length ?? 0;
          const prependLen = result.prependContext?.length ?? 0;
          api.logger.info(
            `${TAG} [before_prompt_build] Recall complete (${elapsedMs}ms), ` +
            `appendSystemContext=${appendLen} chars, prependContext=${prependLen} chars`,
          );
        } else {
          api.logger.info(`${TAG} [before_prompt_build] Recall complete (${elapsedMs}ms), no context to inject`);
        }
        return result;
      } catch (err) {
        const elapsedMs = Date.now() - startMs;
        api.logger.error(`${TAG} [before_prompt_build] Auto-recall failed after ${elapsedMs}ms: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
        if (instanceId) {
          report("error_degradation", {
            module: "auto-recall",
            action: "performAutoRecall",
            errorType: "exception",
            errorMessage: err instanceof Error ? err.message : String(err),
            degradedTo: "no_recall",
            impact: "non-blocking",
          });
        }
      }
    });
  }

  // Strip <relevant-memories> from user messages before they are persisted to
  // the session JSONL.  The current-turn LLM already saw the full prompt
  // (effectivePrompt lives in memory), but we don't want recall artifacts
  // polluting the historical transcript for future replays.
  api.logger.debug?.(`${TAG} Registering before_message_write hook (strip <relevant-memories>)`);
  api.on("before_message_write", (event) => {
    const msg = event.message as { role?: string; content?: unknown };
    const contentType = typeof msg.content === "string" ? "string" : Array.isArray(msg.content) ? "parts" : typeof msg.content;
    api.logger.debug?.(`${TAG} [before_message_write] role=${msg.role}, contentType=${contentType}`);

    if (msg.role !== "user") return;

    // UserMessage.content: string | (TextContent | ImageContent)[]
    const STRIP_RE = /<relevant-memories>[\s\S]*?<\/relevant-memories>\s*/g;

    if (typeof msg.content === "string") {
      if (!msg.content.includes("<relevant-memories>")) return;
      const cleaned = msg.content.replace(STRIP_RE, "").trim();
      if (cleaned === msg.content) return;
      api.logger.debug?.(`${TAG} [before_message_write] Stripped: ${msg.content.length} → ${cleaned.length} chars`);
      return { message: { ...event.message, content: cleaned } as typeof event.message };
    }

    if (Array.isArray(msg.content)) {
      let totalStripped = 0;
      const cleanedParts = (msg.content as Array<Record<string, unknown>>).map((part) => {
        if (part.type !== "text" || typeof part.text !== "string") return part;
        if (!(part.text as string).includes("<relevant-memories>")) return part;
        const cleaned = (part.text as string).replace(STRIP_RE, "").trim();
        totalStripped += (part.text as string).length - cleaned.length;
        return { ...part, text: cleaned };
      });
      if (totalStripped === 0) return;
      api.logger.debug?.(`${TAG} [before_message_write] Stripped from parts: removed ${totalStripped} chars`);
      return { message: { ...event.message, content: cleanedParts } as unknown as typeof event.message };
    }
  });

  // After agent end: auto-capture + L0 record + L1/L2/L3 schedule
  if (cfg.capture.enabled) {
    api.logger.debug?.(`${TAG} Registering agent_end hook (auto-capture)`);
    api.on("agent_end", async (event, ctx) => {
      const startMs = Date.now();
      api.logger.debug?.(`${TAG} [agent_end] Hook triggered`);

      const e = event as Record<string, unknown>;
      if (!e.success) {
        api.logger.info(`${TAG} [agent_end] Agent did not succeed, skipping capture`);
        return;
      }

      const sessionKey = ctx.sessionKey;
      const sessionId = ctx.sessionId;

      if (sessionFilter.shouldSkipCtx(ctx)) {
        api.logger.debug?.(`${TAG} [agent_end] Skipping filtered session`);
        return;
      }

      const messages = (e.messages as unknown[]) ?? [];
      const resolvedSessionKey = resolveSessionKey(sessionKey);
      if (!resolvedSessionKey) {
        return;
      }

      // Estimate LLM reasoning time: recallEnd → agentEnd start
      const recallEndTs = pendingRecallEndTimestamps.get(resolvedSessionKey);
      if (recallEndTs) {
        const llmEstimatedMs = startMs - recallEndTs;
        api.logger.info(
          `${TAG} ⏱ Turn timing: recallEnd→agentEnd=${llmEstimatedMs}ms ` +
          `(≈ LLM reasoning + prompt build + tool calls)`,
        );
        pendingRecallEndTimestamps.delete(resolvedSessionKey);
      }

      // Retrieve cached original prompt
      const cachedPrompt = sessionKey ? pendingOriginalPrompts.get(sessionKey) : undefined;
      const originalUserText = cachedPrompt?.text;

      try {
        await coreReady;

        // Pre-warm the embedded agent on first conversation
        if (!core.isSchedulerStarted()) {
          prewarmEmbeddedAgent(api.logger, api.runtime.agent);
        }

        const captureResult = await core.handleTurnCommitted({
          userText: originalUserText ?? "",
          assistantText: "",
          messages,
          sessionKey: resolvedSessionKey,
          sessionId: sessionId || undefined,
          startedAt: pluginStartTimestamp,
          originalUserMessageCount: cachedPrompt?.messageCount,
        });
        const captureMs = Date.now() - startMs;
        api.logger.info(
          `${TAG} [agent_end] Auto-capture complete (${captureMs}ms), ` +
          `l0Recorded=${captureResult.l0RecordedCount}, ` +
          `schedulerNotified=${captureResult.schedulerNotified}`,
        );

        // ── agent_turn metric ──
        const cachedRecall = sessionKey ? pendingRecallCache.get(sessionKey) : undefined;
        if (sessionKey) pendingRecallCache.delete(sessionKey);

        if (instanceId) {
          report("agent_turn", {
            sessionKey: resolvedSessionKey,
            userPrompt: originalUserText ?? null,
            recalledL1Memories: cachedRecall?.l1Memories ?? [],
            recalledL1Count: cachedRecall?.l1Memories?.length ?? 0,
            recalledL3Persona: cachedRecall?.l3Persona ?? null,
            recallStrategy: cachedRecall?.strategy ?? null,
            recallDurationMs: cachedRecall?.durationMs ?? 0,
            l0CapturedMessages: captureResult.filteredMessages.map((m) => ({
              role: m.role,
              content: m.content,
              ts: m.timestamp,
            })),
            l0CapturedCount: captureResult.l0RecordedCount,
            l0VectorsWritten: captureResult.l0VectorsWritten,
            captureDurationMs: captureMs,
            totalDurationMs: Date.now() - startMs,
          });
        }
      } catch (err) {
        const elapsedMs = Date.now() - startMs;
        api.logger.error(`${TAG} [agent_end] Auto-capture failed after ${elapsedMs}ms: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
        if (instanceId) {
          report("error_degradation", {
            module: "auto-capture",
            action: "performAutoCapture",
            errorType: "exception",
            errorMessage: err instanceof Error ? err.message : String(err),
            degradedTo: "no_capture",
            impact: "non-blocking",
          });
        }
      }
    });

    // gateway_stop: ordered shutdown via TdaiCore.destroy()
    api.on("gateway_stop", async () => {
      const GATEWAY_STOP_TIMEOUT_MS = 3_000;
      const hookStartMs = Date.now();

      await coreReady.catch(() => {});

      const doCleanup = async (): Promise<void> => {
        // 1. Stop memory cleaner first
        if (memoryCleaner) {
          try {
            memoryCleaner.destroy();
            if (sharedMemoryCleaner === memoryCleaner) {
              sharedMemoryCleaner = undefined;
            }
          } catch (error) {
            api.logger.error(`${TAG} [gateway_stop] memoryCleaner error: ${error instanceof Error ? error.message : String(error)}`);
          }
        }

        // 2. Destroy TdaiCore (scheduler flush + VectorStore close + EmbeddingService close)
        await core.destroy();
      };

      // Race cleanup against a hard timeout
      let timeoutId: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          doCleanup(),
          new Promise<never>((_, reject) => {
            timeoutId = setTimeout(
              () => reject(new Error("timeout")),
              GATEWAY_STOP_TIMEOUT_MS,
            );
          }),
        ]);
      } catch (err) {
        api.logger.warn(
          `${TAG} [gateway_stop] Aborted (${Date.now() - hookStartMs}ms): ${err instanceof Error ? err.message : String(err)}. ` +
          `Pending work will recover on next startup.`,
        );
      } finally {
        if (timeoutId !== undefined) clearTimeout(timeoutId);
      }

      resetStores();
      api.logger.info(`${TAG} [gateway_stop] Cleanup finished, all resources released (${Date.now() - hookStartMs}ms)`);
    });
  } else {
    api.logger.debug?.(`${TAG} Auto-capture disabled`);
  }

  // memoryCleaner gateway_stop for capture-enabled-but-extraction-disabled case
  if (memoryCleaner && !cfg.extraction.enabled) {
    api.on("gateway_stop", async () => {
      const startMs = Date.now();
      try {
        memoryCleaner?.destroy();
        if (sharedMemoryCleaner === memoryCleaner) {
          sharedMemoryCleaner = undefined;
        }
        api.logger.info(`${TAG} [gateway_stop] Memory cleaner destroyed (${Date.now() - startMs}ms)`);
      } catch (error) {
        api.logger.error(`${TAG} [gateway_stop] Error during memory cleaner destruction (${Date.now() - startMs}ms): ${error instanceof Error ? error.message : String(error)}`);
      }
    });
  }

  // ============================
  // Context Offload (conditional)
  // ============================
  if (cfg.offload.enabled) {
    api.logger.debug?.(`${TAG} Offload enabled, registering offload module...`);
    try {
      if (cfg.offload.mode === "client") {
        // New: stateless client mode — all compression delegated to server
        registerOffloadClient(api as any, {
          enabled: cfg.offload.enabled,
          serverUrl: cfg.offload.serverUrl ?? "",
          apiKey: cfg.offload.apiKey ?? "",
          serviceId: cfg.offload.serviceId ?? "",
          agentName: cfg.offload.agentName ?? "default",
          compactionRatio: cfg.offload.compactionRatio ?? 0.5,
          ingestTimeoutMs: cfg.offload.ingestTimeoutMs ?? 5000,
          compactionTimeoutMs: cfg.offload.compactionTimeoutMs ?? 30000,
        });
        api.logger.debug?.(`${TAG} Offload client module registered (mode=client)`);
      } else {
        // Legacy: full local L3 compression
        registerOffload(api, cfg.offload);
        api.logger.debug?.(`${TAG} Offload module registered successfully (mode=${cfg.offload.mode})`);
      }
    } catch (err) {
      api.logger.error(`${TAG} Offload module registration failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  } else {
    api.logger.debug?.(`${TAG} Offload disabled (offload.enabled=false)`);
  }

  // ============================
  // CLI registration
  // ============================

  api.registerCli(
    ({ program, config, logger: cliLogger }) => {
      const memoryTdai = program
        .command("memory-tdai")
        .description("memory-tdai plugin commands (seed, query, stats)");

      registerMemoryTdaiCli(memoryTdai, {
        config,
        pluginConfig: api.pluginConfig,
        stateDir: openclawStateDir,
        logger: cliLogger,
      });
    },
    { commands: ["memory-tdai"] },
  );

  api.logger.debug?.(
    `${TAG} Plugin registration complete (v3.1 — TdaiCore). ` +
    `startTimestamp=${pluginStartTimestamp} (${new Date(pluginStartTimestamp).toISOString()})`,
  );
}
