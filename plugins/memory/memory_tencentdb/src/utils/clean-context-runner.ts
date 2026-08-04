/**
 * CleanContextRunner: executes LLM calls in a fully isolated context
 * using runEmbeddedPiAgent (same mechanism as the llm-task extension).
 *
 * Guarantees:
 * 1. Blank conversation history (temporary session file)
 * 2. Independent system prompt (only the task prompt)
 * 3. No tool calls when enableTools=false (disableTools:true — no tool definitions sent to API)
 * 4. No contamination from the main agent's context
 */

import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { getEnv } from "./env.js";
import { report } from "../core/report/reporter.js";
import type { Logger } from "../core/types.js";

/**
 * Resolve a preferred temporary directory for memory-tdai operations.
 *
 * Previously imported from `openclaw/plugin-sdk` as `resolvePreferredOpenClawTmpDir`,
 * but that export was removed in openclaw 2026.2.23+. This local implementation
 * provides equivalent behavior:
 *   1. Try `/tmp/openclaw` (if writable)
 *   2. Fall back to `os.tmpdir()/openclaw-<uid>`
 */
function resolveOpenClawTmpDir(): string {
  const POSIX_DIR = "/tmp/openclaw";
  try {
    if (fsSync.existsSync(POSIX_DIR)) {
      fsSync.accessSync(POSIX_DIR, fsSync.constants.W_OK | fsSync.constants.X_OK);
      return POSIX_DIR;
    }
    // Try to create it
    fsSync.mkdirSync(POSIX_DIR, { recursive: true, mode: 0o700 });
    return POSIX_DIR;
  } catch {
    // Fall back to os.tmpdir()
    const uid = typeof process.getuid === "function" ? process.getuid() : undefined;
    const suffix = uid === undefined ? "openclaw" : `openclaw-${uid}`;
    const fallback = path.join(os.tmpdir(), suffix);
    fsSync.mkdirSync(fallback, { recursive: true });
    return fallback;
  }
}

const TAG = "[memory-tdai] [runner]";

type RunnerLogger = Logger;

// Dynamic import type — runEmbeddedPiAgent is an internal API
// Prefer the public plugin runtime signature so host-injected runtimes stay assignable.
type RunEmbeddedPiAgentFn = OpenClawPluginApi["runtime"]["agent"]["runEmbeddedPiAgent"];

export interface EmbeddedAgentRuntimeLike {
  runEmbeddedPiAgent?: RunEmbeddedPiAgentFn;
}

let _preferredAgentRuntime: EmbeddedAgentRuntimeLike | undefined;

export function setPreferredEmbeddedAgentRuntime(
  agentRuntime: EmbeddedAgentRuntimeLike | undefined,
): void {
  _preferredAgentRuntime = agentRuntime;
}

function resolveInjectedRunEmbeddedPiAgent(
  agentRuntime?: EmbeddedAgentRuntimeLike,
): RunEmbeddedPiAgentFn | undefined {
  const candidate =
    agentRuntime?.runEmbeddedPiAgent ?? _preferredAgentRuntime?.runEmbeddedPiAgent;
  return typeof candidate === "function" ? candidate : undefined;
}

async function resolveRunEmbeddedPiAgent(
  agentRuntime: EmbeddedAgentRuntimeLike | undefined,
  logger?: RunnerLogger,
): Promise<RunEmbeddedPiAgentFn> {
  const injected = resolveInjectedRunEmbeddedPiAgent(agentRuntime);
  if (injected) {
    logger?.debug?.(
      `${TAG} resolveRunEmbeddedPiAgent: using injected runtime.agent.runEmbeddedPiAgent`,
    );
    logger?.debug?.(`${TAG} [l1-debug] RESOLVE source=injected`);
    return injected;
  }
  logger?.debug?.(`${TAG} [l1-debug] RESOLVE source=dist-fallback`);
  return loadRunEmbeddedPiAgent(logger);
}

// ── Core import (mirrors voice-call/core-bridge.ts — dist/ only, no jiti) ──

let _rootCache: string | null = null;

function findPackageRoot(startDir: string, name: string): string | null {
  let dir = startDir;
  for (;;) {
    const pkgPath = path.join(dir, "package.json");
    try {
      if (fsSync.existsSync(pkgPath)) {
        const raw = fsSync.readFileSync(pkgPath, "utf8");
        const pkg = JSON.parse(raw) as { name?: string };
        if (pkg.name === name) return dir;
      }
    } catch { /* ignore */ }
    const parent = path.dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

function resolveOpenClawRoot(): string {
  if (_rootCache) return _rootCache;
  const override = getEnv("OPENCLAW_ROOT")?.trim();
  if (override) { _rootCache = override; return override; }

  const candidates = new Set<string>();
  if (process.argv[1]) candidates.add(path.dirname(process.argv[1]));
  candidates.add(process.cwd());
  try { candidates.add(path.dirname(fileURLToPath(import.meta.url))); } catch { /* ignore */ }

  for (const start of candidates) {
    const found = findPackageRoot(start, "openclaw");
    if (found) { _rootCache = found; return found; }
  }
  throw new Error("Unable to resolve OpenClaw root. Set OPENCLAW_ROOT or run `pnpm build`.");
}

let _loadPromise: Promise<RunEmbeddedPiAgentFn> | null = null;

function loadRunEmbeddedPiAgent(logger?: RunnerLogger): Promise<RunEmbeddedPiAgentFn> {
  if (_loadPromise) return _loadPromise;

  _loadPromise = (async () => {
    const t0 = Date.now();
    const distPath = path.join(resolveOpenClawRoot(), "dist", "extensionAPI.js");
    if (!fsSync.existsSync(distPath)) {
      throw new Error(`Missing core module at ${distPath}. Run \`pnpm build\` or install the official package.`);
    }
    const mod = await import(pathToFileURL(distPath).href);
    if (typeof mod.runEmbeddedPiAgent !== "function") {
      throw new Error("runEmbeddedPiAgent not exported from dist/extensionAPI.js");
    }
    logger?.info(`${TAG} loadRunEmbeddedPiAgent: dist/ import OK (${Date.now() - t0}ms)`);
    return mod.runEmbeddedPiAgent as RunEmbeddedPiAgentFn;
  })();

  _loadPromise.catch(() => { _loadPromise = null; });
  return _loadPromise;
}

/**
 * Pre-warm the embedded agent import. Call this during plugin init to avoid
 * the cold-start penalty on the first actual extraction run.
 * Returns immediately (fire-and-forget) — errors are swallowed.
 */
export function prewarmEmbeddedAgent(
  logger?: RunnerLogger,
  agentRuntime?: EmbeddedAgentRuntimeLike,
): void {
  if (resolveInjectedRunEmbeddedPiAgent(agentRuntime)) {
    logger?.debug?.(
      `${TAG} prewarmEmbeddedAgent: runtime capability already available, skipping legacy preload`,
    );
    return;
  }

  loadRunEmbeddedPiAgent(logger).catch((err) => {
    logger?.warn(`${TAG} prewarmEmbeddedAgent: failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
  });
}

function collectText(payloads: Array<{ text?: string; isError?: boolean }> | undefined): string {
  const texts = (payloads ?? [])
    .filter((p) => !p.isError && typeof p.text === "string")
    .map((p) => p.text ?? "");
  return texts.join("\n").trim();
}

// ── Model resolution utilities ──

/** Parsed model reference: { provider, model } */
export interface ModelRef {
  provider: string;
  model: string;
}

/**
 * Parse a "provider/model" string into its components.
 * Returns undefined if the input is empty or doesn't contain a "/".
 *
 * Examples:
 *   "azure/gpt-5.2-chat"          → { provider: "azure", model: "gpt-5.2-chat" }
 *   "custom-host/org/model-v2"    → { provider: "custom-host", model: "org/model-v2" }
 *   ""                            → undefined
 *   "bare-model-name"             → undefined (no "/" — may be an alias)
 */
export function parseModelRef(raw: string | undefined): ModelRef | undefined {
  if (!raw) return undefined;
  const trimmed = raw.trim();
  if (!trimmed) return undefined;

  const slashIdx = trimmed.indexOf("/");
  if (slashIdx <= 0 || slashIdx === trimmed.length - 1) return undefined;

  return {
    provider: trimmed.slice(0, slashIdx),
    model: trimmed.slice(slashIdx + 1),
  };
}

/**
 * Resolve the user's default model from the main OpenClaw config.
 *
 * Resolution order:
 * 1. Read `agents.defaults.model` (string or { primary })
 * 2. If the value contains "/", parse directly
 * 3. If not (may be an alias), look up in `agents.defaults.models` alias table
 * 4. Return undefined if nothing resolves — let the core use its built-in default
 */
export function resolveModelFromMainConfig(config: unknown): ModelRef | undefined {
  if (!config || typeof config !== "object") return undefined;

  const cfg = config as Record<string, unknown>;
  const agents = cfg.agents as Record<string, unknown> | undefined;
  if (!agents || typeof agents !== "object") return undefined;

  const defaults = agents.defaults as Record<string, unknown> | undefined;
  if (!defaults || typeof defaults !== "object") return undefined;

  // Step 1: extract raw model value (string | { primary?: string })
  const modelCfg = defaults.model;
  let raw: string | undefined;
  if (typeof modelCfg === "string") {
    raw = modelCfg.trim();
  } else if (modelCfg && typeof modelCfg === "object") {
    const primary = (modelCfg as Record<string, unknown>).primary;
    raw = typeof primary === "string" ? primary.trim() : undefined;
  }
  if (!raw) return undefined;

  // Step 2: try direct "provider/model" parse
  const direct = parseModelRef(raw);
  if (direct) return direct;

  // Step 3: alias lookup — raw doesn't contain "/", check agents.defaults.models
  const models = defaults.models as Record<string, unknown> | undefined;
  if (!models || typeof models !== "object") return undefined;

  const rawLower = raw.toLowerCase();
  for (const [key, entry] of Object.entries(models)) {
    if (!entry || typeof entry !== "object") continue;
    const alias = (entry as Record<string, unknown>).alias;
    if (typeof alias !== "string") continue;
    if (alias.trim().toLowerCase() !== rawLower) continue;

    // key is "provider/model" format
    const resolved = parseModelRef(key);
    if (resolved) return resolved;
  }

  return undefined;
}

export interface CleanContextRunnerOptions {
  config: unknown; // OpenClawConfig
  provider?: string;
  model?: string;
  /**
   * Convenience field: full "provider/model" string.
   * Takes precedence over separate `provider`/`model` fields.
   * When all three (modelRef, provider, model) are omitted,
   * automatically falls back to the main config's `agents.defaults.model`.
   */
  modelRef?: string;
  /** Preferred runtime seam. When absent, falls back to the legacy dist bridge. */
  agentRuntime?: EmbeddedAgentRuntimeLike;
  /** Allow the LLM to use tools (read_file, write_to_file, etc). Default: false */
  enableTools?: boolean;
  /** Logger instance for detailed tracing */
  logger?: RunnerLogger;
}

// Stable empty directory used as default workspaceDir so that:
// 1. Bootstrap/skills scans find nothing → clean LLM context
// 2. The path is constant → plugin cacheKey stays stable (no re-registration)
let _cleanWorkspaceDir: string | undefined;
async function getCleanWorkspaceDir(): Promise<string> {
  if (_cleanWorkspaceDir) return _cleanWorkspaceDir;
  const dir = path.join(resolveOpenClawTmpDir(), "memory-tdai-clean-workspace");
  await fs.mkdir(dir, { recursive: true });
  _cleanWorkspaceDir = dir;
  return dir;
}

export class CleanContextRunner {
  private options: CleanContextRunnerOptions;
  private logger: RunnerLogger | undefined;
  /** Resolved provider after modelRef / config fallback */
  private resolvedProvider: string | undefined;
  /** Resolved model after modelRef / config fallback */
  private resolvedModel: string | undefined;

  constructor(options: CleanContextRunnerOptions) {
    this.options = options;
    this.logger = options.logger;

    // Model resolution priority:
    // 1. modelRef ("provider/model" string)  — highest
    // 2. explicit provider + model fields
    // 3. main config agents.defaults.model   — automatic fallback
    // 4. undefined (let core use built-in default)
    const fromRef = parseModelRef(options.modelRef);
    if (fromRef) {
      this.resolvedProvider = fromRef.provider;
      this.resolvedModel = fromRef.model;
    } else if (options.provider || options.model) {
      this.resolvedProvider = options.provider;
      this.resolvedModel = options.model;
    } else {
      // No explicit model specified — fall back to main config
      const fromConfig = resolveModelFromMainConfig(options.config);
      if (fromConfig) {
        this.resolvedProvider = fromConfig.provider;
        this.resolvedModel = fromConfig.model;
        this.logger?.debug?.(
          `${TAG} Using model from main config: ${fromConfig.provider}/${fromConfig.model}`,
        );
      }
      // else: both undefined → core will use its built-in default (anthropic/claude-opus-4-6)
    }
  }

  /**
   * Run a prompt in a fully isolated clean context.
   * Returns the LLM's text output.
   *
   * When `workspaceDir` is provided it overrides the default `process.cwd()`,
   * letting the LLM's file-tool calls resolve paths relative to a custom root.
   */
  async run(params: {
    prompt: string;
    /** Optional system prompt. When provided, `prompt` is used as the user message. */
    systemPrompt?: string;
    taskId: string;
    timeoutMs?: number;
    maxTokens?: number;
    workspaceDir?: string;
    /** Plugin instance ID for llm_call metric (optional) */
    instanceId?: string;
  }): Promise<string> {
    const runStartMs = Date.now();
    this.logger?.debug?.(`${TAG} run() start: taskId=${params.taskId}, timeout=${params.timeoutMs ?? 120_000}ms, tools=${this.options.enableTools ? "enabled" : "disabled"}, workspaceDir=${params.workspaceDir ?? "(default)"}`);

    const tmpDir = await fs.mkdtemp(
      path.join(resolveOpenClawTmpDir(), `memory-tdai-${params.taskId}-`),
    );
    const cleanWorkspace = params.workspaceDir ?? await getCleanWorkspaceDir();
    this.logger?.debug?.(`${TAG} run() tmpDir=${tmpDir}, cleanWorkspace=${cleanWorkspace}`);

    try {
      const sessionFile = path.join(tmpDir, "session.json");

      // Phase 1: Resolve runEmbeddedPiAgent (prefer runtime, fallback to legacy dist bridge)
      const importStartMs = Date.now();
      const runEmbeddedPiAgent = await resolveRunEmbeddedPiAgent(
        this.options.agentRuntime,
        this.logger,
      );
      const importElapsedMs = Date.now() - importStartMs;
      this.logger?.debug?.(`${TAG} run() runner resolution phase: ${importElapsedMs}ms`);

      // Derive a config with plugins disabled to prevent loadOpenClawPlugins
      // from re-registering plugins when the workspaceDir differs from the
      // gateway's original workspace (cacheKey mismatch triggers full reload).
      //
      // Security: restrict available tools to the minimal set needed for
      // scene extraction (read/write/edit). This prevents the LLM from
      // accessing exec, sessions, browser, cron, or any other powerful tools.
      // File deletion is handled via "soft-delete" (write empty) + cleanup afterward.
      const cleanConfig = {
        ...(this.options.config as Record<string, unknown>),
        plugins: {
          ...((this.options.config as Record<string, unknown>)?.plugins as Record<string, unknown> | undefined),
          enabled: false,
        },
        tools: {
          ...((this.options.config as Record<string, unknown>)?.tools as Record<string, unknown> | undefined),
          // When enableTools=true, restrict to the minimal set needed for
          // scene extraction (read/write/edit).
          // When enableTools=false, pass an empty allow list — disableTools:true
          // will prevent tools from being sent to the API entirely.
          allow: this.options.enableTools ? ["read", "write", "edit"] : [],
        },
        // Override the full agent system prompt with the caller's extraction-specific
        // system prompt. This replaces OpenClaw's default system prompt (identity,
        // AGENTS.md, workspace context, tool guidance, etc.) to:
        //   1. Save ~5000 tokens per LLM call
        //   2. Avoid instruction interference with extraction prompts
        agents: {
          ...((this.options.config as Record<string, unknown>)?.agents as Record<string, unknown> | undefined),
          defaults: {
            ...(((this.options.config as Record<string, unknown>)?.agents as Record<string, unknown> | undefined)?.defaults as Record<string, unknown> | undefined),
            systemPromptOverride:
              params.systemPrompt ||
              "You are a precise data extraction and generation assistant. Follow the user instructions exactly. Respond only with the requested output format.",
          },
        },
      };

      // systemPrompt is now in config.agents.defaults.systemPromptOverride
      // (actual [system] role), so user prompt only contains the actual content.
      const effectivePrompt = params.prompt;

      const ts = Date.now();
      const sessionId = `memory-${params.taskId}-session-${ts}`;
      const runId = `memory-${params.taskId}-run-${ts}`;
      this.logger?.debug?.(`${TAG} run() starting embedded agent: sessionId=${sessionId}, runId=${runId}, provider=${this.resolvedProvider ?? "(default)"}, model=${this.resolvedModel ?? "(default)"}`);

      // [l1-debug] INVOKE — what are we about to send to the embedded agent?
      const sysPromptOverrideLen =
        ((cleanConfig.agents as Record<string, unknown> | undefined)?.defaults as Record<string, unknown> | undefined)?.systemPromptOverride
          ? String(
              ((cleanConfig.agents as Record<string, unknown>).defaults as Record<string, unknown>).systemPromptOverride,
            ).length
          : 0;
      const toolsAllow =
        ((cleanConfig.tools as Record<string, unknown> | undefined)?.allow as unknown[] | undefined) ?? [];
      this.logger?.debug?.(
        `${TAG} [l1-debug] INVOKE taskId=${params.taskId}, provider=${this.resolvedProvider ?? "(default)"}, model=${this.resolvedModel ?? "(default)"}, promptLen=${effectivePrompt.length}, sysPromptOverrideLen=${sysPromptOverrideLen}, toolsAllow=${JSON.stringify(toolsAllow)}, timeoutMs=${params.timeoutMs ?? 120_000}`,
      );

      // Phase 2: Embedded agent run (LLM call + tool calls)
      const agentStartMs = Date.now();
      // extraSystemPrompt: fallback for openclaw < 2026.4.7 which does not support
      // config.agents.defaults.systemPromptOverride. On newer versions the
      // override takes precedence and this becomes a no-op append.
      const effectiveSystemPrompt =
        params.systemPrompt ||
        "You are a precise data extraction and generation assistant. Follow the user instructions exactly. Respond only with the requested output format.";
      const result = await runEmbeddedPiAgent({
        sessionId,
        sessionFile,
        workspaceDir: cleanWorkspace,
        config: cleanConfig,
        prompt: effectivePrompt,
        timeoutMs: params.timeoutMs ?? 120_000,
        runId,
        provider: this.resolvedProvider,
        model: this.resolvedModel,
        // When enableTools=false, pass disableTools:true so that no tool
        // definitions are sent to the API. This avoids polluting the LLM
        // context with tool schemas and prevents the model from attempting
        // tool calls during pure text extraction tasks.
        // If a provider (e.g. qwencode) rejects empty tools[], users should
        // switch to StandaloneLLMRunner via LLM configuration instead.
        disableTools: !this.options.enableTools,
        extraSystemPrompt: effectiveSystemPrompt,
        streamParams: {
          maxTokens: params.maxTokens,
        },
      });
      const agentElapsedMs = Date.now() - agentStartMs;
      this.logger?.debug?.(`${TAG} run() embedded agent completed: ${agentElapsedMs}ms`);

      // [l1-debug] RESULT — what did the embedded agent return?
      {
        const payloadsRaw = (result as Record<string, unknown> | undefined)?.payloads;
        const payloads = Array.isArray(payloadsRaw)
          ? (payloadsRaw as Array<Record<string, unknown>>)
          : [];
        const payloadKinds = payloads.map((p) => {
          if (typeof p?.type === "string") return p.type as string;
          if (typeof p?.kind === "string") return p.kind as string;
          return Object.keys(p ?? {}).slice(0, 3).join("|") || "unknown";
        });
        const errorPayloadCount = payloads.filter((p) => p?.isError === true).length;
        const joinedText = payloads
          .filter((p) => !p?.isError && typeof p?.text === "string")
          .map((p) => String(p.text ?? ""))
          .join("\n");
        const textPreview = joinedText.replace(/\s+/g, " ").slice(0, 200);
        this.logger?.debug?.(
          `${TAG} [l1-debug] RESULT taskId=${params.taskId}, elapsedMs=${agentElapsedMs}, payloadCount=${payloads.length}, payloadKinds=${JSON.stringify(payloadKinds)}, errorPayloadCount=${errorPayloadCount}, textLen=${joinedText.length}, textPreview=${JSON.stringify(textPreview)}`,
        );
      }

      // Phase 3: Collect output
      const text = collectText((result as Record<string, unknown>).payloads as Array<{ text?: string; isError?: boolean }> | undefined);
      const totalMs = Date.now() - runStartMs;

      if (!text) {
        // Empty output is normal when the LLM decides there is nothing to
        // extract (e.g. trivial greetings).  Log a warning instead of
        // throwing so the caller can handle it gracefully.
        this.logger?.warn?.(`${TAG} run() empty output after ${totalMs}ms (import=${importElapsedMs}ms, agent=${agentElapsedMs}ms) — treating as empty result`);
        // [l1-debug] EMPTY_DUMP — dump the full result shape so we can see where text went
        try {
          const dump = JSON.stringify(result, (_k, v) => {
            if (typeof v === "string" && v.length > 500) return v.slice(0, 500) + `…(+${v.length - 500})`;
            return v;
          }).slice(0, 2048);
          this.logger?.warn?.(`${TAG} [l1-debug] EMPTY_DUMP taskId=${params.taskId}, resultJson=${dump}`);
        } catch (dumpErr) {
          this.logger?.warn?.(`${TAG} [l1-debug] EMPTY_DUMP taskId=${params.taskId}, dumpFailed=${dumpErr instanceof Error ? dumpErr.message : String(dumpErr)}`);
        }
        // llm_call metric (empty output)
        if (params.instanceId && this.logger) {
          report("llm_call", {
            taskId: params.taskId,
            provider: this.resolvedProvider ?? "default",
            model: this.resolvedModel ?? "default",
            inputLength: params.prompt.length,
            outputLength: 0,
            totalDurationMs: totalMs,
            success: true,
            error: "empty_output",
          });
        }
        return "";
      }

      this.logger?.debug?.(`${TAG} run() completed: ${totalMs}ms total (import=${importElapsedMs}ms, agent=${agentElapsedMs}ms), output=${text.length} chars`);

      // ── llm_call metric (success) ──
      if (params.instanceId && this.logger) {
        report("llm_call", {
          taskId: params.taskId,
          provider: this.resolvedProvider ?? "default",
          model: this.resolvedModel ?? "default",
          inputLength: params.prompt.length,
          outputLength: text.length,
          totalDurationMs: totalMs,
          success: true,
          error: null,
        });
      }

      return text;
    } catch (err) {
      const totalMs = Date.now() - runStartMs;
      this.logger?.error(`${TAG} run() failed after ${totalMs}ms: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
      // ── llm_call metric (failure) ──
      if (params.instanceId && this.logger) {
        report("llm_call", {
          taskId: params.taskId,
          provider: this.resolvedProvider ?? "default",
          model: this.resolvedModel ?? "default",
          inputLength: params.prompt.length,
          outputLength: 0,
          totalDurationMs: totalMs,
          success: false,
          error: err instanceof Error ? err.message : String(err),
        });
      }
      throw err;
    } finally {
      await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    }
  }
}
