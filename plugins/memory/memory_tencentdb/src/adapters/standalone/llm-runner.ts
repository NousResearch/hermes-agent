/**
 * StandaloneLLMRunner — powered by Vercel AI SDK (`ai` + `@ai-sdk/openai`).
 *
 * This runner does NOT depend on OpenClaw's `runEmbeddedPiAgent`. It is designed
 * for the Hermes Gateway scenario where TDAI runs as an independent Node.js sidecar
 * without the OpenClaw host.
 *
 * Capabilities:
 * - `enableTools: false`: pure text output (L1 extraction, L1 dedup)
 * - `enableTools: true`: automatic tool-call loop with local file operations
 *   (L2 scene, L3 persona) via AI SDK's `maxSteps`
 *
 * Tool sandbox:
 *   When tools are enabled, three basic file operations are exposed:
 *   `read`, `write`, `edit` — aligned with OpenClaw host tool names.
 *   All file paths are resolved relative to `workspaceDir`, enforcing sandbox boundaries.
 */

import fsPromises from "node:fs/promises";
import path from "node:path";
import { generateText, tool, stepCountIs, jsonSchema } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { report } from "../../core/report/reporter.js";
import type {
  LLMRunner,
  LLMRunParams,
  LLMRunnerFactory,
  LLMRunnerCreateOptions,
  Logger,
} from "../../core/types.js";
import type { LLMUsage } from "../../core/report/metric-tracking-runner.js";

const TAG = "[memory-tdai] [standalone-runner]";

// Max iterations in the tool-call loop to prevent infinite loops
const MAX_TOOL_ITERATIONS = 20;

// ============================
// experimental_telemetry.metadata 组装
// ============================

/**
 * 组装传给 Vercel AI SDK 的 experimental_telemetry.metadata。
 *
 * 字段策略：
 *   - instanceId  : 始终写入（未传时降级为 "unknown"）
 *   - traceName   : 存在时 → 写入 langfuseTraceName + langfuseUpdateParent=true
 *                  （让 Langfuse 用业务语义命名 trace，覆盖默认的 Unnamed）
 *   - tags        : 非空数组才写入（避免空 tag 污染 Langfuse 索引）
 *   - sessionId   : 非空字符串才写入（Langfuse UI 顶级筛选字段）
 *   - userId      : 非空字符串才写入（Langfuse UI 顶级筛选字段）
 *
 * 未传对应字段时，metadata 里也不出现该键 —— 保持与旧行为完全一致。
 */
function buildTelemetryMetadata(params: LLMRunParams): Record<string, unknown> {
  const meta: Record<string, unknown> = {
    instanceId: params.instanceId ?? "unknown",
  };
  if (params.traceName) {
    meta.langfuseTraceName = params.traceName;
    // langfuseUpdateParent=true 让子 span 的 name/attrs 传播到 Langfuse trace 根
    meta.langfuseUpdateParent = true;
  }
  if (Array.isArray(params.tags) && params.tags.length > 0) {
    meta.tags = params.tags;
  }
  if (typeof params.sessionId === "string" && params.sessionId.length > 0) {
    meta.sessionId = params.sessionId;
  }
  if (typeof params.userId === "string" && params.userId.length > 0) {
    meta.userId = params.userId;
  }
  return meta;
}

// ============================
// Configuration
// ============================

export interface StandaloneLLMConfig {
  /** OpenAI-compatible API base URL (e.g. "https://api.openai.com/v1"). */
  baseUrl: string;
  /** API key for authentication. */
  apiKey: string;
  /** Default model name (e.g. "gpt-4o"). */
  model: string;
  /** Default max output tokens. */
  maxTokens?: number;
  /** Request timeout in milliseconds (default: 120_000). */
  timeoutMs?: number;
  /**
   * LLM 访问模式（gateway 层解释；runner 拿到的是已解析后的 baseUrl/apiKey）：
   *   - "openai": 直连通用 OpenAI 兼容服务（默认，向后兼容）
   *   - "proxy":  走 context_proxy，运行时会自动把 baseUrl 拼成
   *               `${baseUrl}/proxy/<instanceId>/v1`，apiKey 用 metadata.systemUser.memory.userKey
   */
  provider?: "openai" | "proxy";
  /** provider=proxy 时的可选配置。 */
  proxy?: {
    /** 是否用 memory systemUser.userKey 作为 Authorization（默认 true）。 */
    useMemorySystemUserKey?: boolean;
  };
}

// ============================
// Sandboxed tool execution helpers
// ============================

function resolveSandboxedPath(workspaceDir: string, relativePath: string): string | null {
  const resolved = path.resolve(workspaceDir, relativePath);
  if (!resolved.startsWith(path.resolve(workspaceDir))) {
    return null;
  }
  return resolved;
}

// ============================
// Tool definitions (Vercel AI SDK `tool()` format)
// ============================

function createSandboxedTools(workspaceDir: string, logger?: Logger) {
  return {
    read: tool({
      description: "Read the contents of a file at the given relative path.",
      inputSchema: jsonSchema<{ path: string }>({
        type: "object",
        properties: {
          path: { type: "string", description: "Relative file path to read." },
        },
        required: ["path"],
      }),
      execute: (async (args: { path: string }) => {
        const resolved = resolveSandboxedPath(workspaceDir, args.path);
        if (!resolved) return JSON.stringify({ error: `Path "${args.path}" escapes workspace boundary.` });
        try {
          const content = await fsPromises.readFile(resolved, "utf-8");
          logger?.debug?.(`${TAG} read: "${args.path}" → ${content.length} chars`);
          return content;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          logger?.warn?.(`${TAG} read failed: ${msg}`);
          return JSON.stringify({ error: msg });
        }
      }) as any,
    }),

    write: tool({
      description: "Write content to a file at the given relative path. Creates or overwrites.",
      inputSchema: jsonSchema<{ path: string; content: string }>({
        type: "object",
        properties: {
          path: { type: "string", description: "Relative file path to write." },
          content: { type: "string", description: "Content to write." },
        },
        required: ["path", "content"],
      }),
      execute: (async (args: { path: string; content: string }) => {
        const resolved = resolveSandboxedPath(workspaceDir, args.path);
        if (!resolved) return JSON.stringify({ error: `Path "${args.path}" escapes workspace boundary.` });
        try {
          await fsPromises.mkdir(path.dirname(resolved), { recursive: true });
          await fsPromises.writeFile(resolved, args.content, "utf-8");
          logger?.debug?.(`${TAG} write: "${args.path}" → ${args.content.length} chars`);
          return JSON.stringify({ success: true });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          logger?.warn?.(`${TAG} write failed: ${msg}`);
          return JSON.stringify({ error: msg });
        }
      }) as any,
    }),

    edit: tool({
      description: "Apply one or more text replacements to a file. Each edit replaces an exact substring.",
      inputSchema: jsonSchema<{ path: string; edits: Array<{ oldText: string; newText: string }> }>({
        type: "object",
        properties: {
          path: { type: "string", description: "Relative file path." },
          edits: {
            type: "array",
            description: "Array of replacements to apply sequentially.",
            items: {
              type: "object",
              properties: {
                oldText: { type: "string", description: "Exact string to find." },
                newText: { type: "string", description: "Replacement string." },
              },
              required: ["oldText", "newText"],
            },
          },
        },
        required: ["path", "edits"],
      }),
      execute: (async (args: { path: string; edits: Array<{ oldText: string; newText: string }> }) => {
        const resolved = resolveSandboxedPath(workspaceDir, args.path);
        if (!resolved) return JSON.stringify({ error: `Path "${args.path}" escapes workspace boundary.` });
        if (!args.edits || args.edits.length === 0) return JSON.stringify({ error: "edits array cannot be empty." });
        try {
          let content = await fsPromises.readFile(resolved, "utf-8");
          for (const edit of args.edits) {
            if (!edit.oldText) return JSON.stringify({ error: "oldText cannot be empty." });
            if (!content.includes(edit.oldText)) {
              return JSON.stringify({ error: `oldText not found in file "${args.path}": ${edit.oldText.slice(0, 80)}` });
            }
            content = content.replace(edit.oldText, edit.newText);
          }
          await fsPromises.writeFile(resolved, content, "utf-8");
          logger?.debug?.(`${TAG} edit: "${args.path}" → ${args.edits.length} replacement(s), ${content.length} chars`);
          return JSON.stringify({ success: true });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          logger?.warn?.(`${TAG} edit failed: ${msg}`);
          return JSON.stringify({ error: msg });
        }
      }) as any,
    }),
  };
}

/** Read-only tool subset — currently empty.
 *
 * Historically returned `{ read: all.read }` so the AI SDK wouldn't reject
 * an empty tools object. In practice this caused weak models (e.g. small
 * Doubao endpoints) to hallucinate calls like `read({"path":"."})` during
 * pure-text tasks (L1 extraction), triggering EISDIR on the sandbox dir
 * and burning a turn on a useless tool call.
 *
 * Modern AI SDK (v6) accepts an undefined `tools` field, so the runner now
 * skips the `tools`/`stopWhen` parameters entirely when tools are disabled
 * — see `generateText` invocation below.
 */
function createReadOnlyTools(_workspaceDir: string, _logger?: Logger) {
  return {};
}

// ============================
// StandaloneLLMRunner
// ============================

export class StandaloneLLMRunner implements LLMRunner {
  private config: StandaloneLLMConfig;
  private model: string;
  private enableTools: boolean;
  private logger?: Logger;

  /**
   * Side-channel: 最近一次 run() 调用的 token usage。
   * 由 MetricTrackingRunner 装饰器读取，用于精确上报 credit。
   * 不改变 LLMRunner 接口签名。
   */
  lastUsage?: LLMUsage;

  constructor(opts: {
    config: StandaloneLLMConfig;
    model?: string;
    enableTools?: boolean;
    logger?: Logger;
  }) {
    this.config = opts.config;
    this.model = opts.model ?? opts.config.model;
    this.enableTools = opts.enableTools ?? false;
    this.logger = opts.logger;
  }

  async run(params: LLMRunParams): Promise<string> {
    const runStartMs = Date.now();
    const timeoutMs = params.timeoutMs ?? this.config.timeoutMs ?? 120_000;
    const maxTokens = params.maxTokens ?? this.config.maxTokens ?? 4096;
    const workspaceDir = params.workspaceDir ?? process.cwd();
    // Per-call overrides — when the caller supplies their own tools (e.g.
    // SkillExtractor's skill_list/skill_view/skill_manage), they trump the
    // runner-level enableTools default. This lets one runner instance
    // serve both pure-text L1 extraction and tool-driven skill review.
    const callerProvidedTools = params.tools && Object.keys(params.tools).length > 0;
    const effectiveEnableTools = params.enableTools ?? this.enableTools;
    const maxIterations = params.maxIterations ?? MAX_TOOL_ITERATIONS;

    this.logger?.debug?.(
      `${TAG} run() start: taskId=${params.taskId}, model=${this.model}, ` +
      `tools=${effectiveEnableTools}${callerProvidedTools ? "(caller)" : ""}, timeout=${timeoutMs}ms`,
    );

    // Create OpenAI-compatible provider via AI SDK
    // Use "compatible" mode to call /chat/completions (not Responses API),
    // which works with all OpenAI-compatible backends (DeepSeek, Qwen, etc.)
    const provider = createOpenAI({
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
      compatibility: "compatible",
    });

    // Select tools based on mode + storage
    // Service mode (COS): use storage-backed tools → LLM reads/writes via StorageAdapter
    // Standalone mode (local FS): use sandboxed FS tools → LLM reads/writes local files
    // enableTools=false: omit tools entirely so the model cannot hallucinate calls.
    // Caller-provided tools (params.tools) override the defaults — used by
    // SkillExtractor to inject domain-specific tools (skill_list, etc.).
    let tools: Record<string, unknown> | undefined;
    if (callerProvidedTools && effectiveEnableTools) {
      tools = params.tools;
      this.logger?.debug?.(`${TAG} Using caller-provided tools: [${Object.keys(tools!).join(", ")}]`);
    } else if (effectiveEnableTools && params.storage) {
      const { createStorageTools } = await import("./storage-tools.js");
      tools = createStorageTools(params.storage, params.storagePrefix ?? "", this.logger);
      this.logger?.debug?.(`${TAG} Using storage-backed tools (prefix="${params.storagePrefix ?? ""}")`);
    } else if (effectiveEnableTools) {
      tools = createSandboxedTools(workspaceDir, this.logger);
    } else {
      tools = undefined; // pure-text task — never expose any tool to the model
    }

    try {
      // H-11 Step 2: combine internal timeout with caller-provided abortSignal
      // (e.g. pipeline-worker lost its lock and wants the LLM call to bail out).
      // AbortSignal.any (Node 20+) aborts when ANY of the listed signals abort.
      const timeoutSignal = AbortSignal.timeout(timeoutMs);
      const combinedSignal = params.abortSignal
        ? AbortSignal.any([timeoutSignal, params.abortSignal])
        : timeoutSignal;

      const result = await generateText({
        model: provider.chat(this.model),
        system: params.systemPrompt,
        prompt: params.prompt,
        // Only attach tools when actually enabled — passing an empty object
        // (or even a tools-only-with-`read`) makes some OpenAI-compatible
        // backends emit spurious tool calls on pure-text tasks.
        ...(tools && Object.keys(tools).length > 0
          ? { tools, stopWhen: stepCountIs(maxIterations) }
          : {}),
        maxOutputTokens: maxTokens,
        abortSignal: combinedSignal,
        experimental_telemetry: {
          isEnabled: true,
          functionId: params.taskId,
          metadata: buildTelemetryMetadata(params),
        },
      });

      const text = (result.text ?? "").trim();
      const totalMs = Date.now() - runStartMs;

      // 暴露 token usage 到 side-channel（供 MetricTrackingRunner 读取）
      if (result.usage) {
        this.lastUsage = {
          promptTokens: result.usage.promptTokens ?? 0,
          completionTokens: result.usage.completionTokens ?? 0,
          totalTokens: (result.usage.promptTokens ?? 0) + (result.usage.completionTokens ?? 0),
        };
      } else {
        this.lastUsage = undefined;
      }

      this.logger?.debug?.(
        `${TAG} run() completed: ${totalMs}ms, steps=${result.steps.length}, output=${text.length} chars`,
      );

      // Log each step's activity (tool calls + text output)
      for (const step of result.steps) {
        const calls = step.toolCalls ?? [];
        const textLen = step.text?.length ?? 0;
        if (calls.length > 0) {
          const callSummary = calls.map((tc) =>
            `${tc.toolName}(${JSON.stringify(tc.input).slice(0, 120)})`,
          ).join(", ");
          this.logger?.debug?.(
            `${TAG} step[${step.stepNumber}] toolCalls: ${callSummary}`,
          );
        }
        if (textLen > 0) {
          this.logger?.debug?.(
            `${TAG} step[${step.stepNumber}] text: ${textLen} chars, finishReason=${step.finishReason}`,
          );
        }
        if (calls.length === 0 && textLen === 0) {
          this.logger?.debug?.(
            `${TAG} step[${step.stepNumber}] empty (no tools, no text), finishReason=${step.finishReason}`,
          );
        }
      }

      // Metric
      if (params.instanceId) {
        report("llm_call", {
          taskId: params.taskId,
          provider: "standalone",
          model: this.model,
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
      const errMsg = err instanceof Error ? err.message : String(err);
      this.logger?.error(`${TAG} run() failed after ${totalMs}ms: ${errMsg}`);

      if (params.instanceId) {
        report("llm_call", {
          taskId: params.taskId,
          provider: "standalone",
          model: this.model,
          inputLength: params.prompt.length,
          outputLength: 0,
          totalDurationMs: totalMs,
          success: false,
          error: errMsg,
        });
      }

      throw err;
    }
  }
}

// ============================
// StandaloneLLMRunnerFactory
// ============================

export interface StandaloneLLMRunnerFactoryOptions {
  /** LLM API configuration. */
  config: StandaloneLLMConfig;
  /** Logger instance. */
  logger?: Logger;
}

/**
 * Factory that creates StandaloneLLMRunner instances.
 *
 * Used by the Gateway and Hermes host adapters.
 */
export class StandaloneLLMRunnerFactory implements LLMRunnerFactory {
  private config: StandaloneLLMConfig;
  private logger?: Logger;

  constructor(opts: StandaloneLLMRunnerFactoryOptions) {
    this.config = opts.config;
    this.logger = opts.logger;
  }

  createRunner(opts?: LLMRunnerCreateOptions): LLMRunner {
    const enableTools = opts?.enableTools ?? false;
    const modelRef = opts?.modelRef;

    // Parse "provider/model" → just use the model part for OpenAI-compatible API
    let model = this.config.model;
    if (modelRef) {
      const slashIdx = modelRef.indexOf("/");
      model = slashIdx > 0 ? modelRef.slice(slashIdx + 1) : modelRef;
    }

    this.logger?.debug?.(
      `${TAG} Creating StandaloneLLMRunner: model=${model}, tools=${enableTools}`,
    );

    return new StandaloneLLMRunner({
      config: this.config,
      model,
      enableTools,
      logger: this.logger,
    });
  }
}
