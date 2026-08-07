/**
 * Opik observability tracer for offload server.
 * Wraps the opik npm package with graceful degradation when not installed.
 * Configuration is read from environment variables (no OpenClaw plugin config).
 */

// Opik client types (minimal shape to avoid hard dependency)
interface OpikClient {
  trace(params: Record<string, unknown>): OpikTrace;
  flush(): Promise<void>;
}
interface OpikTrace {
  update(params: Record<string, unknown>): void;
  end(): void;
  span(params: Record<string, unknown>): OpikSpan;
}
interface OpikSpan {
  update(params: Record<string, unknown>): void;
  end(): void;
}

// ─── Module State ────────────────────────────────────────────────────────────

let client: OpikClient | null = null;
let tracerEnabled = false;
let tracerInitTried = false;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function extractLayerTag(stage: string): string {
  const match = stage.match(/^(L\d+(?:\.\d+)?)/i);
  if (!match) return "Lx-unknown";
  return match[1].toUpperCase();
}

function durationBucketTag(ms: number): string {
  if (typeof ms !== "number" || ms < 0) return "duration:unknown";
  if (ms < 1000) return "duration:<1s";
  if (ms < 5000) return "duration:1-5s";
  if (ms < 15000) return "duration:5-15s";
  if (ms < 30000) return "duration:15-30s";
  return "duration:>30s";
}

function formatDuration(ms: number): string {
  if (typeof ms !== "number" || ms < 0) return "?";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// ─── Logger Interface ────────────────────────────────────────────────────────

export interface TracerLogger {
  info: (message: string, ...args: unknown[]) => void;
  warn: (message: string, ...args: unknown[]) => void;
  debug?: (message: string, ...args: unknown[]) => void;
}

// ─── Init ────────────────────────────────────────────────────────────────────

/**
 * Initialize the offload server Opik tracer.
 * Reads config from environment variables:
 *   OPIK_ENABLED        — set to "true" to enable (default: disabled)
 *   OPIK_URL_OVERRIDE   — Opik server URL
 *   OPIK_API_KEY        — API key
 *   OPIK_WORKSPACE      — workspace name (default: "default")
 *   OPIK_PROJECT_NAME   — project name (default: "openclaw-offload-server")
 */
export async function initServerOpikTracer(logger: TracerLogger): Promise<void> {
  if (tracerInitTried) return;
  tracerInitTried = true;
  try {
    const enabled = process.env.OPIK_ENABLED === "true";
    if (!enabled) {
      logger.debug?.("[offload-server] Opik tracer disabled (OPIK_ENABLED != true)");
      return;
    }

    const apiUrl = process.env.OPIK_URL_OVERRIDE;
    const apiKey = process.env.OPIK_API_KEY;
    const workspaceName = process.env.OPIK_WORKSPACE ?? "default";
    const projectName = process.env.OPIK_PROJECT_NAME ?? "openclaw-offload-server";

    // Dynamic import — graceful when opik is not installed
    let OpikConstructor: new (params: Record<string, unknown>) => OpikClient;
    let disableOpikLogger: (() => void) | undefined;
    try {
      const opikModule = await import("opik") as {
        Opik: new (params: Record<string, unknown>) => OpikClient;
        disableLogger?: () => void;
        setLoggerLevel?: (level: string) => void;
      };
      OpikConstructor = opikModule.Opik;
      disableOpikLogger = opikModule.disableLogger;
    } catch {
      logger.debug?.("[offload-server] opik package not available, tracer disabled");
      return;
    }

    // Suppress opik internal logs (flush messages, ANSI color noise)
    if (disableOpikLogger) {
      disableOpikLogger();
    }

    client = new OpikConstructor({
      ...(apiKey ? { apiKey } : {}),
      ...(apiUrl ? { apiUrl } : {}),
      workspaceName,
      projectName,
    });
    tracerEnabled = true;
    logger.info(
      `[offload-server] Opik tracer enabled: project=${projectName}, workspace=${workspaceName}`,
    );
  } catch (err) {
    tracerEnabled = false;
    client = null;
    logger.warn(`[offload-server] Opik tracer init failed: ${String(err)}`);
  }
}

/**
 * Check if the tracer is enabled and ready to trace.
 */
export function isTracerEnabled(): boolean {
  return tracerEnabled && client !== null;
}

// ─── Trace: Model I/O (L1/L1.5/L2 LLM calls) ───────────────────────────────

/**
 * Trace LLM model I/O for offload server L1/L1.5/L2 stages.
 */
export function traceServerModelIo(params: {
  sessionId: string;
  stage: string;
  model: string;
  systemPrompt: string;
  userPrompt: string;
  responseContent: string;
  usage?: { promptTokens?: number; completionTokens?: number; totalTokens?: number };
  status: "ok" | "error";
  errorMessage?: string;
  durationMs: number;
  logger?: TracerLogger;
}): void {
  if (!tracerEnabled || !client) return;
  try {
    const layerTag = extractLayerTag(params.stage);
    const threadId = params.sessionId || `offload-server-${Date.now()}`;
    const dur = params.durationMs;
    const durStr = formatDuration(dur);
    const durBucket = durationBucketTag(dur);
    const skTag = `session:${params.sessionId || "unknown"}`;

    const trace = client.trace({
      name: `${params.model} · offload-server · ${params.stage} · ${durStr}`,
      threadId,
      metadata: {
        plugin: "openclaw-offload-server",
        category: "llm",
        stage: params.stage,
        layer: layerTag,
        model: params.model,
        sessionId: params.sessionId,
        durationMs: dur,
        duration: durStr,
      },
      tags: ["offload-server", "llm", layerTag, durBucket, skTag],
    });

    const span = trace.span({
      name: `${params.model} · ${params.stage} · ${durStr}`,
      type: "llm",
      model: params.model,
      input: {
        systemPrompt: params.systemPrompt,
        userPrompt: params.userPrompt,
      },
      metadata: {
        stage: params.stage,
        layer: layerTag,
        sessionId: params.sessionId,
        durationMs: dur,
        duration: durStr,
      },
    });

    span.update({
      output: {
        responseContent: params.responseContent,
        usage: params.usage,
        durationMs: dur,
        duration: durStr,
        error: params.errorMessage,
      },
      metadata: {
        status: params.status,
        durationMs: dur,
      },
    });
    span.end();
    trace.end();
    void client.flush().catch(() => undefined);
  } catch (err) {
    params.logger?.warn?.(`[offload-server] Opik model I/O trace failed: ${String(err)}`);
  }
}

// ─── Trace: Compaction Decision ──────────────────────────────────────────────

/**
 * Trace compaction (L3) decision and results.
 */
export function traceServerCompaction(params: {
  sessionId: string;
  level: string;
  ratio: number;
  contextWindow: number;
  totalTokensBefore: number;
  totalTokensAfter: number;
  originalMsgCount: number;
  compactedMsgCount: number;
  report: Record<string, unknown>;
  messages: unknown[];
  durationMs: number;
  logger?: TracerLogger;
}): void {
  if (!tracerEnabled || !client) return;
  try {
    const threadId = params.sessionId || `offload-server-${Date.now()}`;
    const dur = params.durationMs;
    const durStr = formatDuration(dur);
    const durBucket = durationBucketTag(dur);
    const skTag = `session:${params.sessionId || "unknown"}`;

    const trace = client.trace({
      name: `compaction · L3 · ${params.level} · ${durStr} [${params.sessionId}]`,
      threadId,
      input: {
        level: params.level,
        ratio: params.ratio,
        contextWindow: params.contextWindow,
        totalTokensBefore: params.totalTokensBefore,
        originalMsgCount: params.originalMsgCount,
      },
      metadata: {
        plugin: "openclaw-offload-server",
        category: "compaction",
        stage: "L3",
        layer: "L3",
        level: params.level,
        sessionId: params.sessionId,
        durationMs: dur,
        duration: durStr,
      },
      tags: ["offload-server", "compaction", "L3", `level:${params.level}`, durBucket, skTag],
    });

    // Serialize messages for full snapshot
    const serializedMessages = params.messages.map((msg: any, i: number) => {
      const role = msg.role ?? "unknown";
      const content = typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map((c: any) => {
              if (c.type === "text") return c.text;
              if (c.type === "tool_use") return `[tool_use: ${c.name} id=${c.id}]`;
              if (c.type === "tool_result") return `[tool_result: id=${c.tool_use_id} content=${typeof c.content === "string" ? c.content.slice(0, 500) : JSON.stringify(c.content).slice(0, 500)}]`;
              return `[${c.type ?? "unknown"}]`;
            }).join("\n")
          : "";
      return { i, role, content, ...(msg._mmdContextMessage ? { mmdCtx: true } : {}), ...(msg._offloaded ? { offloaded: true } : {}) };
    });

    trace.update({
      output: {
        totalTokensAfter: params.totalTokensAfter,
        compactedMsgCount: params.compactedMsgCount,
        tokenReduction: params.totalTokensBefore - params.totalTokensAfter,
        report: params.report,
        messages: serializedMessages,
      },
    });
    trace.end();
    void client.flush().catch(() => undefined);
  } catch (err) {
    params.logger?.warn?.(`[offload-server] Opik compaction trace failed: ${String(err)}`);
  }
}

// ─── Trace: Task Decision (L1.5 judgment result) ─────────────────────────────

/**
 * Trace L1.5 task judgment decision.
 */
export function traceServerTaskDecision(params: {
  sessionId: string;
  judgment: Record<string, unknown>;
  durationMs: number;
  logger?: TracerLogger;
}): void {
  if (!tracerEnabled || !client) return;
  try {
    const threadId = params.sessionId || `offload-server-${Date.now()}`;
    const dur = params.durationMs;
    const durStr = formatDuration(dur);
    const skTag = `session:${params.sessionId || "unknown"}`;

    const trace = client.trace({
      name: `task-decision · L1.5 · ${durStr} [${params.sessionId}]`,
      threadId,
      input: {
        stage: "L1.5",
        sessionId: params.sessionId,
      },
      metadata: {
        plugin: "openclaw-offload-server",
        category: "decision",
        stage: "L1.5",
        layer: "L1.5",
        sessionId: params.sessionId,
        durationMs: dur,
        duration: durStr,
      },
      tags: ["offload-server", "decision", "L1.5", skTag],
    });

    trace.update({ output: params.judgment });
    trace.end();
    void client.flush().catch(() => undefined);
  } catch (err) {
    params.logger?.warn?.(`[offload-server] Opik task decision trace failed: ${String(err)}`);
  }
}
