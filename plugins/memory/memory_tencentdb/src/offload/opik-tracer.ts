/**
 * Opik observability tracer for context offload plugin.
 * Wraps the opik npm package with graceful degradation when not installed.
 */
import type { PluginLogger } from "./types.js";
import { getEnv } from "../utils/env.js";

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

let client: OpikClient | null = null;
let tracerEnabled = false;
let tracerInitTried = false;

function extractLayerTag(stage: string): string {
  const match = stage.match(/^(L\d+(?:\.\d+)?)/i);
  if (!match) return "Lx-unknown";
  return match[1].toUpperCase();
}

function extractL3TriggerSource(stage: string): string | undefined {
  if (!stage || !stage.startsWith("L3")) return undefined;
  if (stage.includes("after_tool_call")) return "after_tool_call";
  if (stage.includes("llm_input")) return "llm_input";
  if (stage.includes("before_prompt")) return "before_prompt_reapply";
  return "L3_unknown";
}

function isInLoopStage(stage: string): boolean {
  return typeof stage === "string" && stage.includes("after_tool_call");
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

function getOpikConfigFromOpenClawConfig(config: Record<string, unknown>): {
  enabled: boolean;
  apiUrl?: string;
  apiKey?: string;
  workspaceName: string;
  projectName: string;
} {
  const plugins = config.plugins as Record<string, unknown> | undefined;
  const entries = plugins?.entries as Record<string, Record<string, unknown>> | undefined;
  const opikEntry = entries?.["opik-openclaw"];
  const opikCfg = opikEntry?.config as Record<string, unknown> | undefined;
  const enabled = opikEntry?.enabled !== false && opikCfg?.enabled !== false;
  const apiUrl =
    typeof opikCfg?.apiUrl === "string" ? opikCfg.apiUrl : getEnv("OPIK_URL_OVERRIDE");
  const apiKey =
    typeof opikCfg?.apiKey === "string" ? opikCfg.apiKey : getEnv("OPIK_API_KEY");
  const workspaceName =
    typeof opikCfg?.workspaceName === "string" && (opikCfg.workspaceName as string).trim()
      ? (opikCfg.workspaceName as string)
      : getEnv("OPIK_WORKSPACE") ?? "default";
  const projectName =
    typeof opikCfg?.projectName === "string" && (opikCfg.projectName as string).trim()
      ? (opikCfg.projectName as string)
      : getEnv("OPIK_PROJECT_NAME") ?? "openclaw";
  return { enabled, apiUrl, apiKey, workspaceName, projectName };
}

export function initOffloadOpikTracer(
  openClawConfig: Record<string, unknown>,
  logger: PluginLogger,
): void {
  if (tracerInitTried) return;
  tracerInitTried = true;
  try {
    const cfg = getOpikConfigFromOpenClawConfig(openClawConfig);
    if (!cfg.enabled) return;
    // Dynamic import — graceful when opik is not installed
    let OpikConstructor: new (params: Record<string, unknown>) => OpikClient;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const opikModule = require("opik") as { Opik: new (params: Record<string, unknown>) => OpikClient };
      OpikConstructor = opikModule.Opik;
    } catch {
      logger.debug?.("[context-offload] opik package not available, tracer disabled");
      return;
    }
    client = new OpikConstructor({
      ...(cfg.apiKey ? { apiKey: cfg.apiKey } : {}),
      ...(cfg.apiUrl ? { apiUrl: cfg.apiUrl } : {}),
      workspaceName: cfg.workspaceName,
      projectName: cfg.projectName,
    });
    tracerEnabled = true;
    logger.debug?.(
      `[context-offload] Opik tracer enabled: project=${cfg.projectName}, workspace=${cfg.workspaceName}`,
    );
  } catch (err) {
    tracerEnabled = false;
    client = null;
    logger.debug?.(`[context-offload] Opik tracer init failed: ${String(err)}`);
  }
}

export function traceOffloadDecision(params: {
  sessionKey?: string | null;
  stage: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  logger?: PluginLogger;
}): void {
  if (!tracerEnabled || !client) return;
  try {
    const layerTag = extractLayerTag(params.stage);
    const l3TriggerSource = extractL3TriggerSource(params.stage);
    const threadId =
      params.sessionKey && params.sessionKey.trim()
        ? params.sessionKey
        : `offload-${Date.now()}`;
    const inLoop = isInLoopStage(params.stage);
    const out = params.output ?? {};
    const phase = typeof params.input.phase === "string" ? params.input.phase : undefined;
    const skTag = params.sessionKey ? `session:${params.sessionKey}` : "session:unknown";
    const trace = client.trace({
      name: `context-offload:${params.stage} [${params.sessionKey ?? "no-session"}]`,
      threadId,
      input: params.input,
      metadata: {
        plugin: "openclaw-context-offload",
        category: "decision",
        stage: params.stage,
        layer: layerTag,
        sessionKey: params.sessionKey ?? undefined,
        ...(inLoop ? { inloop: true } : {}),
        ...(l3TriggerSource ? { l3TriggerSource } : {}),
        ...(phase ? { phase } : {}),
      },
      tags: [
        "context-offload",
        "decision",
        layerTag,
        skTag,
        ...(inLoop ? ["inloop"] : []),
        ...(l3TriggerSource ? [`trigger:${l3TriggerSource}`] : []),
        ...(phase ? [`phase:${phase}`] : []),
      ],
    });
    trace.update({ output: out });
    trace.end();
    void client.flush().catch(() => undefined);
  } catch (err) {
    params.logger?.warn?.(`[context-offload] Opik decision trace failed: ${String(err)}`);
  }
}

/**
 * Serialize a single message into a diagnostic object for tracing.
 * Outputs full content text (no truncation) for debugging purposes.
 */
function serializeMessageForTrace(msg: any, index: number): Record<string, unknown> {
  const role = msg.role ?? msg.message?.role ?? msg.type ?? "unknown";
  const flags: string[] = [];
  if (msg._mmdContextMessage) flags.push(`mmdCtx=${msg._mmdContextMessage}`);
  if (msg._mmdInjection) flags.push("mmdInj");
  if (msg._offloaded) flags.push("offloaded");

  const content = msg.content ?? msg.message?.content;
  let contentText: string;
  let contentLength: number;
  if (typeof content === "string") {
    contentLength = content.length;
    contentText = content;
  } else if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const c of content) {
      if (typeof c !== "object" || c === null) continue;
      if (c.type === "text" && typeof c.text === "string") {
        parts.push(c.text);
      } else if (c.type === "tool_use") {
        const inputStr = c.input != null ? JSON.stringify(c.input) : "";
        parts.push(`[tool_use: ${c.name ?? "?"} id=${c.id ?? "?"} input=${inputStr}]`);
      } else if (c.type === "tool_result") {
        const resultStr = typeof c.content === "string" ? c.content : JSON.stringify(c.content ?? "");
        parts.push(`[tool_result: id=${c.tool_use_id ?? "?"} content=${resultStr}]`);
      } else {
        parts.push(`[${c.type ?? "unknown_block"}]`);
      }
    }
    contentText = parts.join("\n");
    contentLength = contentText.length;
  } else {
    contentLength = 0;
    contentText = "(empty)";
  }

  const toolCallId = msg.toolCallId ?? msg.tool_call_id ?? msg.message?.toolCallId ?? msg.message?.tool_call_id;

  return {
    i: index,
    role,
    ...(flags.length > 0 ? { flags } : {}),
    len: contentLength,
    content: contentText,
    ...(toolCallId ? { toolCallId } : {}),
  };
}

/**
 * Trace a full messages snapshot — used for debugging message state at key points.
 * Creates a separate "messages-snapshot" category trace.
 */
export function traceMessagesSnapshot(params: {
  sessionKey?: string | null;
  stage: string;
  messages: any[];
  label?: string;
  extra?: Record<string, unknown>;
  logger?: PluginLogger;
}): void {
  if (!tracerEnabled || !client) return;
  try {
    const threadId =
      params.sessionKey && params.sessionKey.trim()
        ? params.sessionKey
        : `offload-${Date.now()}`;
    const skTag = params.sessionKey ? `session:${params.sessionKey}` : "session:unknown";
    const msgs = params.messages ?? [];
    const serialized = msgs.map((m, i) => serializeMessageForTrace(m, i));

    // Aggregate stats
    const mmdCount = msgs.filter((m: any) => m._mmdContextMessage || m._mmdInjection).length;
    const offloadedCount = msgs.filter((m: any) => m._offloaded).length;
    const roleBreakdown: Record<string, number> = {};
    for (const m of msgs) {
      const role = m.role ?? m.message?.role ?? m.type ?? "unknown";
      roleBreakdown[role] = (roleBreakdown[role] ?? 0) + 1;
    }

    const trace = client.trace({
      name: `messages-snapshot:${params.stage}${params.label ? ` (${params.label})` : ""} [${params.sessionKey ?? "no-session"}]`,
      threadId,
      input: {
        stage: params.stage,
        label: params.label,
        messageCount: msgs.length,
        mmdCount,
        offloadedCount,
        roleBreakdown,
        ...(params.extra ?? {}),
      },
      metadata: {
        plugin: "openclaw-context-offload",
        category: "messages-snapshot",
        stage: params.stage,
        sessionKey: params.sessionKey ?? undefined,
      },
      tags: ["context-offload", "messages-snapshot", skTag],
    });
    trace.update({
      output: {
        messages: serialized,
        messageCount: msgs.length,
        mmdCount,
        offloadedCount,
        roleBreakdown,
      },
    });
    trace.end();
    void client.flush().catch(() => undefined);
  } catch (err) {
    params.logger?.warn?.(`[context-offload] Opik messages-snapshot trace failed: ${String(err)}`);
  }
}

export function traceOffloadModelIo(params: {
  sessionKey?: string | null;
  stage: string;
  provider?: string;
  model: string;
  url: string;
  systemPrompt: string;
  userPrompt: string;
  responseContent: string;
  usage?: Record<string, unknown>;
  status: "ok" | "error";
  errorMessage?: string;
  durationMs: number;
  logger?: PluginLogger;
}): void {
  if (!tracerEnabled || !client) return;
  try {
    const layerTag = extractLayerTag(params.stage);
    const threadId =
      params.sessionKey && params.sessionKey.trim()
        ? params.sessionKey
        : `offload-${Date.now()}`;
    const dur = params.durationMs;
    const durStr = formatDuration(dur);
    const durBucket = durationBucketTag(dur);
    const skTag = params.sessionKey ? `session:${params.sessionKey}` : "session:unknown";
    const trace = client.trace({
      name: `${params.model} · context-offload · ${durStr} [${params.sessionKey ?? "no-session"}]`,
      threadId,
      metadata: {
        plugin: "openclaw-context-offload",
        category: "llm",
        stage: params.stage,
        layer: layerTag,
        provider: params.provider,
        model: params.model,
        sessionKey: params.sessionKey ?? undefined,
        durationMs: dur,
        duration: durStr,
      },
      tags: ["context-offload", "llm", layerTag, durBucket, skTag],
    });
    const span = trace.span({
      name: `${params.model} · ${durStr}`,
      type: "llm",
      model: params.model,
      provider: params.provider,
      input: {
        url: params.url,
        systemPrompt: params.systemPrompt,
        userPrompt: params.userPrompt,
      },
      metadata: {
        stage: params.stage,
        layer: layerTag,
        sessionKey: params.sessionKey ?? undefined,
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
        duration: durStr,
      },
    });
    span.end();
    trace.end();
    void client.flush().catch(() => undefined);
  } catch (err) {
    params.logger?.warn?.(`[context-offload] Opik model I/O trace failed: ${String(err)}`);
  }
}
