/**
 * API trace 统一写日志：stdout 单行 JSON（interface: tdai-metadata-api）。
 */
import { getApiRequestContext } from "./api-request-context.js";
import { getApiTraceConfig, isApiTraceActive } from "./api-log-config.js";
import { buildStdoutPayload, writeApiTraceStdout } from "./api-trace-stdout.js";

export type ApiTraceLayer = "http" | "service" | "store" | "db" | "remote";
export type ApiTraceLevel = "info" | "warn" | "error";

function inferLayer(event: string): ApiTraceLayer {
  if (event.startsWith("api.remote.")) return "remote";
  if (event.startsWith("api.http.")) return "http";
  if (event.startsWith("api.service.")) return "service";
  if (event.startsWith("api.store.")) return "store";
  if (event.startsWith("api.db.")) return "db";
  return "http";
}

export function logApiTrace(
  level: ApiTraceLevel,
  event: string,
  attrs: Record<string, string | number | boolean>,
  opts?: { requestId?: string; err?: Error },
): void {
  if (!isApiTraceActive()) return;

  try {
    const cfg = getApiTraceConfig();
    const ctx = getApiRequestContext();
    const requestId = opts?.requestId ?? ctx?.requestId;
    if (!requestId) return;

    const module = ctx?.module ?? cfg.policy.module;
    const merged: Record<string, string | number | boolean> = {
      module,
      layer: inferLayer(event),
      request_id: requestId,
      ...attrs,
    };
    if (ctx?.route) merged.route = ctx.route;
    if (ctx?.internal) merged.internal = true;

    const levelUpper = level.toUpperCase();
    const payload = buildStdoutPayload(levelUpper, event, cfg.policy.profile, merged);
    writeApiTraceStdout(payload);
  } catch {
    // 静默失败
  }
}
