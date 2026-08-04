/**
 * 元数据 API 可观测性：入口 / 出口 / 异常 trace 日志（含脱敏后的入参、出参）。
 *
 * 覆盖 /v3/meta/* 与 /v3/internal/meta/* 的 dispatch 层。
 * stdout 单行 JSON（interface: tdai-metadata-api，经 api-trace 模块）。
 */
import { trace } from "../../core/report/trace.js";
import { getObservabilityBackend } from "../../core/report/factory.js";
import type { ApiResponseEnvelope } from "../../gateway/v2-schemas.js";
import {
  getApiTraceConfig,
  logApiTrace,
  sanitizeApiPayload,
  serializeForApiLog,
} from "../../api-trace/index.js";

const MAX_LOG_FIELD_CHARS = 1_024;
const MAX_LOG_JSON_CHARS = 8_192;

/** @deprecated 使用 sanitizeApiPayload；保留单测与外部导入兼容。 */
export function sanitizeMetaPayload(value: unknown, depth = 0): unknown {
  return sanitizeApiPayload(value, MAX_LOG_FIELD_CHARS, depth);
}

export interface MetaApiTraceContext {
  route: string;
  requestId: string;
  instanceId?: string;
  userId?: string;
  internal?: boolean;
  startedAtMs: number;
}

export function createMetaApiTraceContext(args: {
  route: string;
  requestId: string;
  instanceId?: string;
  userId?: string;
  internal?: boolean;
}): MetaApiTraceContext {
  return {
    ...args,
    startedAtMs: Date.now(),
  };
}

function durationMs(ctx: MetaApiTraceContext): number {
  return Date.now() - ctx.startedAtMs;
}

function resolveActiveTraceId(): string {
  try {
    const ctx = getObservabilityBackend().tracePropagation.serializeTraceContext();
    const traceId = (ctx as Record<string, unknown>)._traceId;
    return typeof traceId === "string" ? traceId : "";
  } catch {
    return "";
  }
}

function baseAttrs(ctx: MetaApiTraceContext): Record<string, string | number | boolean> {
  const cfg = getApiTraceConfig();
  const attrs: Record<string, string | number | boolean> = {
    source_file: ctx.internal ? "internal-meta-router.ts" : "v3-meta-router.ts",
    route: ctx.route,
    request_id: ctx.requestId,
    duration_ms: durationMs(ctx),
    module: cfg.policy.module,
    profile: cfg.policy.profile,
  };
  const traceId = resolveActiveTraceId();
  if (traceId) attrs.trace_id = traceId;
  if (ctx.instanceId) attrs.instance_id = ctx.instanceId;
  if (ctx.userId) attrs.user_id = ctx.userId;
  if (ctx.internal) attrs.internal = true;
  return attrs;
}

function serializeBody(value: unknown): string {
  const policy = getApiTraceConfig().policy;
  return serializeForApiLog(
    value,
    policy.maxFieldChars || MAX_LOG_FIELD_CHARS,
    policy.maxJsonChars || MAX_LOG_JSON_CHARS,
  );
}

function maybeReportOtel(event: string, attrs: Record<string, string | number | boolean>): void {
  if (!getApiTraceConfig().policy.httpOtelReport) return;
  trace.report(event, attrs);
}

/** 请求进入 dispatch（鉴权通过后、handler 执行前）。 */
export function logMetaApiEntry(ctx: MetaApiTraceContext, body?: unknown): void {
  const policy = getApiTraceConfig().policy;
  const attrs: Record<string, string | number | boolean> = {
    ...baseAttrs(ctx),
  };
  if (policy.httpBodyOnSuccess) {
    attrs.request_body = serializeBody(body ?? {});
  }
  logApiTrace("info", "api.http.request", attrs, { requestId: ctx.requestId });
  maybeReportOtel("api.http.request", { ...attrs, success: true });
}

/** 正常返回 envelope（含业务 4xx）。 */
export function logMetaApiResponse(ctx: MetaApiTraceContext, envelope: ApiResponseEnvelope, httpStatus: number): void {
  const success = envelope.code === 0;
  const policy = getApiTraceConfig().policy;
  const attrs: Record<string, string | number | boolean> = {
    ...baseAttrs(ctx),
    http_status: httpStatus,
    envelope_code: envelope.code,
    envelope_message: envelope.message ?? "",
    success,
  };
  if (!success || policy.httpBodyOnSuccess) {
    attrs.response_body = serializeBody(envelope.data ?? {});
  }
  logApiTrace(success ? "info" : "warn", "api.http.response", attrs, { requestId: ctx.requestId });
  maybeReportOtel("api.http.response", attrs);
}

/** 未捕获异常或 MetadataError 抛出路径。 */
export function logMetaApiError(
  ctx: MetaApiTraceContext,
  err: unknown,
  extra?: { envelopeCode?: number; httpStatus?: number },
): void {
  const message = err instanceof Error ? err.message : String(err);
  const code = extra?.envelopeCode ?? 500;
  const attrs = {
    ...baseAttrs(ctx),
    http_status: extra?.httpStatus ?? code,
    envelope_code: code,
    error_message: message,
    success: false,
  };
  logApiTrace(
    "error",
    "api.http.error",
    attrs,
    { requestId: ctx.requestId, err: err instanceof Error ? err : undefined },
  );
  maybeReportOtel("api.http.error", attrs);
}

/** 鉴权/参数校验等提前返回（无 handler 执行）。 */
export function logMetaApiRejected(
  ctx: MetaApiTraceContext,
  args: { httpStatus: number; envelopeCode: number; message: string; body?: unknown },
): void {
  const attrs: Record<string, string | number | boolean> = {
    ...baseAttrs(ctx),
    http_status: args.httpStatus,
    envelope_code: args.envelopeCode,
    envelope_message: args.message,
    success: false,
    request_body: serializeBody(args.body ?? {}),
  };
  logApiTrace("warn", "api.http.rejected", attrs, { requestId: ctx.requestId });
  maybeReportOtel("api.http.rejected", attrs);
}
