/**
 * API trace 日志脱敏与序列化（多模块复用）。
 */
import { maskKeyValue } from "../metadata/utils/user-key.js";

export const API_TRACE_SENSITIVE_KEYS = new Set([
  "password",
  "initial_password",
  "default_user_key",
  "user_key",
  "key_value",
  "granted_by_key",
  "owner_user_key",
  "creator_user_key",
  "authorization",
  "api_key",
  "token",
  "secret",
  "bearer",
]);

export function truncateApiString(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars)}…[truncated ${value.length - maxChars} chars]`;
}

export function sanitizeApiPayload(value: unknown, maxFieldChars: number, depth = 0): unknown {
  if (depth > 8) return "[max_depth]";
  if (value === null || value === undefined) return value;
  if (typeof value === "string") return truncateApiString(value, maxFieldChars);
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeApiPayload(item, maxFieldChars, depth + 1));
  }
  if (typeof value !== "object") return String(value);

  const out: Record<string, unknown> = {};
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    const lower = key.toLowerCase();
    if (API_TRACE_SENSITIVE_KEYS.has(lower)) {
      out[key] = typeof raw === "string" && raw.length > 0 ? maskKeyValue(raw) : "[redacted]";
      continue;
    }
    out[key] = sanitizeApiPayload(raw, maxFieldChars, depth + 1);
  }
  return out;
}

export function serializeForApiLog(value: unknown, maxFieldChars: number, maxJsonChars: number): string {
  try {
    const json = JSON.stringify(sanitizeApiPayload(value, maxFieldChars));
    if (json.length <= maxJsonChars) return json;
    return `${json.slice(0, maxJsonChars)}…[truncated ${json.length - maxJsonChars} chars]`;
  } catch {
    return "[unserializable]";
  }
}

export function redactSqlParams(params: unknown[], maxFieldChars: number): unknown[] {
  return params.map((p) => {
    if (typeof p === "string") {
      if (p.length > 32 && /^uk_|sk_|sk-mem-|Bearer /i.test(p)) return "[redacted]";
      return truncateApiString(p, maxFieldChars);
    }
    return p;
  });
}
