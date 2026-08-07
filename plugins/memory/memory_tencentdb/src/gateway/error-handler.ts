/**
 * Gateway error handler (H-13).
 *
 * Goal: never leak err.message, err.stack, file paths, SQL fragments,
 * upstream API error bodies, or credentials to clients. Translate any
 * unknown error into a {status, client-safe payload, full log line}.
 *
 * Design:
 *   - Recognized error types → preserve their (already safe) code + message
 *     (e.g. PayloadTooLargeError = CR-7, RecallFailure = H-15, SeedValidationError)
 *   - Anything else (5xx) → generic "Internal server error" + traceId
 *   - Server log: traceId + full err.message + stack (for debugging)
 *   - Client log: only the safe message + traceId so the user can quote it
 *     to support team for log correlation
 *
 * Note: this module is deliberately leaf-level (no imports from server.ts /
 * v2-router.ts) so both can import it without circular deps. Use duck-typing
 * for PayloadTooLargeError detection to avoid pulling server.ts in.
 */

import { randomUUID } from "node:crypto";
import { RecallFailure } from "../core/hooks/recall-errors.js";

export interface ClientFacingError {
  /** Stable business code: HTTP status (4xx/5xx) for unknown errors, RecallError.code for recall failures. */
  code: number;
  /** Already-sanitized human-readable message safe for clients. */
  message: string;
  /** UUID for log correlation; user quotes this when reporting issues. */
  trace_id: string;
  /** Whether retry is sensible. */
  retryable?: boolean;
}

export interface ClassifiedError {
  /** HTTP status code to send. */
  status: number;
  /** Payload safe to send to the client. */
  client: ClientFacingError;
  /** Full description (including stack/cause) for the server log only. */
  logLine: string;
}

/**
 * Classify any thrown error into a (status, client payload, log line) triple.
 *
 * Recognized:
 *   - PayloadTooLargeError (CR-7) — detected via duck-typing on statusCode === 413
 *   - Invalid JSON body — parseJsonBody throws Error("Invalid JSON body") on malformed input
 *   - RecallFailure (H-15) — code from RecallError taxonomy
 *   - SeedValidationError — 400 with generic message
 *   - Anything else — 500 Internal server error
 */
export function classifyError(err: unknown): ClassifiedError {
  const trace_id = randomUUID();

  // 1. PayloadTooLargeError (CR-7) — duck-typed to avoid circular import on gateway/server.ts.
  const statusCode = (err as { statusCode?: unknown })?.statusCode;
  if (statusCode === 413) {
    const msg = err instanceof Error ? err.message : "Payload too large";
    return {
      status: 413,
      client: { code: 413, message: msg, trace_id, retryable: false },
      logLine: `[${trace_id}] PayloadTooLargeError: ${msg}`,
    };
  }

  // 1b. Invalid JSON body / decompression error — parseJsonBody throws this fixed message on
  // JSON.parse failure or gzip/deflate decompression failure. Client error, not server fault.
  if (err instanceof Error && err.message === "Invalid JSON body") {
    return {
      status: 400,
      client: { code: 400, message: "Invalid JSON body", trace_id, retryable: false },
      logLine: `[${trace_id}] InvalidJsonBody: request body could not be parsed as JSON`,
    };
  }

  // 1c. COS AppendPositionErr — concurrent append conflict. The client should retry.
  // This is a transient conflict, not a permanent error.
  if (err instanceof Error && /AppendPositionErr|Position not equal object length/i.test(err.message)) {
    return {
      status: 409,
      client: { code: 409, message: "Concurrent write conflict, please retry", trace_id, retryable: true },
      logLine: `[${trace_id}] CosAppendConflict: ${err.message}`,
    };
  }

  // 1d. Unsupported Content-Encoding — parseJsonBody rejects with this message when the
  // client sends an encoding the server does not support (not gzip/deflate/identity).
  if (err instanceof Error && err.message.startsWith("Unsupported Content-Encoding:")) {
    const safeMsg = err.message.replace(/[^\w\s:/-]/g, "");
    return {
      status: 415,
      client: { code: 415, message: safeMsg, trace_id, retryable: false },
      logLine: `[${trace_id}] UnsupportedEncoding: ${err.message}`,
    };
  }

  // 2. RecallFailure (H-15) — the recall layer has already produced a safe message.
  // (Gateway handlers normally translate this into RecallResult.error before reaching
  // the global catch, but if a code path forgets, this fallback ensures no raw leak.)
  if (err instanceof RecallFailure) {
    const re = err.recallError;
    const causeStr = err.cause instanceof Error
      ? (err.cause.stack ?? err.cause.message)
      : err.cause !== undefined ? String(err.cause) : "(no cause)";
    return {
      status: re.category === "config" ? 503 : 500,
      client: { code: re.code, message: re.message, trace_id, retryable: re.retryable },
      logLine: `[${trace_id}] RecallFailure code=${re.code} category=${re.category} cause=${sanitize(causeStr)}`,
    };
  }

  // 3. SeedValidationError — known user-input class
  if (err && typeof err === "object" && (err as { name?: string }).name === "SeedValidationError") {
    const errMsg = err instanceof Error ? err.message : String(err);
    return {
      status: 400,
      client: { code: 400, message: "Invalid seed input", trace_id, retryable: false },
      logLine: `[${trace_id}] SeedValidationError: ${sanitize(errMsg)}`,
    };
  }

  // 4. Generic 5xx — strictly hide err.message from client
  const errMsg = err instanceof Error ? err.message : String(err);
  const errStack = err instanceof Error ? err.stack : undefined;
  return {
    status: 500,
    client: { code: 500, message: "Internal server error", trace_id, retryable: true },
    logLine: `[${trace_id}] UnhandledError: ${sanitize(errMsg)}\n${sanitize(errStack ?? "(no stack)")}`,
  };
}

// ============================
// Sanitize: strip secrets from a string before it enters logs.
// ============================

/**
 * Best-effort secret redaction for log strings (M-2 / H-13).
 *
 * Removes / masks:
 *   - OpenAI / DeepSeek / Anthropic style API keys (sk-*, sk-ant-*)
 *   - HTTP Authorization headers (Bearer, Basic)
 *   - JSON fields named SecretKey / apiKey / password / token / authorization
 *
 * This is a heuristic safety net, not a substitute for not logging secrets
 * in the first place. Apply to err.message / err.stack / any object dumped
 * to the log via JSON.stringify.
 */
export function sanitize(input: string): string {
  if (typeof input !== "string") return String(input);
  return input
    // sk-ant-xxx — Anthropic (must come before the generic sk-* rule below
    // since "sk-ant-..." also matches /sk-[A-Za-z0-9_-]{16,}/)
    .replace(/sk-ant-[A-Za-z0-9_-]{16,}/g, "sk-ant-***")
    // sk-xxxxx (16+ chars) — OpenAI, DeepSeek, etc.
    .replace(/sk-[A-Za-z0-9_-]{16,}/g, "sk-***")
    // Bearer / Basic auth headers
    .replace(/(Bearer|Basic)\s+[A-Za-z0-9._\-+/=]+/gi, "$1 ***")
    // JSON-like "field": "value" for sensitive fields (case-insensitive, both single/double quotes)
    .replace(
      /("(?:SecretKey|apiKey|api_key|password|token|authorization|TmpSecretId|TmpSecretKey|TmpToken)"\s*:\s*)"[^"]*"/gi,
      '$1"***"',
    )
    .replace(
      /('(?:SecretKey|apiKey|api_key|password|token|authorization|TmpSecretId|TmpSecretKey|TmpToken)'\s*:\s*)'[^']*'/gi,
      "$1'***'",
    );
}

/**
 * Recursively sanitize an object/array — replaces values of known-sensitive keys with '***'
 * and runs `sanitize()` on string values to catch ad-hoc secrets that aren't keyed.
 *
 * Useful for `logger.debug(sanitizeObject({cosConfig, request}))`.
 */
export function sanitizeObject(obj: unknown): unknown {
  if (typeof obj === "string") return sanitize(obj);
  if (Array.isArray(obj)) return obj.map(sanitizeObject);
  if (obj && typeof obj === "object") {
    const r: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      if (/secretkey|apikey|api_key|password|token|authorization|TmpSecret/i.test(k)) {
        r[k] = "***";
      } else {
        r[k] = sanitizeObject(v);
      }
    }
    return r;
  }
  return obj;
}
