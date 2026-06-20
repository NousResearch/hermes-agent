// Fatal-error helpers for the Photon Spectrum sidecar.
//
// Keep this separate from index.mjs so the detection logic can be unit-tested
// without starting Spectrum or requiring real Photon credentials.

function appendErrorParts(parts, value, seen) {
  if (value == null) return;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    parts.push(String(value));
    return;
  }
  if (typeof value !== "object") return;
  if (seen.has(value)) return;
  seen.add(value);

  for (const key of ["name", "message", "stack", "code", "grpcCode", "details", "path"]) {
    if (value[key] != null) parts.push(String(value[key]));
  }
  if (value.cause) appendErrorParts(parts, value.cause, seen);
  if (Array.isArray(value.errors)) {
    for (const error of value.errors) appendErrorParts(parts, error, seen);
  }
}

export function errorToText(error) {
  const parts = [];
  appendErrorParts(parts, error, new Set());
  if (!parts.length) return String(error ?? "");
  return parts.join("\n");
}

export function isFatalInboundStreamError(error) {
  const text = errorToText(error);
  return (
    /catchUpEvents concurrency limit/i.test(text) ||
    /EventService\/CatchUpEvents[\s\S]*RESOURCE_EXHAUSTED/i.test(text) ||
    /RESOURCE_EXHAUSTED[\s\S]*EventService\/CatchUpEvents/i.test(text)
  );
}

export function classifyRecoverableOutboundError(error) {
  const text = errorToText(error);
  if (/\[upstream\]\s*Connection dropped/i.test(text)) {
    return "upstream_connection_dropped";
  }
  return null;
}
