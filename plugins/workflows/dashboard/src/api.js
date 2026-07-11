// API error normalization + SDK-backed caller.
//
// formatApiError() unwraps FastAPI 4xx message strings ("400: {"detail":"…"}"),
// prefers structured { message } when present, and falls back to the original
// throw value so we never surface "Unknown error" for a thrown string.
//
// createApi() closes over the SDK's fetchJSON so callers never reach for raw
// fetch — keeps the plugin inside the host's network and auth boundary.

export function formatApiError(error) {
  // ponytail: error.message normalization runs before object handling because
  // FastAPI wraps its responses as `new Error('400: {"detail":"…"}')`.
  if (error instanceof Error && typeof error.message === "string") {
    return formatApiError(error.message);
  }
  if (error && typeof error === "object") {
    if (typeof error.message === "string" && error.message) {
      return error.message;
    }
    if (typeof error.detail === "string" && error.detail) {
      return formatApiError(error.detail);
    }
  }
  if (typeof error === "string") {
    const jsonStart = error.indexOf("{");
    if (jsonStart !== -1) {
      try {
        const parsed = JSON.parse(error.slice(jsonStart));
        return formatApiError(parsed);
      } catch { /* fall through to raw string */ }
    }
    return error;
  }
  return "Unknown error";
}

export function createApi(fetchJSON, basePath) {
  if (typeof fetchJSON !== "function") {
    throw new Error("createApi requires the SDK fetchJSON implementation");
  }
  const base = String(basePath || "").replace(/\/$/, "");
  return function api(path, options) {
    const suffix = String(path || "");
    return fetchJSON(base + suffix, options);
  };
}
