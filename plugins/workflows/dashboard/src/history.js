// History-mode pure helpers — no side effects, no DOM.
//
// ponytail: thin wrappers over URL building and status checks.
// upgrade path: if filters grow beyond 5, switch to URLSearchParams.

const CANCELABLE_STATUSES = new Set(["queued", "running", "waiting"]);

export function serializeFilters(opts) {
  if (!opts || typeof opts !== "object") return "";
  const parts = [];
  if (opts.workflowId) parts.push("workflow_id=" + encodeURIComponent(opts.workflowId));
  if (opts.status) parts.push("status=" + encodeURIComponent(opts.status));
  if (opts.version != null) parts.push("version=" + encodeURIComponent(opts.version));
  if (opts.triggerId) parts.push("trigger_id=" + encodeURIComponent(opts.triggerId));
  if (opts.before && opts.before.createdAt != null && opts.before.executionId) {
    parts.push("before_created_at=" + encodeURIComponent(opts.before.createdAt));
    parts.push("before_execution_id=" + encodeURIComponent(opts.before.executionId));
  }
  if (opts.limit != null) parts.push("limit=" + encodeURIComponent(opts.limit));
  return parts.join("&");
}

export function historyListPath(opts) {
  const qs = serializeFilters(opts);
  return "/executions" + (qs ? "?" + qs : "");
}

export function detailUrl(executionId) {
  if (!executionId) return "";
  return "/executions/" + encodeURIComponent(executionId) + "/detail";
}

export function canCancel(status) {
  return CANCELABLE_STATUSES.has(status);
}

export function buildRerunBody(input) {
  return { input: input && typeof input === "object" ? input : {} };
}

export function buildDetailGraphParams(execution) {
  if (!execution || execution.version == null) return {};
  return { version: execution.version };
}
