// Typed wrappers over the management REST API. Shapes mirror web/src/lib/api.ts
// (the source of truth); only the subset the ported pages consume is declared.
import { apiGet, apiPost, apiDelete, apiFetch } from "./client";

// ── Status / system ─────────────────────────────────────────────────
export interface StatusResponse {
  active_sessions: number;
  auth_required?: boolean;
  can_update_hermes?: boolean;
  config_path: string;
  config_version: number;
  env_path: string;
  gateway_pid: number | null;
  gateway_platforms: Record<string, { connected?: boolean; enabled?: boolean }>;
  gateway_running: boolean;
  gateway_state: string | null;
  hermes_home: string;
  latest_config_version: number;
  release_date: string;
  version: string;
}

export interface SystemStats {
  os: string;
  os_release: string;
  platform: string;
  arch: string;
  hostname: string;
  python_version: string;
  hermes_version: string;
  cpu_count: number | null;
  psutil: boolean;
  cpu_percent?: number;
  uptime_seconds?: number;
  memory?: { total: number; available: number; used: number; percent: number };
  disk?: { total: number; used: number; free: number; percent: number };
  process?: { pid: number; rss: number; num_threads: number };
}

export const getStatus = () => apiGet<StatusResponse>("/api/status");
export const getSystemStats = () => apiGet<SystemStats>("/api/system/stats");

// ── Logs ────────────────────────────────────────────────────────────
export interface LogsResponse {
  file: string;
  lines: string[];
}
export function getLogs(params: {
  file?: string;
  lines?: number;
  level?: string;
  component?: string;
}): Promise<LogsResponse> {
  const qs = new URLSearchParams();
  if (params.file) qs.set("file", params.file);
  if (params.lines) qs.set("lines", String(params.lines));
  if (params.level && params.level !== "ALL") qs.set("level", params.level);
  if (params.component && params.component !== "all") qs.set("component", params.component);
  return apiGet<LogsResponse>(`/api/logs?${qs.toString()}`);
}

// ── Env / keys ──────────────────────────────────────────────────────
export interface EnvVarInfo {
  is_set: boolean;
  redacted_value: string | null;
  description: string;
  url: string | null;
  category: string;
  is_password: boolean;
  tools: string[];
  advanced: boolean;
  channel_managed?: boolean;
  custom?: boolean;
}
export const getEnvVars = () => apiGet<Record<string, EnvVarInfo>>("/api/env");
export const setEnvVar = (key: string, value: string) =>
  apiFetch<{ ok: boolean }>("/api/env", {
    method: "PUT",
    body: JSON.stringify({ key, value }),
  });
export const deleteEnvVar = (key: string) =>
  apiFetch<{ ok: boolean }>("/api/env", {
    method: "DELETE",
    body: JSON.stringify({ key }),
  });
export const revealEnvVar = (key: string) =>
  apiPost<{ key: string; value: string }>("/api/env/reveal", { key });

// ── Config (raw YAML) ───────────────────────────────────────────────
export const getConfigRaw = () => apiGet<{ yaml: string }>("/api/config/raw");
export const saveConfigRaw = (yaml_text: string) =>
  apiFetch<{ ok: boolean }>("/api/config/raw", {
    method: "PUT",
    body: JSON.stringify({ yaml_text }),
  });

// ── Sessions ────────────────────────────────────────────────────────
export interface SessionInfo {
  id: string;
  source: string | null;
  model: string | null;
  title: string | null;
  started_at: number;
  last_active: number;
  is_active: boolean;
  message_count: number;
  input_tokens: number;
  output_tokens: number;
  preview: string | null;
}
export interface PaginatedSessions {
  sessions: SessionInfo[];
  total: number;
  limit: number;
  offset: number;
}
export const getSessions = (limit = 50, offset = 0, order: "created" | "recent" = "recent") =>
  apiGet<PaginatedSessions>(`/api/sessions?limit=${limit}&offset=${offset}&order=${order}`);
export const deleteSession = (id: string) =>
  apiDelete<{ ok: boolean }>(`/api/sessions/${encodeURIComponent(id)}`);
export const renameSession = (id: string, title: string) =>
  apiFetch<{ ok: boolean; title: string }>(`/api/sessions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });

// ── Model ───────────────────────────────────────────────────────────
export interface ModelInfoResponse {
  model: string;
  provider: string;
  effective_context_length: number;
  capabilities: {
    supports_tools?: boolean;
    supports_vision?: boolean;
    supports_reasoning?: boolean;
    max_output_tokens?: number;
    model_family?: string;
  };
}
export interface ModelOptionProvider {
  name: string;
  slug: string;
  models?: string[];
  total_models?: number;
  is_current?: boolean;
  authenticated?: boolean;
  warning?: string;
}
export interface ModelOptionsResponse {
  model?: string;
  provider?: string;
  providers?: ModelOptionProvider[];
}
export interface ModelAssignmentRequest {
  scope: "main" | "auxiliary";
  provider: string;
  model: string;
  base_url?: string;
  task?: string;
  confirm_expensive_model?: boolean;
}
export interface ModelAssignmentResponse {
  ok?: boolean;
  model?: string;
  provider?: string;
  warning?: string;
  error?: string;
}
export const getModelInfo = () => apiGet<ModelInfoResponse>("/api/model/info");
export const getModelOptions = () =>
  apiGet<ModelOptionsResponse>("/api/model/options?include_unconfigured=1");
export const setModelAssignment = (req: ModelAssignmentRequest) =>
  apiPost<ModelAssignmentResponse>("/api/model/set", req);
