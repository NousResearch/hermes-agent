export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) || "http://localhost:8647";

export type TaskStatus = "pending" | "in_progress" | "completed" | "failed" | "cancelled";
export type TaskAgent = "hermes" | "codex" | "chez" | "system";
export type TaskPriority = "low" | "medium" | "high" | "urgent";

export type HealthRecord = {
  status: string;
};

export type SessionRecord = {
  id: string;
  display_name?: string | null;
  platform?: string | null;
  chat_type?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  total_tokens?: number | null;
  estimated_cost_usd?: number | null;
};

export type TaskRecord = {
  id: string;
  title: string;
  status: TaskStatus;
  agent: TaskAgent;
  room?: string;
  priority: TaskPriority;
  goal: string;
  context?: string;
  result?: string | null;
  error?: string | null;
  handoff_id?: string | null;
  created_at: string;
  updated_at: string;
  tags?: string[];
};

export type HandoffRecord = {
  id: string;
  from_agent: string;
  to_agent: string;
  status: string;
  payload: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  created_at: string;
  completed_at?: string | null;
  log_refs?: string[];
};

export type LogRecord = {
  id: string;
  level: "INFO" | "WARN" | "ERROR";
  message: string;
  timestamp: string;
  metadata?: {
    agent?: string;
    task_id?: string | null;
    handoff_id?: string | null;
  };
};

export type AdapterRecord = {
  name: string;
  status: "online" | "offline";
  version: string;
  model: string;
};

export type SettingsRecord = {
  hermes_adapter_path: string;
  codex_adapter_path: string;
  backend_port: number;
  frontend_port: number;
  codex_workdir: string;
};

export type CodexExecResult = {
  output: string;
  stdout?: string;
  stderr?: string;
  exit_code: number;
  tokens_used?: number;
  workdir: string;
  prompt_preview: string;
  session_id: string;
  timeout?: number;
  created_at?: string;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || `Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthRecord> {
  return requestJson<HealthRecord>("/api/health");
}

export function listSessions(): Promise<SessionRecord[]> {
  return requestJson<SessionRecord[]>("/api/sessions");
}

export function listTasks(): Promise<TaskRecord[]> {
  return requestJson<TaskRecord[]>("/api/tasks");
}

export function createTask(payload: Partial<TaskRecord> & { title: string; goal: string }): Promise<TaskRecord> {
  return requestJson<TaskRecord>("/api/tasks", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runTask(taskId: string): Promise<HandoffRecord> {
  return requestJson<HandoffRecord>(`/api/tasks/${taskId}/run`, { method: "POST" });
}

export function retryTask(taskId: string): Promise<HandoffRecord> {
  return requestJson<HandoffRecord>(`/api/tasks/${taskId}/retry`, { method: "POST" });
}

export function requeueTask(taskId: string): Promise<TaskRecord> {
  return requestJson<TaskRecord>(`/api/tasks/${taskId}/requeue`, { method: "POST" });
}

export function patchTask(taskId: string, payload: Partial<TaskRecord>): Promise<TaskRecord> {
  return requestJson<TaskRecord>(`/api/tasks/${taskId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function listHandoffs(): Promise<HandoffRecord[]> {
  return requestJson<HandoffRecord[]>("/api/handoffs");
}

export function createHandoff(payload: {
  from_agent: string;
  to_agent: string;
  payload: Record<string, unknown> | string;
  auto_run?: boolean;
}): Promise<HandoffRecord> {
  return requestJson<HandoffRecord>("/api/handoffs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runAgainHandoff(handoffId: string): Promise<HandoffRecord> {
  return requestJson<HandoffRecord>(`/api/handoffs/${handoffId}/run-again`, { method: "POST" });
}

export function listLogs(filters?: {
  level?: string;
  agent?: string;
  task_id?: string;
  handoff_id?: string;
}): Promise<LogRecord[]> {
  const params = new URLSearchParams();
  if (filters?.level) {
    params.set("level", filters.level);
  }
  if (filters?.agent) {
    params.set("agent", filters.agent);
  }
  if (filters?.task_id) {
    params.set("task_id", filters.task_id);
  }
  if (filters?.handoff_id) {
    params.set("handoff_id", filters.handoff_id);
  }
  const query = params.toString();
  return requestJson<LogRecord[]>(`/api/logs${query ? `?${query}` : ""}`);
}

export function getLog(logId: string): Promise<LogRecord> {
  return requestJson<LogRecord>(`/api/logs/${logId}`);
}

export function listAdapters(): Promise<AdapterRecord[]> {
  return requestJson<AdapterRecord[]>("/api/adapters");
}

export function execCodex(payload: { prompt: string; workdir?: string; timeout?: number }): Promise<CodexExecResult> {
  return requestJson<CodexExecResult>("/api/adapters/codex/exec", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSettings(): Promise<SettingsRecord> {
  return requestJson<SettingsRecord>("/api/settings");
}

export function updateSettings(payload: Partial<SettingsRecord>): Promise<SettingsRecord> {
  return requestJson<SettingsRecord>("/api/settings", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}
