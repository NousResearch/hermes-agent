// Tiny typed REST client. No deps; uses native fetch.

import type {
  ActivityEvent,
  CapacityReport,
  Department,
  Employee,
  Preset,
  ResolvedRole,
  SkillInfo,
  Task,
  ToolsetInfo,
} from "./types";

import { officeUrl } from "./publicPath";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(officeUrl(url), {
    headers: { "content-type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!r.ok) {
    let body: unknown = null;
    try {
      body = await r.json();
    } catch {
      body = await r.text();
    }
    throw new ApiError(r.status, r.statusText, body);
  }
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

export class ApiError extends Error {
  constructor(public status: number, public statusText: string, public body: unknown) {
    super(`${status} ${statusText}`);
  }
}

export const api = {
  health: () => jsonFetch<{ ok: boolean; version: string; office_root: string; runtime_default: string }>("/api/health"),
  capacity: (model?: string) => jsonFetch<CapacityReport>(`/api/capacity${model ? `?model=${encodeURIComponent(model)}` : ""}`),

  presets: () => jsonFetch<Preset[]>("/api/presets"),
  toolsets: () => jsonFetch<ToolsetInfo[]>("/api/toolsets"),
  skills: () => jsonFetch<SkillInfo[]>("/api/skills"),
  resolveRole: (text: string) => jsonFetch<ResolvedRole>("/api/skills/resolve", { method: "POST", body: JSON.stringify({ text }) }),

  listDepartments: () => jsonFetch<Department[]>("/api/departments"),
  createDepartment: (body: Partial<Department>) => jsonFetch<Department>("/api/departments", { method: "POST", body: JSON.stringify(body) }),
  patchDepartment: (id: string, body: Partial<Department>) => jsonFetch<Department>(`/api/departments/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteDepartment: (id: string) => jsonFetch<{ deleted_dept: string; deleted_employees: string[] }>(`/api/departments/${id}`, { method: "DELETE" }),

  listEmployees: (deptId?: string) => jsonFetch<Employee[]>(`/api/employees${deptId ? `?dept_id=${deptId}` : ""}`),
  getEmployee: (id: string) => jsonFetch<Employee & { cli_command: string }>(`/api/employees/${id}`),
  createEmployee: (body: Partial<Employee>) => jsonFetch<Employee>("/api/employees", { method: "POST", body: JSON.stringify(body) }),
  patchEmployee: (id: string, body: Partial<Employee>) => jsonFetch<Employee>(`/api/employees/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteEmployee: (id: string) => jsonFetch<{ deleted: string }>(`/api/employees/${id}`, { method: "DELETE" }),
  employeeActivity: (id: string, cursor?: number) => jsonFetch<{ events: ActivityEvent[]; next_cursor: number | null }>(`/api/employees/${id}/activity${cursor != null ? `?cursor=${cursor}` : ""}`),

  listTasks: () => jsonFetch<Task[]>("/api/tasks"),
  createTask: (body: { text: string; employee_id?: string; department_id?: string }) =>
    jsonFetch<Task>("/api/tasks", { method: "POST", body: JSON.stringify(body) }),
};
