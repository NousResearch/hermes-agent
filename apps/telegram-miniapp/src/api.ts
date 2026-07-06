export type GatewayStatus = {
  running: boolean;
  state: string;
  busy: boolean;
  drainable: boolean;
  active_agents: number;
  restart_requested: boolean;
};

export type ActionApplication = "not-wired" | "record-only" | "live";

export type MiniAppRuntimeStatus = {
  mode: string;
  actions_enabled: boolean;
  public_exposure: boolean;
  // Honest application state, separate from actions_enabled. Optional for
  // backward compatibility with older sidecars that omit them.
  gateway_resolver_active?: boolean;
  action_application?: ActionApplication;
};

export type StatusSnapshot = {
  ok: boolean;
  updated_at: string;
  hermes_home: "configured" | "missing" | "unknown";
  gateway: GatewayStatus;
  miniapp: MiniAppRuntimeStatus;
};

export type AuthenticatedUser = {
  id: string;
  username?: string;
  first_name?: string;
  last_name?: string;
};

export type AuthResponse = {
  ok: boolean;
  user: AuthenticatedUser;
  expires_at: string;
  is_action_owner?: boolean;
};

export type SnapshotMeta = {
  source: "preview" | "live-safe";
  source_label: string;
  redaction: "safe-preview";
  contains_live_actions: boolean;
};

export type CapabilityItem = {
  id: string;
  label: string;
  enabled: boolean;
  mode: "read-only" | "blocked" | "owner-confirmed-action";
  reason: string;
};

export type CapabilitiesSnapshot = {
  ok: boolean;
  meta?: SnapshotMeta;
  items: CapabilityItem[];
};

export type ApprovalRisk = "read_only" | "critical";

export type ApprovalStatus = "waiting" | "blocked";

export type ApprovalDecision = "approve_once" | "reject_once";

export type ApprovalItem = {
  id: string;
  title: string;
  source: string;
  risk: ApprovalRisk;
  summary: string;
  requested_at: string;
  status: ApprovalStatus;
  checks: string[];
  allowed_decisions?: ApprovalDecision[];
};

export type ApprovalsSnapshot = {
  ok: boolean;
  meta?: SnapshotMeta;
  snapshot_version?: string;
  items: ApprovalItem[];
};

export type DecisionResponse = {
  ok: boolean;
  decision_id: string;
  status: string;
  message: string;
};

export type SessionPreviewItem = {
  id: string;
  agent: string;
  state: "observing" | "waiting" | "completed";
  meta: string;
  time: string;
  tone: "ok" | "warn" | "muted";
};

export type SessionsSnapshot = {
  ok: boolean;
  meta?: SnapshotMeta;
  items: SessionPreviewItem[];
};

export type LogPreviewItem = {
  level: "info" | "warn" | "error";
  message: string;
  time: string;
};

export type LogsSnapshot = {
  ok: boolean;
  meta?: SnapshotMeta;
  items: LogPreviewItem[];
};

const API_URL = (import.meta.env.VITE_HERMES_MINIAPP_API_URL ?? "").replace(/\/$/, "");

// Runtime guardrail: the read-only surface is enumerated here, so no code path
// (present or future) can reach an action endpoint through requestJson.
// Backend route absence stays the real security boundary; this narrows the
// client. Kept in sync by test_telegram_miniapp_frontend_guardrails.py.
const ALLOWED_API_PATHS = new Set([
  "/api/auth/telegram",
  "/api/logout",
  "/api/me",
  "/api/status",
  "/api/capabilities",
  "/api/approvals",
  "/api/sessions",
  "/api/logs",
]);

// The single mutating action path is dynamic (opaque approval id) so it cannot
// be a static Set entry. It is allowed only via this strict pattern — the exact
// Phase 1 decision endpoint and nothing else. Backend owner/proof/capability
// gating stays the real security boundary; this narrows the client surface.
const DECISION_PATH = /^\/api\/approvals\/[A-Za-z0-9_-]+\/decision$/;

function isAllowedApiPath(path: string): boolean {
  return ALLOWED_API_PATHS.has(path) || DECISION_PATH.test(path);
}

export function hasMiniAppApi(isTelegramRuntime = false): boolean {
  return API_URL.length > 0 || (import.meta.env.PROD && isTelegramRuntime);
}

export class MiniAppApiError extends Error {
  status: number;

  constructor(status: number, message?: string) {
    super(message || `Mini App API request failed with ${status}`);
    this.name = "MiniAppApiError";
    this.status = status;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  if (!isAllowedApiPath(path)) {
    throw new Error(`Mini App API path is not allowlisted: ${path}`);
  }
  const response = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string; message?: string };
      detail = body.detail || body.message || "";
    } catch {
      detail = "";
    }
    throw new MiniAppApiError(response.status, detail || `Mini App API request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function authenticateTelegram(initData: string): Promise<AuthResponse> {
  return requestJson<AuthResponse>("/api/auth/telegram", {
    method: "POST",
    body: JSON.stringify({ initData }),
  });
}

export async function fetchStatusSnapshot(): Promise<StatusSnapshot> {
  return requestJson<StatusSnapshot>("/api/status");
}

export async function fetchCapabilitiesSnapshot(): Promise<CapabilitiesSnapshot> {
  return requestJson<CapabilitiesSnapshot>("/api/capabilities");
}

export async function fetchApprovalsSnapshot(): Promise<ApprovalsSnapshot> {
  return requestJson<ApprovalsSnapshot>("/api/approvals");
}

export async function fetchSessionsSnapshot(): Promise<SessionsSnapshot> {
  return requestJson<SessionsSnapshot>("/api/sessions");
}

export async function fetchLogsSnapshot(): Promise<LogsSnapshot> {
  return requestJson<LogsSnapshot>("/api/logs");
}

export type DecisionInput = {
  approvalId: string;
  decision: ApprovalDecision;
  clientRequestId: string;
  snapshotVersion: string;
  initData: string;
};

export async function postApprovalDecision(input: DecisionInput): Promise<DecisionResponse> {
  // The owner-confirmed Phase 1 action. The fresh Telegram proof rides in the
  // X-Telegram-Init-Data header (the backend re-verifies it); the session
  // cookie alone is not accepted server-side.
  return requestJson<DecisionResponse>(`/api/approvals/${encodeURIComponent(input.approvalId)}/decision`, {
    method: "POST",
    headers: { "x-telegram-init-data": input.initData },
    body: JSON.stringify({
      decision: input.decision,
      client_request_id: input.clientRequestId,
      snapshot_version: input.snapshotVersion,
    }),
  });
}
