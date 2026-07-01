export type GatewayStatus = {
  running: boolean;
  state: string;
  busy: boolean;
  drainable: boolean;
  active_agents: number;
  restart_requested: boolean;
};

export type MiniAppRuntimeStatus = {
  mode: string;
  actions_enabled: boolean;
  public_exposure: boolean;
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
};

const API_URL = (import.meta.env.VITE_HERMES_MINIAPP_API_URL ?? "").replace(/\/$/, "");

export function hasMiniAppApi(): boolean {
  return API_URL.length > 0;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(`Mini App API request failed with ${response.status}`);
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
