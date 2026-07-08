// REST client for the management API (/api/*), mirroring the mechanism in
// web/src/lib/api.ts: session-token header from the server-injected global,
// optional reverse-proxy base path, credentials for the gated cookie path.
// ht-web reads the HT-prefixed token first, falling back to the legacy one so
// it works whether served by a rebranded or an older gateway build.

declare global {
  interface Window {
    __HT_SESSION_TOKEN__?: string;
    __HERMES_SESSION_TOKEN__?: string;
    __HT_BASE_PATH__?: string;
    __HERMES_BASE_PATH__?: string;
    __HT_AUTH_REQUIRED__?: boolean;
    __HERMES_AUTH_REQUIRED__?: boolean;
  }
}

const SESSION_HEADER = "X-Hermes-Session-Token";

function readBasePath(): string {
  if (typeof window === "undefined") return "";
  const raw = window.__HT_BASE_PATH__ ?? window.__HERMES_BASE_PATH__ ?? "";
  if (!raw) return "";
  const withLead = raw.startsWith("/") ? raw : `/${raw}`;
  return withLead.replace(/\/+$/, "");
}

export const BASE_PATH = readBasePath();

function sessionToken(): string | undefined {
  if (typeof window === "undefined") return undefined;
  return window.__HT_SESSION_TOKEN__ ?? window.__HERMES_SESSION_TOKEN__;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface RequestOptions extends RequestInit {
  /** Do not throw on 401; return the parsed body instead. */
  allowUnauthorized?: boolean;
}

async function parseError(res: Response): Promise<never> {
  let body: unknown;
  let message = `${res.status} ${res.statusText}`;
  try {
    body = await res.clone().json();
    const err = (body as { error?: string; message?: string }) ?? {};
    message = err.message || err.error || message;
  } catch {
    try {
      const text = await res.text();
      if (text) message = text;
    } catch {
      /* ignore */
    }
  }
  throw new ApiError(message, res.status, body);
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { allowUnauthorized, ...init } = options;
  const headers = new Headers(init.headers);
  const token = sessionToken();
  if (token && !headers.has(SESSION_HEADER)) {
    headers.set(SESSION_HEADER, token);
  }
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${BASE_PATH}${path}`, {
    ...init,
    headers,
    credentials: init.credentials ?? "include",
  });

  if (res.status === 401 && !allowUnauthorized) {
    // Gated mode emits { error, login_url } for a full-page re-auth redirect.
    try {
      const body = (await res.clone().json()) as { error?: string; login_url?: string };
      if (
        (body.error === "unauthenticated" || body.error === "session_expired") &&
        body.login_url
      ) {
        window.location.assign(body.login_url);
        return new Promise<T>(() => {});
      }
    } catch {
      /* non-JSON 401 — fall through to throw */
    }
    return parseError(res);
  }

  if (!res.ok) {
    return parseError(res);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  const contentType = res.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await res.json()) as T;
  }
  return (await res.text()) as unknown as T;
}

export const apiGet = <T>(path: string, options?: RequestOptions) =>
  apiFetch<T>(path, { ...options, method: "GET" });

export const apiPost = <T>(path: string, body?: unknown, options?: RequestOptions) =>
  apiFetch<T>(path, {
    ...options,
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const apiDelete = <T>(path: string, options?: RequestOptions) =>
  apiFetch<T>(path, { ...options, method: "DELETE" });
