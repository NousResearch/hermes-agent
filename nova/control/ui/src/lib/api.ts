/* The Control API, and nothing else.
 *
 * The dashboard talks to `/platform/v1/*` and never to a runtime endpoint — the boundary
 * `docs/platform/ARCHITECTURE_BOUNDARIES.md` §5 exists to keep. Every panel loads
 * independently: one 403 (a viewer reading an admin route) must blank its own card and
 * nothing else, which is why there is no single `Promise.all` over the whole page.
 */
export const API = "/platform/v1";

export type Loaded<T> =
  | { state: "loading" }
  | { state: "ok"; data: T }
  | { state: "forbidden" }
  | { state: "error"; message: string };

export async function load<T>(path: string): Promise<Loaded<T>> {
  try {
    const response = await fetch(`${API}${path}`, { headers: { Accept: "application/json" } });
    if (response.status === 403) return { state: "forbidden" };
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        message = body?.error?.message ?? message;
      } catch {
        /* a non-JSON error body is still an error worth showing */
      }
      return { state: "error", message };
    }
    return { state: "ok", data: (await response.json()) as T };
  } catch (cause) {
    return { state: "error", message: cause instanceof Error ? cause.message : "network error" };
  }
}

export function plural(n: number, one: string, many = `${one}s`) {
  return `${n} ${n === 1 ? one : many}`;
}

/** A control-plane write.
 *
 *  Same-origin and JSON-only, because the server requires both: it refuses a
 *  cross-site Origin and a non-JSON content type before reading a byte. The UI is not
 *  the thing enforcing that — it just has to speak the protocol the server enforces.
 *
 *  Throws on failure with the server's own message, so a caller renders what actually
 *  went wrong instead of "something went wrong".
 */
export async function post<T = unknown>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    credentials: "same-origin",
    body: JSON.stringify(body ?? {}),
  });
  let payload: any = null;
  try {
    payload = await response.json();
  } catch {
    /* a non-JSON body is reported below by status alone */
  }
  if (!response.ok) {
    throw new Error(payload?.error?.message || payload?.detail || `HTTP ${response.status}`);
  }
  return payload as T;
}
