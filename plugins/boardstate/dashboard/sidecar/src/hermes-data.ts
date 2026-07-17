// HermesRpcDeps — resolves Boardstate `source:"rpc"` data bindings against the live
// Hermes REST surface, so the board's builtins (usage / sessions / instances / cron)
// show real Hermes data instead of error cells.
//
// AUTH (the keystone seam): the Node sidecar CANNOT authenticate to Hermes off-loopback
// on its own, so `plugin_api` (in-process with the gateway) injects the dashboard base
// URL + a session credential into the sidecar env at spawn. This resolver uses that
// credential server-side only — it never enters the workspace doc or reaches a browser.
//
// A binding this resolver doesn't handle (file bindings, unknown methods) is delegated
// to the injected `fallback` (the node default from `nodeRpcDeps()`), so behavior is a
// pure superset of the stock sidecar.

import type { BindingResolver } from "@boardstate/server/node";

export type HermesDataConfig = {
  /** Dashboard base, e.g. http://127.0.0.1:9119 (no trailing slash needed). */
  baseUrl: string;
  /** Session token sent as X-Hermes-Session-Token (loopback/token mode). */
  sessionToken: string;
  /** The node default resolver — handles file bindings + anything not mapped here. */
  fallback: BindingResolver;
  /** Injected for tests; defaults to global fetch. */
  fetchImpl?: typeof fetch;
};

type RpcBinding = { source: "rpc"; method: string; params?: Record<string, unknown> };

function isRpcBinding(binding: unknown): binding is RpcBinding {
  return (
    typeof binding === "object" &&
    binding !== null &&
    (binding as { source?: unknown }).source === "rpc" &&
    typeof (binding as { method?: unknown }).method === "string"
  );
}

const num = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);
const str = (v: unknown): string => (typeof v === "string" ? v : "");
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const obj = (v: unknown): Record<string, unknown> =>
  typeof v === "object" && v !== null ? (v as Record<string, unknown>) : {};

/**
 * Each handler maps a Boardstate data-read method to a Hermes REST call + shapes the
 * response to what the corresponding builtin expects (SPEC data shapes). Best-effort
 * on the Hermes side: missing keys degrade to empty, never throw.
 */
type Handler = (
  get: (path: string) => Promise<unknown>,
  params: Record<string, unknown>,
) => Promise<unknown>;

const HANDLERS: Record<string, Handler> = {
  // usage widget: { totals: { totalCost, totalTokens }, days? }
  "usage.status": async (get) => {
    const u = obj(await get("/api/analytics/usage"));
    const totals = obj(u.totals ?? u);
    const totalCost = num(totals.total_estimated_cost ?? totals.estimated_cost ?? totals.totalCost);
    const totalTokens =
      num(totals.total_input ?? totals.input_tokens) + num(totals.total_output ?? totals.output_tokens);
    return { totals: { totalCost, totalTokens }, days: num(u.days) || undefined };
  },
  // stat-card (usd): a single number.
  "usage.cost": async (get) => {
    const u = obj(await get("/api/analytics/usage"));
    const totals = obj(u.totals ?? u);
    return num(totals.total_estimated_cost ?? totals.estimated_cost ?? totals.totalCost);
  },
  // sessions widget: rows { key, label, status, hasActiveRun, updatedAt }
  "sessions.list": async (get, params) => {
    const limit = num(params.limit) || 8;
    const raw = await get("/api/sessions");
    const list = Array.isArray(raw) ? raw : arr(obj(raw).sessions);
    return list.slice(0, limit).map((s) => {
      const row = obj(s);
      return {
        key: str(row.id ?? row.session_id ?? row.key),
        label: str(row.title ?? row.label ?? row.name) || str(row.id ?? row.session_id),
        status: str(row.status),
        hasActiveRun: Boolean(row.has_active_run ?? row.hasActiveRun ?? row.active),
        updatedAt: str(row.updated_at ?? row.updatedAt ?? row.last_activity),
      };
    });
  },
  // instances widget: { presence: [{ instanceId, platform, version, lastInputSeconds }] }
  "system-presence": async (get) => {
    const s = obj(await get("/api/status"));
    const agents = arr(s.active_agents ?? s.agents ?? s.presence);
    return {
      presence: agents.map((a) => {
        const row = obj(a);
        return {
          instanceId: str(row.instance_id ?? row.id ?? row.name),
          platform: str(row.platform ?? row.channel),
          version: str(row.version),
          lastInputSeconds: num(row.last_input_seconds ?? row.idle_seconds),
        };
      }),
    };
  },
  // cron widget: { jobs: [{ id, name, enabled, state: { nextRunAtMs, lastRunStatus } }] }
  "cron.list": async (get, params) => {
    const limit = num(params.limit) || 8;
    const raw = await get("/api/cron");
    const list = Array.isArray(raw) ? raw : arr(obj(raw).jobs ?? obj(raw).crons);
    return {
      jobs: list.slice(0, limit).map((j) => {
        const row = obj(j);
        const state = obj(row.state);
        return {
          id: str(row.id ?? row.name),
          name: str(row.name ?? row.id),
          enabled: Boolean(row.enabled ?? row.active),
          state: {
            nextRunAtMs: num(state.nextRunAtMs ?? row.next_run_at_ms ?? row.next_run_ms),
            lastRunStatus: str(state.lastRunStatus ?? row.last_run_status ?? row.last_status),
          },
        };
      }),
    };
  },
};

// `node.list` is an alias of the same presence data for the instances builtin.
HANDLERS["node.list"] = HANDLERS["system-presence"];

/** A Hermes REST getter bound to a base URL + session token (server-side only). */
function makeGet(baseUrl: string, sessionToken: string, fetchImpl: typeof fetch) {
  const base = baseUrl.replace(/\/+$/, "");
  return async (path: string): Promise<unknown> => {
    const res = await fetchImpl(`${base}${path}`, {
      headers: { "X-Hermes-Session-Token": sessionToken, Accept: "application/json" },
    });
    if (!res.ok) {
      throw new Error(`Hermes ${path} → ${res.status}`);
    }
    return res.json();
  };
}

export function createHermesRpcResolver(config: HermesDataConfig): BindingResolver {
  const get = makeGet(config.baseUrl, config.sessionToken, config.fetchImpl ?? fetch);
  return async (binding, options) => {
    if (isRpcBinding(binding)) {
      const handler = HANDLERS[binding.method];
      if (handler) {
        return handler(get, binding.params ?? {});
      }
    }
    // Not a Hermes-mapped rpc binding — delegate (file bindings, unmapped methods).
    return config.fallback(binding, options);
  };
}

/** Minimal shape of the host's RPC registration surface we depend on. */
type RpcHost = {
  registerRpc: (
    method: string,
    handler: (opts: { params?: unknown; respond: (ok: boolean, data: unknown) => void }) => unknown,
    options: { scope: "read" | "write" },
  ) => void;
};

/**
 * Register each Hermes data method as a read-scoped RPC handler on the host.
 *
 * THIS is the seam the browser actually uses. `<boardstate-view>` resolves a
 * `source:"rpc"` binding by calling `transport.request(binding.method, params)`
 * (see @boardstate/host resolveBinding) — it does NOT route rpc bindings through
 * `dashboard.data.read` (that serves file/static only and answers rpc with
 * `binding_client_resolved`). Without these handlers every data-bound widget
 * renders an error cell. Read scope keeps them callable over the networked
 * transport (operator-only/write methods are blocked there).
 */
export function registerHermesDataRpc(
  host: RpcHost,
  config: { baseUrl: string; sessionToken: string; fetchImpl?: typeof fetch },
): string[] {
  const get = makeGet(config.baseUrl, config.sessionToken, config.fetchImpl ?? fetch);
  const methods = Object.keys(HANDLERS);
  for (const method of methods) {
    host.registerRpc(
      method,
      async (opts) => {
        const data = await HANDLERS[method](get, obj(opts?.params));
        opts.respond(true, data);
      },
      { scope: "read" },
    );
  }
  return methods;
}
