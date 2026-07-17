// Operator-authored connectors → the M5 operational workspace (epic #37).
//
// The sidecar wires Boardstate's shipped M5 machinery (`@boardstate/broker` MCP-client
// connectors, §17v2 tool grants, the pending-action engine) onto its single in-process
// host, so a Hermes operator can author connectors → approve per-tool grants → the agent
// READS live external data and ACTS through approved tools, every consequential action
// operator-confirmed.
//
// LOAD-BEARING INVARIANT (§18 / epic invariant #8): a connector exists ONLY because the
// operator named it in `boardstate.connectors.json` — a file in the STATE DIR that the
// operator writes and controls. Its command / url / env can NEVER originate from the
// agent-writable workspace doc. This module reads the config from a FIXED filesystem path
// (`store.stateDir/boardstate.connectors.json`), never from the board document, so a
// connector name that appears in a doc/prompt/model output is inert.
//
// Absent config → this returns `null` and the sidecar wires no broker: byte-identical to
// the pre-M5 board (the action/connector RPCs simply are not registered, as before).

import { chmod, readFile } from "node:fs/promises";
import path from "node:path";
import { McpBroker, parseConnectorsConfig } from "@boardstate/broker";
import type { DashboardStore } from "@boardstate/core";
import {
  installConnectorWorkspace,
  type ConnectorWorkspaceHandle,
  type InProcessHost,
} from "@boardstate/server/node";

/** The operator-authored connectors config, resolved from the STATE DIR (never the doc). */
export const CONNECTORS_CONFIG_FILE = "boardstate.connectors.json";

export type ConnectorWorkspace = {
  /** The installed M5 workspace handle (engine + agent-tool adapter + tool_search backing). */
  workspace: ConnectorWorkspaceHandle;
  /** The MCP-client broker driving the operator's connectors. */
  broker: McpBroker;
  /** Absolute path of the config file that authored these connectors. */
  configPath: string;
  /** The server-side connector config values (command/args/url) that must NEVER surface in
   *  an agent-facing / MCP error payload (a spawn ENOENT or fetch error echoes the literal
   *  command/url — invariant #3). The MCP endpoint redacts these from every error it returns. */
  sensitiveStrings: string[];
};

/** Collect EVERY connector config string an error could leak — command/url/args AND the
 *  secret-bearing fields: env keys + env-ref names + their RESOLVED process-env VALUES (API
 *  keys, the most sensitive field — SEC-2), and header keys + values (incl. `${ENV}`-resolved).
 *  Length-agnostic: a short API key must never leak, so no minimum-length filter here (the
 *  redactor sorts DESC by length to mask overlapping secrets — SEC-3). */
export function collectSensitiveStrings(
  config: { connectors: ConnectorConfigLike[] },
  env: Record<string, string | undefined> = process.env,
): string[] {
  const out = new Set<string>();
  const add = (value: unknown): void => {
    if (typeof value === "string" && value.length > 0) {
      out.add(value);
    }
  };
  for (const c of config.connectors) {
    add(c.command);
    add(c.url);
    for (const arg of c.args ?? []) add(arg);
    // env: `{ CHILD_VAR: SOURCE_ENV_NAME }`. Redact the child var name, the source ref name,
    // AND the resolved secret value forwarded to the child (the actual API key).
    for (const [childVar, ref] of Object.entries(c.env ?? {})) {
      add(childVar);
      add(ref);
      add(env[ref]); // the resolved secret — never let it echo to the agent
    }
    // headers: key + literal value + any `${ENV}`-interpolated secret value.
    for (const [headerKey, raw] of Object.entries(c.headers ?? {})) {
      add(headerKey);
      add(raw);
      for (const m of raw.matchAll(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g)) add(env[m[1]]);
    }
  }
  return [...out];
}

type ConnectorConfigLike = {
  command?: string;
  url?: string;
  args?: string[];
  env?: Record<string, string>;
  headers?: Record<string, string>;
};

/**
 * Load `boardstate.connectors.json` from the state dir and wire the whole M5 connector
 * stack onto `host` (broker → grant lifecycle + pending-action engine → agent-tool adapter
 * → `boardstate_tool_search` backing) in the correct order. Returns the handles the caller
 * threads into `registerBoardstateRpc({ capabilityToolsHash })` and the MCP tool set
 * (`toolSearch` + `host.tools()`), or `null` when the config is absent (no connectors).
 *
 * Throws `BrokerConfigError` on a present-but-malformed config — a fail-closed signal the
 * operator sees at spawn, never a silently-ignored connector.
 */
export async function installConnectorsFromConfig(
  host: InProcessHost,
  store: DashboardStore,
  options: { mutationTimeoutMs?: number } = {},
): Promise<ConnectorWorkspace | null> {
  // The AUTHORSHIP boundary: a fixed path in the state dir, resolved here and nowhere
  // near the workspace doc (invariant #8). `store.stateDir` is the sidecar's own fs root.
  const configPath = path.join(store.stateDir, CONNECTORS_CONFIG_FILE);

  let text: string;
  try {
    text = await readFile(configPath, "utf8");
  } catch {
    return null; // absent → no connectors, zero behavior change
  }

  // Defensive hygiene: the config holds env-var REFERENCES, not secrets, but it is the
  // authorship boundary — keep it owner-only. Best-effort; a failure never blocks boot.
  await chmod(configPath, 0o600).catch(() => undefined);

  // parseConnectorsConfig is the ONE validator the broker trusts (unknown fields, bad
  // transports, non-env-ref `env` values all rejected). A parse throw propagates.
  const config = parseConnectorsConfig(JSON.parse(text) as unknown);
  if (config.connectors.length === 0) {
    return null;
  }

  const broker = new McpBroker(config);
  const workspace = installConnectorWorkspace(host, {
    broker,
    store,
    ...(options.mutationTimeoutMs !== undefined
      ? { mutationTimeoutMs: options.mutationTimeoutMs }
      : {}),
  });

  // `dashboard.workspace.replace` (templates, imports, agent workspace rewrites) carries
  // no `capabilitiesRegistry`, so a replace silently WIPES every configured connector's
  // grant until the next sidecar boot — the approvals widget goes blank and granted tools
  // die mid-session. Re-register on change: when a mutation leaves a configured connector
  // missing from the registry, `workspace.refresh()` re-pends it (status "requested" — a
  // replace can never resurrect a GRANT, only the request; the operator re-approves).
  // Debounced; self-terminating (the refresh's own mutation leaves nothing missing).
  let regTimer: ReturnType<typeof setTimeout> | null = null;
  host.addEventListener("boardstate.changed", () => {
    if (regTimer) return;
    regTimer = setTimeout(() => {
      regTimer = null;
      void (async () => {
        try {
          const doc = await store.read();
          const registry = doc.capabilitiesRegistry ?? {};
          if (broker.connectorNames().some((name) => !registry[name])) {
            await workspace.refresh();
            console.log("[boardstate] connector grants re-registered after a workspace replace");
          }
        } catch {
          /* transient read/refresh failure — the next change retries */
        }
      })();
    }, 300);
  });

  return { workspace, broker, configPath, sensitiveStrings: collectSensitiveStrings(config) };
}
