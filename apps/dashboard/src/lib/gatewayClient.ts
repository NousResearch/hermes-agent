import {
  JsonRpcGatewayClient,
  type ConnectionState,
  type GatewayEvent,
  type GatewayEventName,
} from "@hermes/shared";

export type { ConnectionState, GatewayEvent, GatewayEventName };

/**
 * Browser wrapper for the shared tui_gateway JSON-RPC client.
 *
 * Dashboard resolves its token and host from the served page. Desktop uses the
 * same shared protocol client, but supplies an absolute wsUrl from Electron.
 */
export class GatewayClient extends JsonRpcGatewayClient {
  async connect(token?: string): Promise<void> {
    const resolved = token ?? window.__HERMES_SESSION_TOKEN__ ?? "";
    if (!resolved) {
      throw new Error(
        "Session token not available — page must be served by the metakina-agent dashboard",
      );
    }

    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    await super.connect(
      `${scheme}//${location.host}/api/ws?token=${encodeURIComponent(resolved)}`,
    );
  }
}

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string;
  }
}
