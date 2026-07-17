// The operator transport (Stage 3 — the browser/desktop half of the DECIDED security seam).
//
// `<boardstate-view>` builds its own actions/approvals seams from the ONE `transport` it is
// given and dispatches every control-plane call through it — including the four operator
// verbs when `operator` is true (grant/widget approve, action confirm/deny). But those verbs
// are BLOCKED on the browser WS and excluded from the MCP proxy (invariant #5), so they can
// never travel the live socket. This wrapper is the whole seam: it intercepts exactly those
// four verbs and routes them to the plugin_api operator endpoint (the one privileged path,
// which adds the dashboard-session auth + the operators allowlist before forwarding to the
// sidecar's nonce-gated /operator), and delegates everything else to the live WS transport.
//
// Setting `view.operator = true` then lets the view render the operator affordances enabled;
// this transport makes those affordances actually reach the operator plane. The server still
// independently enforces the verbs as operator-only — this is presentation + routing, not a
// new privilege.

import type { Transport } from "@boardstate/core"; // type-only: erased at build

/** The exact operator verb set — mirrors the sidecar's OPERATOR_ONLY_METHODS. These four are
 *  the ONLY methods this transport diverts to the plugin_api operator endpoint. */
export const OPERATOR_METHODS: ReadonlySet<string> = new Set([
  "dashboard.widget.approve",
  "dashboard.capability.approve",
  "dashboard.action.confirm",
  "dashboard.action.deny",
]);

/** Posts `{ method, params }` to the plugin_api operator endpoint and resolves the sidecar's
 *  raw RPC result (the same value `transport.request` would have returned). Rejects on a
 *  non-operator response (auth/allowlist 401/403, or a sidecar refusal). */
export type OperatorSend = (method: string, params: unknown) => Promise<unknown>;

/** The lifecycle members index.tsx / plugin.tsx read on the WS transport, beyond `Transport`. */
type LifecycleTransport = Transport & { readonly ready: Promise<void>; readonly closed?: boolean; close(): void };

/**
 * Wrap a live WS transport so the four operator verbs route to `sendOperator` (→ plugin_api
 * → sidecar /operator) while every other request rides the socket unchanged. Lifecycle
 * (`ready`/`closed`/`close`) and event subscription delegate straight through.
 */
export function withOperatorGate<T extends LifecycleTransport>(base: T, sendOperator: OperatorSend): T {
  const gated = {
    request(method: string, params?: unknown, ctx?: unknown): Promise<unknown> {
      if (OPERATOR_METHODS.has(method)) {
        return sendOperator(method, params ?? {});
      }
      return base.request(method, params, ctx);
    },
    addEventListener(event: string, fn: (payload: unknown) => void): () => void {
      return base.addEventListener(event, fn);
    },
    close(): void {
      base.close();
    },
    get ready(): Promise<void> {
      return base.ready;
    },
    get closed(): boolean | undefined {
      return base.closed;
    },
  };
  return gated as unknown as T;
}
