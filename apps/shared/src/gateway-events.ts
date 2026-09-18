/**
 * Wire types for the `tui_gateway` JSON-RPC surface shared by the Ink TUI, Desktop and the
 * web dashboard.
 *
 * Every shape here is GENERATED from `tui_gateway/contracts` (Python is the single source):
 * `./gateway-contract.generated.ts` carries `RpcMethods` (client→server method → params/result),
 * `ServerRequestMap` (server→client request → params/result), `GatewayEventMap` (notification
 * type → payload) and every value shape. `scripts/gen_gateway_contracts.py` regenerates it and
 * `tests/tui_gateway/contracts/test_generated.py` fails when the committed file is stale, so a field the
 * backend stops sending fails `tsc` here instead of drifting.
 *
 * This module adds only what the wire does not carry: the client-local synthetic events the TUI
 * transport publishes into the same handler stream, and the `GatewayEvent` envelope.
 */
import type { BackendGatewayEventMap } from './gateway-contract.generated.js'

export * from './gateway-contract.generated.js'

/** Owner-stamped pending prompt on a canonical gateway. */
export interface CanonicalPromptEvent {
  execution_generation?: number
  kind?: 'approval' | 'clarify'
  prompt_id: string
  /** Set by the Desktop canonical protocol layer for the named listeners. */
  request_id?: string
}

/**
 * Client-local synthetic events. Never emitted by `tui_gateway`; the Ink TUI's `gatewayClient`
 * publishes them into the same handler stream to report transport state.
 */
export interface ClientLocalGatewayEventMap {
  /** Canonical gateways (`gateway/session_pending_controls.py`) fan pending prompts out as
   *  events stamped with the owner's `execution_generation`; Desktop's canonical protocol
   *  layer projects them onto the named prompt listeners and answers over `approval.respond`
   *  / `clarify.respond`. Not part of the generated `tui_gateway` contract. */
  'approval.request': CanonicalPromptEvent & { command?: string; description?: string; choices?: string[]; edit?: unknown }
  'approval.settled': CanonicalPromptEvent
  'clarify.request': CanonicalPromptEvent & { question?: string; choices?: string[]; multi_select?: boolean }
  'clarify.settled': CanonicalPromptEvent
  'dashboard.new_session_requested': { reason?: string }
  'gateway.protocol_error': { preview?: string }
  'gateway.reconnecting': { attempt?: number; delay_ms?: number }
  'gateway.start_timeout': { cwd?: string; python?: string; stderr_tail?: string }
  'gateway.stderr': { line: string }
  /** Synthetic on the client (a reconnect replay could not cover the gap: epoch changed /
   * ring truncated) AND emitted by canonical gateways on fanout overflow
   * (`gateway/session_events.py`); consumers re-resume the session for a snapshot. */
  'session.replay_gap': { latest_seq?: number; replay_epoch?: string }
}

export interface GatewayEventMap extends BackendGatewayEventMap, ClientLocalGatewayEventMap {}

export type GatewayEventName = keyof GatewayEventMap

/** One `event` notification's `params`. */
export interface GatewayEvent<K extends GatewayEventName = GatewayEventName> {
  /** Registry connection whose socket delivered the event (renderer-side tag;
   * absent for the local/legacy primary path). */
  connectionId?: string
  /** Owner execution stamp on canonical gateways: the integer runtime epoch and the claimed
   * generation, spread onto the params beside `type`/`payload` (`gateway/session_events.py`). */
  authority_epoch?: number
  execution_generation?: number
  /** Session-scoped replay generation on canonical gateways. */
  replay_epoch?: string
  payload?: GatewayEventMap[K]
  /** Renderer-side source tag added by the Desktop gateway registry. */
  profile?: string
  /** Per-session monotonic counter stamped by `tui_gateway/event_replay.py::_stamp_event`;
   *  absent on session-less broadcasts. */
  seq?: number
  session_id?: string
  type: K
}

/** Backend-emitted notification names (generated `GATEWAY_EVENT_TYPES`), re-exported under the
 *  name the consumers already use. */
export { GATEWAY_EVENT_TYPES as BACKEND_EVENT_NAMES } from './gateway-contract.generated.js'
