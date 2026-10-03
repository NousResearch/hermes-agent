import { type ErrorSurface } from '@/lib/error-surface'

import { delay, type GatewayRequest } from './utils'

/**
 * #123079: `prompt.submit` is deliberately unbounded (PROMPT_SUBMIT_REQUEST_
 * TIMEOUT_MS = 30 min — turn completion is signaled by events, not the RPC
 * return). When the WS transport invalidates mid-request the channel rejects
 * EVERY pending call with the socket-death error (`json-rpc-channel.ts`
 * detach → dropSocket), so the submit pipeline used to settle the LIVE turn
 * with a definitive error card while the backend happily kept running it.
 *
 * Only errors that prove the RPC was in flight when the transport died are
 * transport losses. The pre-send refusal 'Hermes gateway is not connected'
 * (GATEWAY_NOT_CONNECTED_MESSAGE) fires BEFORE anything reaches the backend —
 * no turn exists server-side, so it must NOT gate on this path.
 */
const TRANSPORT_LOSS_ERROR_PATTERNS = [/gateway connection closed/i, /heartbeat acknowledgement timed out/i]

export function isTransportLossError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return TRANSPORT_LOSS_ERROR_PATTERNS.some(pattern => pattern.test(message))
}

/**
 * What the backend said about the turn a transport-lost submit may have
 * started:
 * - 'working' — the session's turn is live server-side; keep the composer busy
 *   and let the 45s silence watchdog own the eventual settle.
 * - 'settled-elsewhere' — the backend answered and the turn is over (or never
 *   started); no error card, the durable transcript is the truth.
 * - 'unreachable' — the gateway never answered the probe; fall through to the
 *   error card, tagged as a retryable streaming-layer drop.
 */
export type TransportLostTurnVerdict = 'settled-elsewhere' | 'unreachable' | 'working'

/** Bounded wait for the backend to answer while the reconnect ladder plays
 * out; each poll while the socket is down rejects fast ('not connected'), so
 * the loop merely waits out the existing reconnect policy — it never adds one. */
export const TRANSPORT_LOST_PROBE_WINDOW_MS = 20_000
export const TRANSPORT_LOST_PROBE_INTERVAL_MS = 1_000

/** Mirrors SILENT_TURN_RETRY (store/session-states.ts): the card renders the
 * streaming layer copy — 'The connection dropped before the reply finished.
 * Retry…' — instead of generic 'Prompt failed', and offers Retry. */
export const TRANSPORT_CLOSED_ERROR_SURFACE: ErrorSurface = {
  code: 'transport_closed',
  layer: 'streaming',
  retryable: true
}

interface ActiveListSessionRow {
  id?: string
  session_key?: string
  status?: string
}

interface ActiveListResponse {
  sessions?: ActiveListSessionRow[]
}

/**
 * Ask the gateway whether the turn a transport-lost submit started is still
 * running, via the authoritative in-memory `session.active_list` snapshot
 * (same probe shape the reconnect backstop uses). The row is matched by
 * runtime id OR stored session key — a resume can have rebound the runtime id
 * while the turn is the same conversation.
 *
 * Routing caveat: `session.active_list` has no session_id param, so the probe
 * rides the ambient (active gateway) dispatcher. For a foreground session
 * that is the owning backend and an answered list without the row is
 * conclusive ('settled-elsewhere'). A secondary-owned session the ambient
 * gateway never lists must NOT be read as settled — fail toward the live-ish
 * path ('unreachable') and let the silence watchdog settle it if the backend
 * is really gone.
 */
export async function reconcileTransportLostTurn(options: {
  foreground: boolean
  requestGateway: GatewayRequest
  runtimeSessionId: string
  storedSessionId: null | string
}): Promise<TransportLostTurnVerdict> {
  const deadline = Date.now() + TRANSPORT_LOST_PROBE_WINDOW_MS

  // Rejected polls are the expected state while the socket is down; only a
  // window of nothing-but-rejections means 'unreachable'.
  for (;;) {
    try {
      const response = await options.requestGateway<ActiveListResponse>('session.active_list', {})

      const row = (response.sessions ?? []).find(
        session =>
          (options.runtimeSessionId && session.id === options.runtimeSessionId) ||
          (options.storedSessionId && session.session_key === options.storedSessionId)
      )

      // A listed live-ish status means the turn is alive server-side. 'waiting'
      // is a live turn paused on the user; 'starting' is the turn spinning up.
      if (row && (row.status === 'starting' || row.status === 'waiting' || row.status === 'working')) {
        return 'working'
      }

      // The gateway answered and owns this session: idle (or an unrecognized
      // status) means the turn is over — or never began. Only an ABSENT row on
      // a non-foreground (potentially secondary-owned) session is
      // inconclusive.
      return row || options.foreground ? 'settled-elsewhere' : 'unreachable'
    } catch {
      const remaining = deadline - Date.now()

      if (remaining <= 0) {
        return 'unreachable'
      }

      await delay(Math.min(TRANSPORT_LOST_PROBE_INTERVAL_MS, remaining))
    }
  }
}
