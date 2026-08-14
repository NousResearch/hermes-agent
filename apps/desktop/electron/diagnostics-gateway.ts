// Main-process client for the gateway's diagnostics ring (U3, KTD2/KTD3).
//
// The gateway half of a hitch capture is pulled over the gateway's EXISTING
// authenticated channel — the same base URL + session token the rest of main
// already uses for `hermes:api` — so no new listener, port, or credential is
// introduced for diagnostics. Three methods, mapped onto REST paths:
//
//   diagnostics/arm      { capture_id, wall_clock_anchor_ms } -> { monotonic_anchor_ms }
//   diagnostics/disarm   { }                                  -> { }
//   diagnostics/collect  { capture_id }                       -> { capture_id, monotonic_anchor_ms, events, dropped }
//
// Everything degrades: a gateway that predates the endpoints (404), refuses the
// caller (401/403), or simply cannot be reached must not fail the capture — the
// exporter (U4) marks the gateway stream absent with a reason instead. That is
// why every method here resolves to a tagged result rather than throwing.
//
// The transport is injected so this module stays Electron-free and unit
// testable, mirroring gateway-ws-probe.ts / stream-throttle.ts.

/** The wire methods, spelled exactly as the gateway advertises them. */
export type GatewayDiagnosticsMethod = 'diagnostics/arm' | 'diagnostics/collect' | 'diagnostics/disarm'

/** Why a gateway stream is missing from a capture bundle. */
export type GatewayAbsentReason = 'disabled' | 'remote-gateway' | 'unauthenticated' | 'unavailable' | 'unsupported'

export interface GatewayDiagnosticsStream {
  captureId: string
  /** The gateway's own monotonic clock at arm time, for KTD3 alignment. */
  monotonicAnchorMs: number
  events: unknown[]
  dropped: number
}

// Tagged results rather than discriminated unions: the electron tsconfig runs
// with `strictNullChecks: false`, under which a boolean discriminant does not
// narrow, so the optional-field shape is the one callers can actually read.
export interface GatewayArmResult {
  ok: boolean
  /** Set when ok. */
  monotonicAnchorMs?: number
  /** Set when not ok. */
  reason?: GatewayAbsentReason
}

export interface GatewayCollectResult {
  ok: boolean
  /** Set when ok. */
  stream?: GatewayDiagnosticsStream
  /** Set when not ok. */
  reason?: GatewayAbsentReason
}

/** Issues one authenticated request against the live gateway. */
export type GatewayDiagnosticsRequest = (
  method: GatewayDiagnosticsMethod,
  params: Record<string, unknown>
) => Promise<unknown>

export interface GatewayDiagnosticsClient {
  arm(input: { captureId: string; wallClockAnchorMs: number }): Promise<GatewayArmResult>
  collect(captureId: string): Promise<GatewayCollectResult>
  /** Best-effort; a gateway that never armed is happy to be told to disarm. */
  disarm(): Promise<void>
}

/** REST path a method rides. Kept next to the method names so the two can
 * never drift apart in opposite directions. */
export function gatewayDiagnosticsPath(method: GatewayDiagnosticsMethod): string {
  return `/api/${method}`
}

/**
 * Map a transport failure onto an absent-stream reason. `fetchJson` rejects
 * HTTP errors as `Error('<status>: <body>')`, so the status is recoverable from
 * the message without threading a richer error type through main.
 */
export function classifyGatewayFailure(error: unknown): GatewayAbsentReason {
  const message = error instanceof Error ? error.message : String(error ?? '')
  const status = /^\s*(\d{3})\b/.exec(message)?.[1]

  if (status === '404' || status === '405' || status === '501') {
    return 'unsupported'
  }

  if (status === '401' || status === '403') {
    return 'unauthenticated'
  }

  return 'unavailable'
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === 'number' ? value : Number(value)

  return Number.isFinite(parsed) ? parsed : null
}

export function createGatewayDiagnosticsClient(request: GatewayDiagnosticsRequest): GatewayDiagnosticsClient {
  return {
    async arm({ captureId, wallClockAnchorMs }) {
      let answer: unknown

      try {
        answer = await request('diagnostics/arm', {
          capture_id: captureId,
          wall_clock_anchor_ms: wallClockAnchorMs
        })
      } catch (error) {
        return { ok: false, reason: classifyGatewayFailure(error) }
      }

      // A gateway that answers 200 with a body this client cannot anchor on is
      // no more usable than one that 404s: without the monotonic anchor the
      // stream cannot be aligned (KTD3), so treat it as unsupported.
      const monotonicAnchorMs = finiteNumber(asRecord(answer)?.monotonic_anchor_ms)

      if (monotonicAnchorMs === null) {
        return { ok: false, reason: 'unsupported' }
      }

      return { ok: true, monotonicAnchorMs }
    },

    async collect(captureId) {
      let answer: unknown

      try {
        answer = await request('diagnostics/collect', { capture_id: captureId })
      } catch (error) {
        return { ok: false, reason: classifyGatewayFailure(error) }
      }

      const body = asRecord(answer)
      const monotonicAnchorMs = finiteNumber(body?.monotonic_anchor_ms)

      if (!body || monotonicAnchorMs === null) {
        return { ok: false, reason: 'unsupported' }
      }

      return {
        ok: true,
        stream: {
          captureId: typeof body.capture_id === 'string' ? body.capture_id : captureId,
          monotonicAnchorMs,
          events: Array.isArray(body.events) ? body.events : [],
          dropped: finiteNumber(body.dropped) ?? 0
        }
      }
    },

    async disarm() {
      try {
        await request('diagnostics/disarm', {})
      } catch {
        // Disarm is advisory: the gateway drops its ring on the next arm
        // anyway, and a capture must never fail on the way out.
      }
    }
  }
}
