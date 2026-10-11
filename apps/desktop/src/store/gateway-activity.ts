import type { GatewayEvent } from '@hermes/shared'

export const TURN_LEASE_SETTLE_DELAY_MS = 500

export function isTurnSettlement(event: GatewayEvent): boolean {
  const payload = event.payload as Record<string, unknown> | undefined

  return event.type === 'subagent.complete' || (event.type === 'session.info' && payload?.running === false)
}

export function scopeHasTurnLease(scope: string, leases: ReadonlyMap<string, unknown>): boolean {
  const prefix = `${scope}\u0000`

  for (const key of leases.keys()) {
    if (key.startsWith(prefix)) {
      return true
    }
  }

  return false
}

// A cooperative-retirement hint, never proof that stopping a backend is safe:
// main still asks the backend itself before stopping anything (#104871).
export function publishTurnLease(scope: string, activeTurn: boolean): void {
  void window.hermesDesktop?.touchBackend?.(scope, { activeTurn }).catch(() => undefined)
}
