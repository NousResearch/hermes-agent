import { isTimeoutError } from '@/lib/with-timeout'

// How reconnectSecondary (store/gateway.ts) classifies a failed secondary dial:
// a stall counts toward parking the entry, a missing connection or profile is
// permanent and disposes it, and anything else keeps backing off.

export function isStalledDialError(error: unknown): boolean {
  if (isTimeoutError(error)) {
    return true
  }

  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.includes('timed out while waiting for a free slot')
}

// Electron's getConnectionFor rejects with `No connection with id "…"` when
// the registry entry is gone. That is a permanent condition for the scoped
// socket, unlike transient transport errors.
export function isMissingConnectionError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.includes('No connection with id')
}

// Electron's spawn guard (assertLocalProfileCanStart) rejects with these when
// the profile's directory is gone or its DELETE is still in flight. For a
// renderer socket that condition is permanent: the backend it reconnects to
// can never come back, and every retry hammers the guard (#88769).
export function isMissingProfileError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.includes('no longer exists') || message.includes('is being deleted')
}
