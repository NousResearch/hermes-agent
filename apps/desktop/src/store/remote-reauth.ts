import { map } from 'nanostores'

/**
 * Latched "automatic sign-in exhausted" state, pushed by the Electron main
 * process (`hermes:oauth:login-exhausted` / `hermes:oauth:login-restored`).
 *
 * Electron is authoritative for login-window outcomes; this store is the
 * renderer's cache of that truth. Latches are keyed by ORIGIN (the scope one
 * cookie jar serves) and are intentionally kept even for gateways that are not
 * the active connection — a banner must never render over the workspace for a
 * background saved row (its auth failures belong on the source row), so the
 * SURFACING decision filters on the active connection; the latch itself just
 * records that the gateway's automatic sign-in is dead.
 */

export interface RemoteReauthExhaustion {
  latchedAt: number
}

export const $remoteReauthExhausted = map<Record<string, RemoteReauthExhaustion | undefined>>({})

/** The scope a login flight serves: the URL's origin. */
export function remoteReauthOrigin(baseUrl: string): string {
  try {
    return new URL(baseUrl).origin
  } catch {
    // an unparsable URL still keys consistently
    return baseUrl
  }
}

export function latchRemoteReauthExhausted(baseUrl: string, latchedAt = Date.now()): void {
  $remoteReauthExhausted.setKey(remoteReauthOrigin(baseUrl), { latchedAt })
}

export function clearRemoteReauthExhausted(baseUrl: string): void {
  $remoteReauthExhausted.setKey(remoteReauthOrigin(baseUrl), undefined)
}

/** The exhaustion episode for a gateway is only a verdict about THAT origin. */
export function isRemoteReauthExhaustedFor(baseUrl: string): boolean {
  return $remoteReauthExhausted.get()[remoteReauthOrigin(baseUrl)] !== undefined
}
