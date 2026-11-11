import { atom } from 'nanostores'

const CLERK_TOKEN_KEY = 'unitalk-clerk-token'

/**
 * Whether a Clerk SSO token is stored (persisted across launches).
 * Does NOT validate expiry — the onboarding store and gateway
 * credential verification do that at use time.
 */
export const $isAuthenticated = atom<boolean>(hasClerkToken())

/**
 * Email extracted from the stored Clerk JWT, or null when no token
 * is present / the JWT is unparseable (not an auth verification —
 * the payload is decoded only for display).
 */
export const $userEmail = atom<null | string>(readEmailFromToken())

function hasClerkToken(): boolean {
  try {
    return typeof window !== 'undefined' && window.localStorage.getItem(CLERK_TOKEN_KEY) !== null
  } catch {
    return false
  }
}

function readClerkToken(): null | string {
  try {
    return typeof window !== 'undefined' ? window.localStorage.getItem(CLERK_TOKEN_KEY) : null
  } catch {
    return null
  }
}

function readEmailFromToken(): null | string {
  const token = readClerkToken()
  if (!token) {
    return null
  }

  try {
    // A Clerk session JWT is a standard three-segment base64url JWT.
    // The payload sits in segment index 1.
    const segments = token.split('.')
    if (segments.length !== 3) {
      return null
    }

    const payload = JSON.parse(atob(segments[1]))
    return typeof payload.email === 'string' ? payload.email : null
  } catch {
    return null
  }
}

/**
 * Open the Unitalk Cloud sign-in page in the system browser.
 * Clerk redirects the user back to unitalk://callback?token=…,
 * which the Electron main process delivers via the
 * `unitalk-oauth-callback` IPC event watched by startListening().
 */
export async function startClerkAuth(): Promise<void> {
  const signInUrl = 'https://unitalk.ai/home/inscription?redirect=unitalk://callback'

  if (window.hermesDesktop?.openExternal) {
    await window.hermesDesktop.openExternal(signInUrl)
  } else {
    window.open(signInUrl, '_blank', 'noopener,noreferrer')
  }
}

/**
 * Retrieve the stored Clerk session token, or null when none is present.
 */
export function getClerkToken(): null | string {
  return readClerkToken()
}

/**
 * Persist a Clerk session token and update derived atoms.
 * Call this from the IPC listener wired to
 * Electron's `unitalk-oauth-callback` channel.
 */
export function storeClerkToken(token: string): void {
  try {
    window.localStorage.setItem(CLERK_TOKEN_KEY, token)
  } catch {
    // localStorage unavailable — auth is session-only.
  }

  $isAuthenticated.set(true)
  $userEmail.set(readEmailFromToken())
}

/** Clear stored token and reset atoms (sign-out). */
export function clearClerkToken(): void {
  try {
    window.localStorage.removeItem(CLERK_TOKEN_KEY)
  } catch {
    // localStorage unavailable — nothing to clear.
  }

  $isAuthenticated.set(false)
  $userEmail.set(null)
}

/**
 * Start listening for the Electron main-process deep-link callback.
 * Returns an unsubscribe function. Safe to call multiple times —
 * each call starts a new listener; the caller should clean up the
 * previous one first.
 */
export function startListeningForCallback(): () => void {
  return window.hermesDesktop?.onUnitalkOauthCallback?.(token => {
    if (token && String(token).trim()) {
      storeClerkToken(String(token).trim())
    }
  }) ?? (() => undefined)
}