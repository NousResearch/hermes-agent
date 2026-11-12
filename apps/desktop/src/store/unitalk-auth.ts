import { atom } from 'nanostores'

import { startLocalhostCallbackServer } from '@/store/unitalk-auth-localhost'

const CLERK_TOKEN_KEY = 'unitalk-clerk-token'
const CLERK_PUBLISHABLE_KEY = 'pk_live_Y2xlcmsudW5pdGFsay5haSQ'

const CLERK_SIGN_IN_BASE = 'https://unitalk.ai/home/inscription'

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
 * Open the Unitalk Cloud sign-in page in the system browser and
 * wait for the Clerk OAuth callback.
 *
 * Strategy (tried in order):
 * 1. Localhost callback server — most reliable, works on all platforms
 * 2. Custom protocol `unitalk://callback` — fallback when port binding fails
 *
 * The localhost server binds on a random port and listens for
 * GET /callback?token=X. Clerk redirects there, the server captures
 * the token, stores it, and closes. The browser shows a completion page.
 *
 * Falls back to the Electron deep-link handler if the server cannot start.
 */
export async function startClerkAuth(): Promise<void> {
  try {
    // Primary path: localhost callback server
    const server = startLocalhostCallbackServer()

    // The url getter yields the address once listening. Use a small
    // polling loop to wait for it, or fall through to the fallback.
    let callbackUrl = server.url
    if (!callbackUrl) {
      // Wait up to 50ms for the server to be listening
      for (let i = 0; i < 10 && !callbackUrl; i++) {
        await new Promise(r => setTimeout(r, 5))
        callbackUrl = server.url
      }
    }

    if (!callbackUrl) {
      server.close()
      throw new Error('Could not bind localhost callback server')
    }

    const redirectUri = encodeURIComponent(callbackUrl)

    const signInUrl = `${CLERK_SIGN_IN_BASE}?redirect_uri=${redirectUri}`

    if (window.hermesDesktop?.openExternal) {
      await window.hermesDesktop.openExternal(signInUrl)
    } else {
      window.open(signInUrl, '_blank', 'noopener,noreferrer')
    }

    // Wait for the token to arrive via the localhost callback
    const token = await server.promise
    if (token && String(token).trim()) {
      storeClerkToken(String(token).trim())
    }
  } catch {
    // Fallback path: custom protocol unitalk://callback
    const signInUrl = 'https://unitalk.ai/home/inscription?redirect=unitalk://callback'

    if (window.hermesDesktop?.openExternal) {
      await window.hermesDesktop.openExternal(signInUrl)
    } else {
      window.open(signInUrl, '_blank', 'noopener,noreferrer')
    }
  }
}

/**
 * Retrieve the stored Clerk session token, or null when none is present.
 */
export function getClerkToken(): null | string {
  return readClerkToken()
}

/**
 * The Clerk publishable key used by the frontend API.
 * Configure via environment variable or the build system.
 */
export function getClerkPublishableKey(): string {
  return CLERK_PUBLISHABLE_KEY
}

/**
 * Persist a Clerk session token and update derived atoms.
 * Call this from the IPC listener wired to
 * Electron's `unitalk-oauth-callback` channel, or from the
 * localhost callback server.
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
 * Start listening for the Electron main-process deep-link callback
 * (fallback path when the localhost server cannot start).
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