import { atom } from 'nanostores'

import type { DesktopAuthGateEvent, DesktopManagedStatus } from '@/global'

// The desktop's login gate is the ApexNodes managed-LLM account (Desktop V0.2,
// China-first): a signed-in user gets zero-key chat via the relay. This store is
// the single source of truth for "is the user logged in?" — the boot gate blocks
// the chat UI until `status === 'signed-in'`, and the continuous auth gate flips
// it back to 'signed-out' / 'disabled' when a backend call reports 401 / 403
// account_disabled. It REUSES the existing managed bridge
// (window.hermesDesktop.managed) — it does not add a second auth system.

export interface AuthAccount {
  email: string
  name: string
  plan: string
}

export type AuthStatus =
  // First status() call hasn't resolved yet — hold the gate (show nothing /
  // the boot overlay), never flash the login screen at a returning user.
  | 'checking'
  // Signed in (relay key on disk) — chat is unblocked.
  | 'signed-in'
  // Not signed in — show the login screen and block chat.
  | 'signed-out'
  // Account abnormal (403 account_disabled) — show the login screen with the
  // account-disabled message; the user must re-authenticate.
  | 'disabled'

export interface DesktopAuthState {
  account: AuthAccount
  /** True on builds where the managed-LLM default path is enabled. On a
   *  managed-disabled build (APEXNODES_MANAGED=0) the account gate is a no-op and
   *  the app relies on the BYOK onboarding instead. null until the first check. */
  enabled: boolean | null
  /** Why the gate last tripped mid-session, so the login screen can show the
   *  right message. null on a clean first run (never signed in) — then the login
   *  screen shows just the buttons, no notice. Cleared on successful sign-in. */
  gateReason: 'account_disabled' | 'unauthorized' | null
  status: AuthStatus
}

const EMPTY_ACCOUNT: AuthAccount = { email: '', name: '', plan: '' }

// Seed "signed-in" from localStorage so a returning user goes straight to chat
// without the login screen flashing while the first status() call resolves. The
// real check reconciles it a beat later (and signs them out if the key is gone).
const SIGNED_IN_CACHE_KEY = 'apexnodes-desktop-signed-in-v1'

function readCachedSignedIn(): boolean {
  if (typeof window === 'undefined') {
    return false
  }

  try {
    return window.localStorage.getItem(SIGNED_IN_CACHE_KEY) === '1'
  } catch {
    return false
  }
}

function writeCachedSignedIn(value: boolean) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    if (value) {
      window.localStorage.setItem(SIGNED_IN_CACHE_KEY, '1')
    } else {
      window.localStorage.removeItem(SIGNED_IN_CACHE_KEY)
    }
  } catch {
    // localStorage unavailable — degrade silently.
  }
}

export const $authState = atom<DesktopAuthState>({
  account: EMPTY_ACCOUNT,
  enabled: null,
  gateReason: null,
  // A cached signed-in user starts unblocked; everyone else waits for the check.
  status: readCachedSignedIn() ? 'signed-in' : 'checking'
})

const patch = (update: Partial<DesktopAuthState>) => $authState.set({ ...$authState.get(), ...update })

function accountFromStatus(status: DesktopManagedStatus): AuthAccount {
  return {
    email: status.email || '',
    name: status.name || '',
    plan: status.plan || ''
  }
}

let refreshPromise: null | Promise<void> = null

// Read the managed status via the desktop bridge and reconcile the gate.
//   - bridge absent (web dashboard / dev preview) → managed disabled, don't gate.
//   - enabled && signedIn → signed-in (unblock chat).
//   - enabled && !signedIn → signed-out (show login) UNLESS the gate was already
//     flipped to 'disabled' this session (a re-check must not downgrade the
//     account-abnormal message to a generic logged-out one).
//   - !enabled → managed off for this build; don't block on the account gate.
// Deduped so concurrent callers share one in-flight request.
export async function refreshAuthStatus(): Promise<void> {
  if (refreshPromise) {
    return refreshPromise
  }

  refreshPromise = (async () => {
    const bridge = typeof window !== 'undefined' ? window.hermesDesktop?.managed : undefined

    if (!bridge) {
      patch({ enabled: false })

      return
    }

    try {
      const status = await bridge.status()

      if (!status.enabled) {
        // Managed off — the account gate doesn't apply; leave chat unblocked.
        patch({ enabled: false, status: 'signed-in', account: EMPTY_ACCOUNT, gateReason: null })
        writeCachedSignedIn(false)

        return
      }

      if (status.signedIn) {
        patch({ enabled: true, status: 'signed-in', account: accountFromStatus(status), gateReason: null })
        writeCachedSignedIn(true)

        return
      }

      // Not signed in. Preserve an already-shown 'disabled' message this session.
      writeCachedSignedIn(false)
      patch({
        enabled: true,
        account: EMPTY_ACCOUNT,
        status: $authState.get().status === 'disabled' ? 'disabled' : 'signed-out'
      })
    } catch {
      // status() threw (bridge error). Don't hard-block a returning user on a
      // transient IPC failure: keep a cached signed-in state, otherwise treat as
      // signed-out so the login screen can offer a retry.
      patch({ enabled: true, status: readCachedSignedIn() ? 'signed-in' : 'signed-out' })
    } finally {
      refreshPromise = null
    }
  })()

  return refreshPromise
}

// Called after a successful sign-in (the onboarding managed flow completes, or
// the login screen's browser flow resolves). Optimistically unblocks chat, then
// re-reads status so the account panel gets the real email/plan.
export function markSignedIn(account?: Partial<AuthAccount>) {
  writeCachedSignedIn(true)
  patch({
    status: 'signed-in',
    gateReason: null,
    account: { ...EMPTY_ACCOUNT, ...(account ?? {}) }
  })
  void refreshAuthStatus()
}

// Continuous auth gate: a backend call reported 401 (login lost) or 403
// account_disabled (account abnormal). Clear the cached signed-in state and flip
// the gate so the login screen takes over — the app cannot keep being used.
export function handleAuthGate(payload: DesktopAuthGateEvent) {
  // Ignore on managed-disabled builds — there's no account gate to trip, and a
  // stray 401/403 from a BYOK backend shouldn't force a login screen that build
  // doesn't have.
  if ($authState.get().enabled === false) {
    return
  }

  writeCachedSignedIn(false)
  patch({
    account: EMPTY_ACCOUNT,
    gateReason: payload.reason,
    status: payload.reason === 'account_disabled' ? 'disabled' : 'signed-out'
  })
}

// Escape hatch for the "logged in but the relay-key endpoint isn't deployed"
// case: the managed browser sign-in succeeded (valid account) but couldn't
// provision a relay key, so the onboarding store degrades to BYOK. Managed is
// effectively unavailable this run — drop the account gate so the BYOK onboarding
// picker (which mounts once the gate is satisfied) can take over instead of
// trapping the user on a login screen that can never succeed. On prod
// provision-key is deployed, so this is a rare fallback, not the normal path.
export function markManagedUnavailable() {
  patch({ enabled: false, status: 'signed-in', account: EMPTY_ACCOUNT, gateReason: null })
}

// User chose "退出登录" (logout) in the account panel. Clears the relay key on
// disk via the managed bridge, then flips the gate to signed-out so the login
// screen takes over.
export async function signOutAccount(): Promise<void> {
  const bridge = typeof window !== 'undefined' ? window.hermesDesktop?.managed : undefined

  try {
    await bridge?.signOut()
  } catch {
    // Best-effort: even if the IPC clear fails, drop the local session so the
    // user isn't stranded in a half-signed-out state.
  }

  writeCachedSignedIn(false)
  patch({ account: EMPTY_ACCOUNT, gateReason: null, status: 'signed-out' })
}
