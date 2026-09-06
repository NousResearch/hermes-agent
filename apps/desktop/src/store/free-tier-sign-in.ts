import { atom } from 'nanostores'

import { cancelOAuthSession, listOAuthProviders, pollOAuthSession, startOAuthLogin } from '@/hermes'

import { type FreeTierRequester, NOUS_PROVIDER_ID, refreshFreeTierStatus } from './free-tier'

const POLL_MS = 2000
const COPY_FLASH_MS = 1500

/** Why a sign-in ended without tokens. Each maps to one ruled screen; anything
 *  the backend does not name (transport failure, `account_busy`) lands on
 *  `error`, which carries the backend's own message when there is one. */
export type FreeTierSignInFailure = 'error' | 'rejected' | 'retired' | 'superseded' | 'timed_out'

export type FreeTierSignInState =
  | { status: 'already_signed_in' }
  | { status: 'closed' }
  | { status: 'finishing' }
  // A "please open the dialog" request from an entry point that has no gateway
  // requester of its own. The mounted host picks it up and drives the flow.
  | { status: 'requested' }
  | { email: null | string; model: null | string; status: 'completed' }
  | { kind: FreeTierSignInFailure; message: null | string; status: 'failed' }
  // `minting` is true only when the backend still has to create the free-tier
  // identity (the first `start` does it), which is the one case where the user
  // waits on something worth naming.
  | { minting: boolean; status: 'setting_up' }
  | {
      code: string
      codeCopied: boolean
      sessionId: string
      url: string
      urlCopied: boolean
      status: 'code'
    }

export const $freeTierSignIn = atom<FreeTierSignInState>({ status: 'closed' })

// Several surfaces can mount the dialog host (the shell, a test harness). The
// FIRST mount claims it; the rest render nothing, so one open never stacks two
// identical dialogs. Mirrors the real-profile-consent claim.
const $claim = atom<null | string>(null)

export function claimFreeTierSignIn(id: string) {
  if ($claim.get() === null) {
    $claim.set(id)
  }
}

export function releaseFreeTierSignIn(id: string) {
  if ($claim.get() === id) {
    $claim.set(null)
  }
}

export function freeTierSignInClaim() {
  return $claim
}

let pollTimer: number | null = null
let expiryTimer: number | null = null

function clearTimers() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }

  if (expiryTimer !== null) {
    window.clearTimeout(expiryTimer)
    expiryTimer = null
  }
}

const set = (state: FreeTierSignInState) => $freeTierSignIn.set(state)

const fail = (kind: FreeTierSignInFailure, message: null | string = null) => {
  clearTimers()
  set({ kind, message: message?.trim() || null, status: 'failed' })
}

/** Every entry point calls this — Settings › Billing, the statusbar chip, the
 *  first-launch intro. It only records the intent; the mounted host owns the
 *  gateway requester and drives the flow. Re-entrant by design: a second click
 *  while a sign-in is already on screen must not restart it. */
export function openFreeTierSignIn() {
  if ($freeTierSignIn.get().status === 'closed') {
    set({ status: 'requested' })
  }
}

/** Close and abandon. Cancels a live device-code session so the backend is not
 *  left polling a window nobody is watching. */
export function closeFreeTierSignIn() {
  const state = $freeTierSignIn.get()

  clearTimers()

  if (state.status === 'code') {
    cancelOAuthSession(state.sessionId).catch(() => undefined)
  }

  set({ status: 'closed' })
}

// The reasons the backend names on a non-approved terminal poll. Anything else
// (including a bare `account_busy`) falls through to the generic error screen,
// which shows the backend's own message.
const FAILURE_BY_REASON: Record<string, FreeTierSignInFailure> = {
  account_not_anonymous: 'retired',
  account_retired: 'retired',
  superseded: 'superseded',
  timeout: 'timed_out',
  user_declined: 'rejected'
}

// Open a sign-in URL through the desktop bridge, falling back to window.open
// when the bridge isn't there (dev preview, tests) so the flow never strands in
// a waiting state. Same contract as the onboarding store's opener.
async function openSignInUrl(url: string) {
  if (window.hermesDesktop?.openExternal) {
    try {
      await window.hermesDesktop.openExternal(url)

      return
    } catch {
      // Bridge present but failed (no OS handler, user denied). Fall through.
    }
  }

  window.open(url, '_blank', 'noopener,noreferrer')
}

/**
 * Drive one sign-in attempt end to end: resolve what identity this Hermes is
 * on, start the transfer, open the consent page, then poll until it resolves.
 * Safe to call again from a "Try again" button — it clears any previous timers
 * first.
 */
export async function beginFreeTierSignIn(requestGateway: FreeTierRequester) {
  clearTimers()

  const status = await refreshFreeTierStatus(requestGateway)
  const minting = !status?.has_guest

  // No free-tier identity AND a real Nous account already connected: there is
  // nothing to transfer. Say so instead of minting a guest the user does not
  // need. A failed provider read is not proof either way — fall through and let
  // the start call be the authority.
  if (minting) {
    try {
      const { providers } = await listOAuthProviders()
      const nous = providers.find(provider => provider.id === NOUS_PROVIDER_ID)

      if (nous?.status.logged_in && nous.status.free_tier !== true) {
        set({ status: 'already_signed_in' })

        return
      }
    } catch {
      // Provider list unavailable — continue with the sign-in.
    }
  }

  set({ minting, status: 'setting_up' })

  try {
    const start = await startOAuthLogin(NOUS_PROVIDER_ID)

    if (start.flow !== 'device_code') {
      fail('error', null)

      return
    }

    await openSignInUrl(start.verification_url)

    set({
      code: start.user_code,
      codeCopied: false,
      sessionId: start.session_id,
      status: 'code',
      url: start.verification_url,
      urlCopied: false
    })

    // Lapse locally when the code's own window closes, instead of polling a
    // dead session forever. The backend usually flips it first and its message
    // wins; this is the floor.
    const ttlMs = Math.max(1, Number(start.expires_in) || 0) * 1000
    expiryTimer = window.setTimeout(() => {
      expiryTimer = null
      fail('timed_out', null)
    }, ttlMs)

    pollTimer = window.setInterval(() => void pollOnce(start.session_id, requestGateway), POLL_MS)
  } catch (error) {
    fail('error', error instanceof Error ? error.message : String(error))
  }
}

async function pollOnce(sessionId: string, requestGateway: FreeTierRequester) {
  try {
    const result = await pollOAuthSession(NOUS_PROVIDER_ID, sessionId)

    if (result.status === 'pending') {
      return
    }

    clearTimers()

    if (result.status !== 'approved') {
      const kind = FAILURE_BY_REASON[result.reason ?? ''] ?? 'error'
      fail(kind, result.error_message ?? null)

      return
    }

    set({ status: 'finishing' })

    // The tokens are on disk now, so the backend's view of this identity has
    // changed: reload its env and re-read the free-tier verdict before the
    // completed screen claims the user is signed in.
    await requestGateway('reload.env').catch(() => undefined)
    await refreshFreeTierStatus(requestGateway)

    set({
      email: result.account_email ?? null,
      model: result.model ?? null,
      status: 'completed'
    })
  } catch (error) {
    fail('error', error instanceof Error ? error.message : String(error))
  }
}

async function copyAndFlash(text: string, field: 'codeCopied' | 'urlCopied') {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    return
  }

  const current = $freeTierSignIn.get()

  if (current.status !== 'code') {
    return
  }

  const { sessionId } = current
  set({ ...current, [field]: true })

  window.setTimeout(() => {
    const later = $freeTierSignIn.get()

    if (later.status === 'code' && later.sessionId === sessionId) {
      set({ ...later, [field]: false })
    }
  }, COPY_FLASH_MS)
}

export function copyFreeTierCode() {
  const state = $freeTierSignIn.get()

  return state.status === 'code' ? copyAndFlash(state.code, 'codeCopied') : Promise.resolve()
}

export function copyFreeTierUrl() {
  const state = $freeTierSignIn.get()

  return state.status === 'code' ? copyAndFlash(state.url, 'urlCopied') : Promise.resolve()
}
