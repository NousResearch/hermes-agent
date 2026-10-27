import type { FreeTierClaimNudgeResult } from '@hermes/shared'
import { atom } from 'nanostores'

import { invalidateAllConnectors } from '@/app/capabilities/connectors/data/keys'
import {
  cancelOAuthSession,
  getNousHandoffStatus,
  listOAuthProviders,
  OAUTH_LONG_POLL_SECONDS,
  pollOAuthSession,
  type ProfileScope,
  startOAuthLogin
} from '@/hermes'
import { LONG_POLL_MIN_GAP_MS, pause, restOfLongPollGap } from '@/lib/long-poll'
import { gatewayActivationEpoch } from '@/store/gateway'
import type { AppHandoffStatus, FreeTierStatus, OAuthPollResponse } from '@/types/hermes'

import { type FreeTierRequester, NOUS_PROVIDER_ID, refreshFreeTierStatus } from './free-tier'
import { $localModelOfferPending } from './local-model-offer'
import { onboardingSurfaceActive } from './onboarding-presence'

/** Why a sign-in ended without tokens. Each maps to one ruled screen:
 *  `busy` (the account service asked for a short wait — a busy account, a
 *  rate limit, the ops pause), `unreachable` (the service could not be
 *  reached or errored; a sign-in cannot help until it is back), `unavailable`
 *  (a terminal refusal: this version, a proof-of-work request, a locked
 *  session), and `timed_out` (the code expired). Anything the backend does
 *  not name lands on `error`, which carries its own message. */
export type FreeTierSignInFailure = 'busy' | 'error' | 'timed_out' | 'unavailable' | 'unreachable'

export type FreeTierOfferKind = 'connector' | 'trial'

export type FreeTierSignInState =
  | { status: 'already_signed_in' }
  | { status: 'closed' }
  | { status: 'finishing' }
  // A "please open the dialog" request from an entry point that has no gateway
  // requester of its own. The mounted host picks it up and drives the flow.
  | { status: 'requested' }
  // A sign-in offer the backend timed (repeating with back-off): `trial` after a
  // finished free-tier task, `connector` after apps connect without the trial.
  // Unlike `requested` it waits for the user's Sign in before starting the transfer.
  | { kind: FreeTierOfferKind; status: 'offer' }
  // `handoff`: the free tier's apps moving to this account (null when the
  // sign-in replaced no free tier). Followed while it is still retrying.
  | { email: null | string; handoff: AppHandoffStatus | null; model: null | string; status: 'completed' }
  // `retryAfter`: the seconds the backend asked us to wait before trying
  // again (0 when it named none); only `busy` screens read it.
  | { kind: FreeTierSignInFailure; message: null | string; retryAfter: number; status: 'failed' }
  // `minting` is true only when the backend still has to create the free-tier
  // identity (the first `start` does it), which is the one case where the user
  // waits on something worth naming.
  | { minting: boolean; status: 'setting_up' }
  // `code` is shown only to compare with the browser page; `url` reopens it.
  | { code: string; sessionId: string; status: 'code'; url: string }

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

let expiryTimer: number | null = null
// Each begin/close bumps the generation. A continuation that resumes after an await (a poll that was
// already in flight when the dialog closed, a start call for an abandoned attempt) compares its own
// generation and drops out, so a stale attempt never publishes over the one now on screen. The poll
// loops end the same way.
let attempt = 0

function clearTimers() {
  if (expiryTimer !== null) {
    window.clearTimeout(expiryTimer)
    expiryTimer = null
  }
}

const set = (state: FreeTierSignInState) => $freeTierSignIn.set(state)

// `requestGateway` keeps one identity across backend switches, so a reply is tied to the active route
// it was asked under: a late answer from the previous connection or profile must not land on this one.
export function sameGatewayRoute(): () => boolean {
  const epoch = gatewayActivationEpoch()

  return () => gatewayActivationEpoch() === epoch
}

const fail = (kind: FreeTierSignInFailure, message: null | string = null, retryAfter = 0) => {
  clearTimers()
  set({
    kind,
    message: message?.trim() || null,
    retryAfter: Math.max(0, Math.round(retryAfter) || 0),
    status: 'failed'
  })
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

let offerTimer: number | null = null
let offerTimerFor: FreeTierStatus | null = null
let offerTimerRoute: () => boolean = () => false
let offerClaiming = false

function clearOfferTimer() {
  if (offerTimer !== null) {
    window.clearTimeout(offerTimer)
    offerTimer = null
  }
}

// Bumped when a turn completes or an app connects. The backend records the finished task (or the
// connected app) before it reports either, so the dialog owner's re-read on this signal learns when
// the next offer is due. Not gated on the cached status: a turn's own connect can create the identity
// the cache predates, and the re-read is local and zero-network.
export const $freeTierTurnCompleted = atom(0)

export function noteFreeTierTurnComplete() {
  $freeTierTurnCompleted.set($freeTierTurnCompleted.get() + 1)
}

/** An app connected from the Connectors page, outside any turn. */
export const noteFreeTierAppConnected = noteFreeTierTurnComplete

/** Stop a pending offer timer (the dialog owner unmounted). */
export function stopFreeTierOffer() {
  clearOfferTimer()
}

// The trial offer rides on the trial being live; the connector offer is for users without it, so
// the backend owes at most one of the two.
function dueOffer(status: FreeTierStatus | null): { dueIn: number; kind: FreeTierOfferKind } | null {
  const trial = status?.available ? (status.nudge_due_in ?? null) : null

  if (trial !== null) {
    return { dueIn: trial, kind: 'trial' }
  }

  const connector = status?.connector_nudge_due_in ?? null

  return connector === null ? null : { dueIn: connector, kind: 'connector' }
}

/**
 * Act on the backend's `nudge_due_in` / `connector_nudge_due_in` (it alone
 * decides when the next offer is due). Later: one timer that re-reads the
 * status, whose fresh answer comes back through here. Due now: claim it, and
 * only the one caller the backend says `claimed` opens the offer. Called on
 * every status read and when guided onboarding leaves the screen; a guarded
 * call simply waits for the next.
 */
export function syncFreeTierOffer(status: FreeTierStatus | null, requestGateway: FreeTierRequester) {
  // A re-sync on the same read (an onboarding change) keeps its running timer, so it is not pushed back.
  if (offerTimer !== null && status === offerTimerFor && offerTimerRoute()) {
    return
  }

  clearOfferTimer()
  const due = dueOffer(status)

  if (due === null) {
    return
  }

  const { dueIn, kind } = due

  if (dueIn > 0) {
    const isCurrent = sameGatewayRoute()
    offerTimerFor = status
    offerTimerRoute = isCurrent
    offerTimer = window.setTimeout(() => {
      offerTimer = null

      if (isCurrent()) {
        void refreshFreeTierStatus(requestGateway, isCurrent)
      }
    }, dueIn * 1000)

    return
  }

  void claimFreeTierOffer(requestGateway, kind)
}

/** Connector app windows on screen. The offer waits for them to close, so it never covers one. */
export const $openAppWindows = atom(0)

// The local-model offer after onboarding goes first; the sign-in offer is claimed once it is answered.
const offerBlocked = () =>
  offerClaiming ||
  onboardingSurfaceActive() ||
  $openAppWindows.get() > 0 ||
  $localModelOfferPending.get() ||
  $freeTierSignIn.get().status !== 'closed'

async function claimFreeTierOffer(requestGateway: FreeTierRequester, kind: FreeTierOfferKind) {
  if (offerBlocked()) {
    return
  }

  const isCurrent = sameGatewayRoute()
  offerClaiming = true
  let claimed: boolean

  try {
    // A trial claim sends no `kind`: an older backend rejects the unknown key, and defaults to the trial.
    const params = kind === 'trial' ? undefined : { kind }
    claimed = (await requestGateway<FreeTierClaimNudgeResult>('free_tier.claim_nudge', params))?.claimed === true
  } catch {
    // Not claimed; the next status read tries again. No re-read here, or a failing call would loop.
    return
  } finally {
    offerClaiming = false
  }

  if (!isCurrent()) {
    return
  }

  if (claimed && $freeTierSignIn.get().status === 'closed') {
    set({ kind, status: 'offer' })
  }

  // The claim settled the offer either way; re-read so the cached due time drops.
  void refreshFreeTierStatus(requestGateway, isCurrent)
}

/** Close and abandon. Cancels a live device-code session so the backend is not
 *  left polling a window nobody is watching. */
export function closeFreeTierSignIn() {
  const state = $freeTierSignIn.get()

  attempt += 1
  clearTimers()

  if (state.status === 'code') {
    cancelOAuthSession(state.sessionId).catch(() => undefined)
  }

  set({ status: 'closed' })
}

// The reasons the backend names on a non-approved terminal poll: an expired
// code, and the account service's `anon_*` verdicts when it was busy,
// unreachable or refused mid sign-in (`hermes_cli/anon_sign_in.py`). Anything
// else falls through to the generic error screen, which shows the backend's
// own message.
const FAILURE_BY_REASON: Record<string, FreeTierSignInFailure> = {
  anon_account_locked: 'unavailable',
  anon_gate_closed: 'unavailable',
  anon_gate_paused: 'busy',
  anon_pow_required: 'unavailable',
  anon_rate_limited: 'busy',
  anon_server_error: 'unreachable',
  anon_unreachable: 'unreachable',
  timeout: 'timed_out'
}

/** The screen a failed poll maps to, exported for the dialog's tests. */
export function signInFailureKind(reason: null | string | undefined): FreeTierSignInFailure {
  return FAILURE_BY_REASON[reason ?? ''] ?? 'error'
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
 * on, start the device-code sign-in, open its page, then long-poll until it
 * resolves. Safe to call again from a "Try again" button — it clears any
 * previous timers first.
 */
export async function beginFreeTierSignIn(requestGateway: FreeTierRequester) {
  clearTimers()
  const mine = ++attempt
  const stale = () => mine !== attempt

  const status = await refreshFreeTierStatus(requestGateway)

  if (stale()) {
    return
  }

  const minting = !status?.has_guest

  // No free-tier identity AND a real Nous account already connected: there is
  // nothing to transfer. Say so instead of minting a guest the user does not
  // need. A failed provider read is not proof either way — fall through and let
  // the start call be the authority.
  if (minting) {
    try {
      const { providers } = await listOAuthProviders()

      if (stale()) {
        return
      }

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

    if (stale()) {
      // The user closed the dialog while the start call was out: do not leave the backend
      // polling a session nobody is watching.
      cancelOAuthSession(start.session_id).catch(() => undefined)

      return
    }

    if (start.flow !== 'device_code') {
      fail('error', null)

      return
    }

    await openSignInUrl(start.verification_url)

    if (stale()) {
      cancelOAuthSession(start.session_id).catch(() => undefined)

      return
    }

    set({ code: start.user_code, sessionId: start.session_id, status: 'code', url: start.verification_url })

    // Lapse locally when the code's own window closes, instead of polling a
    // dead session forever. The backend usually flips it first and its message
    // wins; this is the floor.
    const ttlMs = Math.max(1, Number(start.expires_in) || 0) * 1000
    expiryTimer = window.setTimeout(() => {
      expiryTimer = null
      fail('timed_out', null)
    }, ttlMs)

    void pollUntilDone(start.session_id, requestGateway, mine)
  } catch (error) {
    if (!stale()) {
      fail('error', error instanceof Error ? error.message : String(error))
    }
  }
}

// Long-poll the sign-in until it leaves `pending`, then finish it. The loop ends with its screen: a
// close, a newer attempt or the local expiry floor.
async function pollUntilDone(sessionId: string, requestGateway: FreeTierRequester, mine: number) {
  const stale = () => mine !== attempt || $freeTierSignIn.get().status !== 'code'

  try {
    for (;;) {
      const askedAt = Date.now()
      const result = await pollOAuthSession(NOUS_PROVIDER_ID, sessionId, undefined, OAUTH_LONG_POLL_SECONDS)

      if (stale()) {
        return
      }

      if (result.status !== 'pending') {
        clearTimers()

        return await finishSignIn(result, requestGateway, mine)
      }

      await restOfLongPollGap(askedAt)

      if (stale()) {
        return
      }
    }
  } catch (error) {
    if (!stale()) {
      fail('error', error instanceof Error ? error.message : String(error))
    }
  }
}

async function finishSignIn(result: OAuthPollResponse, requestGateway: FreeTierRequester, mine: number) {
  if (result.status !== 'approved') {
    fail(signInFailureKind(result.reason), result.error_message ?? null, Number(result.retry_after) || 0)

    return
  }

  set({ status: 'finishing' })

  // The tokens are on disk now, so the backend's view of this identity has
  // changed: reload its env and re-read the free-tier verdict before the
  // completed screen claims the user is signed in.
  await requestGateway('reload.env').catch(() => undefined)
  await refreshFreeTierStatus(requestGateway)
  invalidateAllConnectors()

  if (mine !== attempt) {
    return
  }

  const handoff = result.handoff_status ?? null
  set({ email: result.account_email ?? null, handoff, model: result.model ?? null, status: 'completed' })

  if (handoff && handoffStillMoving(handoff)) {
    const live = () => mine === attempt && $freeTierSignIn.get().status === 'completed'

    const publish = (next: AppHandoffStatus) => {
      const screen = $freeTierSignIn.get()

      if (screen.status === 'completed') {
        set({ ...screen, handoff: next })
      }
    }

    void followHandoff(handoff, live, publish)
  }
}

/** The move is still being retried in the background, so its result can still change. */
export function handoffStillMoving(handoff: AppHandoffStatus): boolean {
  return handoff.state === 'pending' || handoff.state === 'retrying'
}

// Failed handoff reads in a row before the result stops following the move. The note on screen stays
// true either way: the move continues in the background whether or not this window hears about it.
const HANDOFF_READ_RETRIES = 4

/** Keep a shown handoff current while the move is retried, until it settles or `live` goes false
 *  (the screen showing it left). A failed read leaves the last answer up: it already says the move
 *  continues. A settled move changed the apps, so the cached connector reads are dropped. */
export async function followHandoff(
  shown: AppHandoffStatus,
  live: () => boolean,
  publish: (handoff: AppHandoffStatus) => void,
  scope?: ProfileScope
) {
  let current = shown
  let failures = 0

  while (live() && handoffStillMoving(current)) {
    const askedAt = Date.now()

    try {
      current = await getNousHandoffStatus(current.updated_at ?? null, scope)
      failures = 0
    } catch {
      if (++failures > HANDOFF_READ_RETRIES) {
        return
      }

      // Back off: 4 s, 8 s, 16 s, 32 s.
      await pause(LONG_POLL_MIN_GAP_MS * 2 ** failures)

      continue
    }

    if (!live()) {
      return
    }

    publish(current)

    if (!handoffStillMoving(current)) {
      invalidateAllConnectors()

      return
    }

    await restOfLongPollGap(askedAt)
  }
}

/** Reopen the sign-in page the code belongs to. */
export function reopenFreeTierSignInPage() {
  const state = $freeTierSignIn.get()

  return state.status === 'code' ? openSignInUrl(state.url) : Promise.resolve()
}
