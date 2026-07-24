import { atom } from 'nanostores'

import { $gateway } from '@/store/gateway'

// "Hey Hermes" wake-word listener state for the composer toggle. The gateway is
// the single source of truth (the listener lives in the backend and is shared
// with the TUI under a single-owner mic lease); this atom is the renderer's
// cache of that truth, refreshed from every wake.* RPC response we see.

export interface WakeWordState {
  /** Wake word can run at all (deps + mic + key). False hides the toggle. */
  available: boolean
  /** The listener is armed and owned by this surface. */
  listening: boolean
  /** Last failure reason/hint (start refused, unavailable, …) for the tooltip. */
  notice: string
  /** A toggle RPC is in flight — guards double-clicks. */
  pending: boolean
  /** Human-facing wake phrase, e.g. "hey hermes". */
  phrase: string
}

const INITIAL_WAKE_WORD_STATE: WakeWordState = {
  available: false,
  listening: false,
  notice: '',
  pending: false,
  phrase: ''
}

export const $wakeWord = atom<WakeWordState>(INITIAL_WAKE_WORD_STATE)

export interface WakeStatusResponse {
  available?: boolean
  hint?: string
  listening?: boolean
  owned_by_caller?: boolean
  owner_surface?: string | null
  phrase?: string
  provider?: string
}

export interface WakeStartResponse {
  hint?: string
  owner_surface?: string | null
  phrase?: string
  provider?: string
  reason?: string
  started?: boolean
}

export interface WakeStopResponse {
  reason?: string | null
  stopped?: boolean
}

/** Minimal requester shape — satisfied by both `useGatewayRequest`'s
 *  `requestGateway` and the `$gateway` instance wrapper below. */
export type WakeRequester = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

const gatewayRequester: WakeRequester = async <T>(method: string, params: Record<string, unknown> = {}) => {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Hermes gateway unavailable')
  }

  return gateway.request<T>(method, params)
}

const noticeFrom = (result: { hint?: string; reason?: string | null } | null | undefined): string =>
  result?.hint?.trim() || result?.reason?.trim() || ''

/** Sync the atom from a `wake.status` payload (mount / gateway-ready). */
export function applyWakeStatus(status: WakeStatusResponse | null | undefined): void {
  const current = $wakeWord.get()
  const listening = Boolean(status?.listening)

  $wakeWord.set({
    ...current,
    available: Boolean(status?.available),
    listening,
    notice: listening ? '' : noticeFrom(status),
    phrase: status?.phrase?.trim() || current.phrase
  })
}

/** Sync the atom from a `wake.start` response. A `{started:false, reason}`
 *  refusal keeps the toggle off and surfaces the reason as the tooltip. */
export function applyWakeStartResult(result: WakeStartResponse | null | undefined): void {
  const current = $wakeWord.get()

  if (result?.started) {
    $wakeWord.set({
      ...current,
      available: true,
      listening: true,
      notice: '',
      pending: false,
      phrase: result.phrase?.trim() || current.phrase
    })

    return
  }

  $wakeWord.set({
    ...current,
    // The backend probes requirements on start; an explicit "unavailable"
    // refusal means the feature can't run here, so hide the toggle.
    available: result?.reason === 'unavailable' ? false : current.available,
    listening: false,
    notice: noticeFrom(result),
    pending: false
  })
}

/** Sync the atom from a `wake.stop` response. `{stopped:false, reason:'not_owner'}`
 *  still means WE are not listening, so the toggle lands on off either way. */
export function applyWakeStopResult(result: WakeStopResponse | null | undefined): void {
  const current = $wakeWord.get()

  $wakeWord.set({
    ...current,
    listening: false,
    notice: result?.stopped ? '' : noticeFrom(result),
    pending: false
  })
}

/**
 * Gateway-ready sync + auto-arm (wiring.tsx). Queries `wake.status` first so
 * the button knows availability/phrase even when arming is refused, then arms
 * the listener for this surface exactly like the historical auto-arm did.
 * Best-effort: a gateway without the wake.* methods leaves the atom at its
 * hidden default.
 */
export async function armWakeWord(request: WakeRequester = gatewayRequester): Promise<void> {
  try {
    const status = await request<WakeStatusResponse>('wake.status', {})
    applyWakeStatus(status)

    if (!status?.available || status.listening) {
      return
    }

    const result = await request<WakeStartResponse>('wake.start', { surface: 'gui' })
    applyWakeStartResult(result)
  } catch {
    // Older backends / transient failures — keep whatever we last knew.
  }
}

/** The composer button's click handler: stop when listening, start otherwise. */
export async function toggleWakeWord(request: WakeRequester = gatewayRequester): Promise<void> {
  const state = $wakeWord.get()

  if (state.pending) {
    return
  }

  $wakeWord.set({ ...state, pending: true })

  try {
    if (state.listening) {
      applyWakeStopResult(await request<WakeStopResponse>('wake.stop', {}))
    } else {
      applyWakeStartResult(await request<WakeStartResponse>('wake.start', { surface: 'gui' }))
    }
  } catch (error) {
    const current = $wakeWord.get()

    $wakeWord.set({
      ...current,
      notice: error instanceof Error ? error.message : String(error),
      pending: false
    })
  }
}

/** Test-only reset. */
export function resetWakeWordState(): void {
  $wakeWord.set(INITIAL_WAKE_WORD_STATE)
}
