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
  /** Armed but the mic delivers only silence (macOS backend-permission gap). */
  audio_silent?: boolean
  available?: boolean
  /** Config truth (wake_word.enabled) — drives post-voice re-arm. */
  enabled?: boolean
  hint?: string
  listening?: boolean
  owned_by_caller?: boolean
  owner_surface?: string | null
  phrase?: string
  provider?: string
}

export interface WakeStartResponse {
  enabled_persisted?: boolean
  hint?: string
  owner_surface?: string | null
  phrase?: string
  provider?: string
  reason?: string
  started?: boolean
}

export interface WakeStopResponse {
  disabled_persisted?: boolean
  reason?: string | null
  stopped?: boolean
}

/** Minimal requester shape — satisfied by both `useGatewayRequest`'s
 *  `requestGateway` and the `$gateway` instance wrapper below. */
export type WakeRequester = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

// First-use wake.start lazy-installs the detection engine (onnxruntime is a
// large wheel) — that legitimately takes minutes. The default 30s WS timeout
// fired mid-install, leaving a dead button that went blue on its own later.
const WAKE_START_TIMEOUT_MS = 180_000

const gatewayRequester: WakeRequester = async <T>(method: string, params: Record<string, unknown> = {}) => {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Hermes gateway unavailable')
  }

  return method === 'wake.start'
    ? gateway.request<T>(method, params, WAKE_START_TIMEOUT_MS)
    : gateway.request<T>(method, params)
}

// Friendly text for the gateway's wake refusal codes (mirrors the TUI's
// START_REASON_TEXT). Unknown codes fall through raw so new server-side
// codes stay visible instead of silently disappearing.
const REASON_TEXT: Record<string, string> = {
  disabled: 'click to enable',
  disabled_for_surface: 'scoped to another surface (config wake_word.surface)',
  not_owner: 'another surface owns the listener',
  owned: 'another surface owns the listener',
  unavailable: 'unavailable'
}

const noticeFrom = (result: { hint?: string; reason?: string | null } | null | undefined): string => {
  const hint = result?.hint?.trim()

  if (hint) {
    return hint
  }

  const reason = result?.reason?.trim()

  return reason ? (REASON_TEXT[reason] ?? reason) : ''
}

/** Sync the atom from a `wake.status` payload (mount / gateway-ready). */
export function applyWakeStatus(status: WakeStatusResponse | null | undefined): void {
  const current = $wakeWord.get()
  const listening = Boolean(status?.listening)
  // "Armed but deaf" (macOS backend without mic permission) keeps its hint
  // visible in the tooltip even though the toggle shows listening.
  const silent = Boolean(status?.audio_silent)

  $wakeWord.set({
    ...current,
    available: Boolean(status?.available),
    listening,
    notice: listening && !silent ? '' : noticeFrom(status),
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

  $wakeWord.set({
    ...state,
    // First arm may lazy-install the detection engine — say so instead of
    // freezing a silent disabled button for the duration.
    notice: state.listening ? '' : 'arming — first use may take a minute while the engine installs',
    pending: true
  })

  try {
    if (state.listening) {
      applyWakeStopResult(await request<WakeStopResponse>('wake.stop', { persist: true }))
    } else {
      // persist: true — a deliberate click is consent, so the backend flips
      // wake_word.enabled in config.yaml (on/off) and the choice sticks for
      // future sessions. Auto-arm (armWakeWord) never passes it.
      applyWakeStartResult(await request<WakeStartResponse>('wake.start', { persist: true, surface: 'gui' }))
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

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

/**
 * Post-voice-turn reconcile: the wake word is a persistent setting, so ending a
 * voice conversation must land the listener back where config says it belongs.
 * `wake.resume` alone isn't enough — the mic can still be held by the just-torn
 * -down WebRTC capture, and a fire-and-forget resume that loses that race left
 * the ear silently off until the user re-toggled. Resume, then verify against
 * `wake.status` (config `enabled` is the authority) and re-arm, with a couple
 * of spaced retries to ride out mic-release latency. Never passes `persist` —
 * this is a passive path and must not flip config.
 */
export async function resumeWakeAfterVoice(request: WakeRequester = gatewayRequester): Promise<void> {
  try {
    await request('wake.resume', {})
  } catch {
    // Older backend without wake.* — nothing to reconcile.
    return
  }

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const status = await request<WakeStatusResponse>('wake.status', {})
      applyWakeStatus(status)

      // Config says off (or the feature can't run) — off is the correct rest
      // state. A user /wake off during the voice turn stays respected.
      if (!status?.enabled || !status.available) {
        return
      }

      if (status.listening) {
        return
      }

      const started = await request<WakeStartResponse>('wake.start', { surface: 'gui' })
      applyWakeStartResult(started)

      if (started?.started) {
        return
      }

      // Another surface holds the mic lease — theirs to keep.
      if (started?.reason === 'owned') {
        return
      }
    } catch {
      // Transient (mic still releasing) — fall through to the next attempt.
    }

    await sleep(1500)
  }
}

/** Test-only reset. */
export function resetWakeWordState(): void {
  $wakeWord.set(INITIAL_WAKE_WORD_STATE)
}
