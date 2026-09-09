/**
 * Quick Entry (renderer side) — the mini composer's own state, and the
 * primary window's bridge back into the real prompt-submit path.
 *
 * The quick window carries NO gateway connection: it hands its text to the main
 * process, which forwards it to the primary renderer, which sends it through the
 * SAME `submitText` the normal composer uses (see
 * app/contrib/hooks/use-quick-entry-bridge). There is no second submit path and
 * no new gateway RPC.
 *
 * The device-local preference (enabled + shortcut) is authoritative in the MAIN
 * process — it owns the OS registration and must restore it on a cold launch
 * without the renderer ever visiting Settings. This module treats what the
 * bridge returns as the truth and caches it for the settings UI, same authority
 * split as keep-awake.
 */

import { atom } from 'nanostores'

export interface QuickEntryState {
  enabled: boolean
  /** null before the first read; the settings row shows a skeleton until then. */
  registered: boolean | null
  /** Why the OS shortcut isn't live: taken by another app, or unusable. */
  error: null | QuickEntryRegistrationError
  shortcut: string
}

export type QuickEntryRegistrationError = 'invalid' | 'taken'

export interface QuickEntryStatus {
  enabled: boolean
  error: null | QuickEntryRegistrationError
  registered: boolean
  shortcut: string
}

export const QUICK_ENTRY_DEFAULT_SHORTCUT = 'CommandOrControl+Shift+Space'

export const $quickEntry = atom<QuickEntryState>({
  enabled: true,
  error: null,
  registered: null,
  shortcut: QUICK_ENTRY_DEFAULT_SHORTCUT
})

function applyStatus(status: QuickEntryStatus | undefined): void {
  if (!status) {
    return
  }

  $quickEntry.set({
    enabled: status.enabled === true,
    error: status.error ?? null,
    registered: status.registered === true,
    shortcut: typeof status.shortcut === 'string' && status.shortcut ? status.shortcut : QUICK_ENTRY_DEFAULT_SHORTCUT
  })
}

/** True when the shell exposes the Quick Entry capability (desktop only). */
export function canUseQuickEntry(): boolean {
  return typeof window !== 'undefined' && typeof window.hermesDesktop?.quickEntry?.getSettings === 'function'
}

/** Read the live registration state into the store (Settings mount). */
export async function loadQuickEntrySettings(): Promise<void> {
  if (!canUseQuickEntry()) {
    return
  }

  try {
    applyStatus(await window.hermesDesktop.quickEntry.getSettings())
  } catch {
    // A failed read leaves the store as-is; the row keeps its last known copy.
  }
}

/**
 * Write a preference and adopt whatever the main process reports back — a
 * rejected shortcut or an already-taken chord comes back as an error state
 * instead of a silently-lost setting.
 */
export async function saveQuickEntrySettings(patch: { enabled?: boolean; shortcut?: string }): Promise<void> {
  if (!canUseQuickEntry()) {
    return
  }

  // Optimistic: paint the intent immediately, then let the authoritative reply
  // (which knows whether the OS accepted it) get the last word.
  const previous = $quickEntry.get()
  $quickEntry.set({ ...previous, ...patch, registered: previous.registered })

  try {
    applyStatus(await window.hermesDesktop.quickEntry.setSettings(patch))
  } catch {
    $quickEntry.set(previous)
  }
}

// ── Quick window submit state machine ───────────────────────────────────────

/** A recent session the quick window can target (pushed by the primary). */
export interface QuickEntrySessionOption {
  id: string
  title: string
}

/** Send into whatever chat the main window currently has in front. */
export const QUICK_TARGET_CURRENT = 'current'
/** Start a brand-new session for this prompt. */
export const QUICK_TARGET_NEW = 'new'

/**
 * The primary renderer's push into the quick window: is the gateway usable, and
 * which recent sessions can be targeted. The quick window has NO gateway of its
 * own, so this pushed copy is its only view of backend truth — it starts
 * disconnected (input disabled) until the first push proves otherwise.
 */
export interface QuickEntryStatePush {
  connected: boolean
  sessions: QuickEntrySessionOption[]
}

/** What a quick-window submit carries back to the primary renderer. */
export interface QuickEntrySubmitPayload {
  /** QUICK_TARGET_CURRENT, QUICK_TARGET_NEW, or a stored session id. */
  target: string
  text: string
}

export interface QuickEntrySubmitResult {
  ok: boolean
  code?: string
  message?: string
  retryable?: boolean
  sessionId?: string | null
  runtimeSessionId?: string | null
}

/**
 * The quick window's own composer state. Deliberately a tiny pure reducer: the
 * behavior that would actually break a user — an empty submit must not send but
 * must still not hide the window, a real submit clears the draft AND hides, a
 * double-fire while already submitting must not send twice, and a dead gateway
 * must disable sending entirely — is the part worth proving, and none of it
 * needs React or Electron.
 */
export interface QuickComposerState {
  /** Last pushed gateway truth. False (the initial value) disables submit. */
  connected: boolean
  draft: string
  /** Recent sessions the picker offers, pushed by the primary renderer. */
  sessions: QuickEntrySessionOption[]
  /** True between a send and its acknowledgement. Blocks a double-send. */
  submitting: boolean
  /** Inline delivery failure retained with the draft until retry. */
  error: null | string
  /** Local correlation for ignoring an acknowledgement from an older submit. */
  pendingSubmitId: number | null
  /** Text of the in-flight submit, kept so a late failure can restore it. */
  lastSubmitText: string
  /** A failure whose generation no longer owns the window. The next summon
   *  restores this text instead of silently dropping the prompt. */
  orphanedFailure: null | { message: string; text: string }
  /** A submit whose outcome is UNKNOWN (relay timeout): the backend may still
   *  have accepted it. Kept so a late acknowledgement can reconcile, and never
   *  presented as retryable until non-acceptance is proven. */
  unknownSubmitId: number | null
  /** Where a submit lands: current / new / a stored session id. */
  target: string
  /** Whether the window should be visible. False asks the shell to hide. */
  visible: boolean
}

export type QuickComposerEvent =
  | { type: 'blur' }
  | { type: 'dismiss' }
  | { type: 'edit'; draft: string }
  | { type: 'shown' }
  | { type: 'state'; connected: boolean; sessions: QuickEntrySessionOption[] }
  | { type: 'submit'; submitId?: number }
  | { type: 'submit-error'; message: string; submitId: number }
  | { message: string; submitId: number; type: 'submit-unknown' }
  | { message: string; ok: boolean; type: 'late-result' }
  | { type: 'submit-ok'; submitId: number }
  | { type: 'target'; target: string }

/**
 * Map a relay result to the composer event that reconciles it. A timeout is an
 * UNKNOWN outcome — the prompt may already be accepted — so it must never be
 * mapped to a retryable failure.
 */
export function quickEntryResultEvent(result: QuickEntrySubmitResult, submitId: number): QuickComposerEvent {
  if (result.ok) {
    return { submitId, type: 'submit-ok' }
  }

  if (result.code === 'timeout') {
    return {
      message: result.message || 'Hermes has not confirmed the prompt yet — it may still be delivered.',
      submitId,
      type: 'submit-unknown'
    }
  }

  return {
    message: result.message || 'Quick Entry could not deliver the prompt.',
    submitId,
    type: 'submit-error'
  }
}

export interface QuickComposerTransition {
  /** Payload to send through the real prompt-submit path, or null for none. */
  send: null | QuickEntrySubmitPayload
  state: QuickComposerState
}

export const initialQuickComposerState: QuickComposerState = {
  // Disconnected until the primary renderer's first push proves otherwise — a
  // capture window that accepts text it can never deliver is a lie.
  connected: false,
  draft: '',
  error: null,
  lastSubmitText: '',
  orphanedFailure: null,
  pendingSubmitId: null,
  sessions: [],
  submitting: false,
  target: QUICK_TARGET_CURRENT,
  unknownSubmitId: null,
  visible: true
}

export function quickComposerReducer(state: QuickComposerState, event: QuickComposerEvent): QuickComposerTransition {
  switch (event.type) {
    case 'blur':
    case 'dismiss': {
      // Escape / focus loss discards a surface with nothing unresolved. A submit
      // already handed to main keeps its correlation and draft: the promise
      // still resolves, and a late failure must be able to restore the text.
      const unresolved = state.submitting || state.unknownSubmitId !== null

      return {
        send: null,
        state: unresolved
          ? { ...state, error: null, visible: false }
          : {
              ...state,
              draft: '',
              error: null,
              lastSubmitText: '',
              orphanedFailure: null,
              pendingSubmitId: null,
              submitting: false,
              target: QUICK_TARGET_CURRENT,
              unknownSubmitId: null,
              visible: false
            }
      }
    }

    case 'edit': {
      return { send: null, state: { ...state, draft: event.draft } }
    }

    case 'shown': {
      // Re-summoned. With a submit unresolved the window reconnects to the SAME
      // generation and shows the text still being delivered. Otherwise the
      // surface is fresh — except that a failure whose generation lost the
      // window hands its text back rather than dropping it.
      if (state.submitting || state.unknownSubmitId !== null) {
        return { send: null, state: { ...state, error: null, visible: true } }
      }

      return {
        send: null,
        state: {
          ...state,
          draft: state.orphanedFailure?.text ?? '',
          error: state.orphanedFailure?.message ?? null,
          lastSubmitText: '',
          orphanedFailure: null,
          pendingSubmitId: null,
          submitting: false,
          target: QUICK_TARGET_CURRENT,
          unknownSubmitId: null,
          visible: true
        }
      }
    }

    case 'state': {
      // Adopt the pushed truth. A selected session that no longer exists in the
      // pushed list must not silently swallow the prompt — fall back to current.
      const targetStillValid =
        event.connected &&
        (state.target === QUICK_TARGET_CURRENT ||
          state.target === QUICK_TARGET_NEW ||
          event.sessions.some(session => session.id === state.target))

      return {
        send: null,
        state: {
          ...state,
          connected: event.connected,
          sessions: event.sessions,
          target: targetStillValid ? state.target : QUICK_TARGET_CURRENT
        }
      }
    }

    case 'submit': {
      const text = state.draft.trim()

      // Nothing to send — or nowhere to send it (gateway down): stay open and
      // keep the draft so a stray Enter can't make the text vanish. Re-entering
      // changed text supersedes the pending generation; an unknown outcome never
      // invites a second delivery attempt.
      if (
        !text ||
        !state.connected ||
        state.unknownSubmitId !== null ||
        (state.submitting && text === state.lastSubmitText)
      ) {
        return { send: null, state }
      }

      return {
        send: { target: state.target, text },
        // Wait for main's result before clearing or hiding the capture window (#85590).
        state: {
          ...state,
          error: null,
          // If changed text supersedes an unresolved generation, retain the
          // older prompt: its late failure must hand that text back later.
          lastSubmitText: state.submitting ? state.lastSubmitText : text,
          pendingSubmitId: event.submitId ?? null,
          submitting: true
        }
      }
    }

    case 'submit-ok': {
      // The owning generation — or a submit whose outcome was unknown — clears
      // the surface. A late success for a superseded generation is still good
      // news, but it must not wipe the draft the window now owns.
      const owner = state.pendingSubmitId === event.submitId || state.unknownSubmitId === event.submitId

      return event.submitId > 0 && owner
        ? {
            send: null,
            state: {
              ...state,
              draft: '',
              error: null,
              lastSubmitText: '',
              // A successful newer generation does not erase an older
              // generation's already-recorded late failure.
              orphanedFailure: state.orphanedFailure,
              pendingSubmitId: null,
              submitting: false,
              unknownSubmitId: null,
              visible: false
            }
          }
        : { send: null, state }
    }

    case 'submit-unknown': {
      if (event.submitId <= 0 || state.pendingSubmitId !== event.submitId) {
        return { send: null, state }
      }

      // Delivery is UNCONFIRMED, not failed: keep the draft, keep the
      // correlation, and never present this as retryable.
      return {
        send: null,
        state: {
          ...state,
          error: event.message,
          pendingSubmitId: null,
          submitting: false,
          unknownSubmitId: event.submitId,
          visible: true
        }
      }
    }

    case 'late-result': {
      // A late outcome only reconciles a submit the window still holds as
      // UNKNOWN. Without one it must not clobber a fresh draft.
      if (state.unknownSubmitId === null) {
        return { send: null, state }
      }

      return event.ok
        ? {
            send: null,
            state: {
              ...state,
              draft: '',
              error: null,
              lastSubmitText: '',
              orphanedFailure: null,
              unknownSubmitId: null,
              visible: false
            }
          }
        : {
            send: null,
            state: {
              ...state,
              error: event.message,
              lastSubmitText: '',
              unknownSubmitId: null,
              visible: true
            }
          }
    }

    case 'submit-error': {
      if (event.submitId > 0 && state.pendingSubmitId === event.submitId) {
        return {
          send: null,
          state: {
            ...state,
            error: event.message,
            lastSubmitText: '',
            pendingSubmitId: null,
            submitting: false,
            visible: true
          }
        }
      }

      if (event.submitId > 0 && state.unknownSubmitId === event.submitId) {
        // Non-acceptance is now proven: drop the unknown correlation so a
        // retry is legitimate, and keep the text.
        return {
          send: null,
          state: { ...state, error: event.message, lastSubmitText: '', unknownSubmitId: null }
        }
      }

      // Late failure whose generation no longer owns the window: keep the text
      // for the next summon and surface the message now if nothing else has.
      return {
        send: null,
        state: {
          ...state,
          error: state.error ?? event.message,
          orphanedFailure: state.lastSubmitText
            ? { message: event.message, text: state.lastSubmitText }
            : state.orphanedFailure,
          lastSubmitText: ''
        }
      }
    }

    case 'target': {
      return { send: null, state: { ...state, target: event.target } }
    }

    default: {
      return { send: null, state }
    }
  }
}

// ── Primary-renderer bridge ────────────────────────────────────────────────

let submitHandler: ((payload: QuickEntrySubmitPayload & { correlationId: string }) => void) | null = null
let unsubscribeSubmit: (() => void) | null = null

/**
 * Register the handler that turns a quick-window submit into a real send. The
 * primary window routes it by target: current chat → `submitText`, a stored
 * session id → resume + submit, new → fresh draft + submit.
 */
export function setQuickEntrySubmitHandler(
  fn: ((payload: QuickEntrySubmitPayload & { correlationId: string }) => void) | null
): void {
  submitHandler = fn
}

function normalizeSubmitPayload(raw: unknown): null | QuickEntrySubmitPayload {
  // Tolerate the v1 bare-string wire shape (an older quick window after a
  // partial update) by treating it as "send to the current chat".
  if (typeof raw === 'string') {
    return raw.trim() ? { target: QUICK_TARGET_CURRENT, text: raw } : null
  }

  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const text = typeof record.text === 'string' ? record.text : ''

  if (!text.trim()) {
    return null
  }

  return {
    target: typeof record.target === 'string' && record.target ? record.target : QUICK_TARGET_CURRENT,
    text
  }
}

/**
 * Wire the quick-window → primary-renderer submit channel once. Returns a
 * disposer. Idempotent — a second call while wired is a no-op.
 */
export function initQuickEntryBridge(): () => void {
  const api = typeof window === 'undefined' ? undefined : window.hermesDesktop?.quickEntry

  if (!api?.onSubmit || unsubscribeSubmit) {
    return () => {}
  }

  unsubscribeSubmit = api.onSubmit(raw => {
    const payload = normalizeSubmitPayload(raw)

    if (payload && typeof raw === 'object' && raw !== null) {
      const correlationId = (raw as unknown as Record<string, unknown>).correlationId
      if (typeof correlationId === 'string') {
        submitHandler?.({ ...payload, correlationId })
      }
    }
  })

  return () => {
    unsubscribeSubmit?.()
    unsubscribeSubmit = null
  }
}
