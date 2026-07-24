/**
 * Debug trace — optional, verbose instrumentation of stateful desktop events.
 *
 * When enabled (Settings → Advanced), dumps structured `console.debug` entries
 * for every session state transition, compaction event, message completion,
 * session switch, persistence write, and gateway event. The output lands in the
 * renderer devtools console and any attached log capture, so you can point at
 * the trace after reproducing a bug.
 *
 * Zero cost when disabled: every call site is a single function call that
 * early-returns when the atom is off. The subscriptions (persistence, gateway
 * events, atom watchers) are only attached once at module init, and their
 * callbacks also early-return — so the overhead is one closure call per event,
 * which is negligible.
 */

import { atom } from 'nanostores'

import { onGatewayEvent } from '@/contrib/events'
import { onPersistenceEvent } from '@/lib/storage'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $activeSessionStoredIdRotation,
  $awaitingResponse,
  $busy,
  $connection,
  $gatewayState,
  $messages,
  $resumeExhaustedSessionId,
  $resumeFailedSessionId,
  $selectedStoredSessionId,
  $sessions
} from '@/store/session'
import { $clarifyRequest } from '@/store/clarify'
import { $approvalRequest, $secretRequest, $sudoRequest } from '@/store/prompts'

const KEY = 'hermes.desktop.debugTrace.v1'

/** Device-local preference — off by default, per machine. */
export const $debugTraceEnabled = atom<boolean>(
  typeof window === 'undefined' ? false : (() => {
    try {
      return window.localStorage.getItem(KEY) === 'true'
    } catch {
      return false
    }
  })()
)

export function setDebugTraceEnabled(on: boolean): void {
  $debugTraceEnabled.set(on)
}

if (typeof window !== 'undefined') {
  $debugTraceEnabled.subscribe(on => {
    try {
      window.localStorage.setItem(KEY, String(on))
    } catch {
      // Storage best-effort.
    }
  })
}

export type DebugCategory =
  | 'session-state'
  | 'compaction'
  | 'message'
  | 'session-switch'
  | 'persistence'
  | 'gateway-event'
  | 'connection'
  | 'profile-switch'
  | 'resume'
  | 'busy'
  | 'error-boundary'
  | 'prompt'
  | 'sessions-list'

const CATEGORY_PREFIX: Record<DebugCategory, string> = {
  'session-state': '[trace:session-state]',
  compaction: '[trace:compaction]',
  message: '[trace:message]',
  'session-switch': '[trace:session-switch]',
  persistence: '[trace:persistence]',
  'gateway-event': '[trace:gateway-event]',
  connection: '[trace:connection]',
  'profile-switch': '[trace:profile-switch]',
  resume: '[trace:resume]',
  busy: '[trace:busy]',
  'error-boundary': '[trace:error-boundary]',
  prompt: '[trace:prompt]',
  'sessions-list': '[trace:sessions-list]'
}

/**
 * Emit a debug trace entry. No-ops entirely when tracing is disabled.
 *
 * `data` is spread as additional console arguments (not stringified) so
 * devtools can expand/inspect objects natively.
 */
export function debugTrace(
  category: DebugCategory,
  message: string,
  ...data: unknown[]
): void {
  if (!$debugTraceEnabled.get()) {
    return
  }

  const ts = new Date().toISOString()

  // eslint-disable-next-line no-console
  console.debug(`${CATEGORY_PREFIX[category]} ${ts} ${message}`, ...data)
}

// ---------------------------------------------------------------------------
// Subscriptions — attached once at module init, no-op when disabled.
// ---------------------------------------------------------------------------

if (typeof window !== 'undefined') {
  // --- Session switches ---
  let prevActive: string | null = null
  $activeSessionId.subscribe(id => {
    if (id !== prevActive) {
      debugTrace('session-switch', `activeSessionId ${prevActive ?? 'null'} → ${id ?? 'null'}`)
      prevActive = id
    }
  })

  let prevSelected: string | null = null
  $selectedStoredSessionId.subscribe(id => {
    if (id !== prevSelected) {
      debugTrace('session-switch', `selectedStoredSessionId ${prevSelected ?? 'null'} → ${id ?? 'null'}`)
      prevSelected = id
    }
  })

  // --- Persistence events ---
  onPersistenceEvent(event => {
    const valuePreview =
      event.value === null
        ? 'null'
        : event.value.length > 120
          ? `${event.value.slice(0, 120)}…(${event.value.length} chars)`
          : event.value

    debugTrace('persistence', `${event.op} ${event.key}`, { value: valuePreview })
  })

  // --- Gateway events (wildcard) ---
  onGatewayEvent('*', event => {
    const type = event.type ?? 'unknown'
    const raw = event as unknown as Record<string, unknown>
    const sessionId = raw.session_id ?? raw.sessionId ?? null

    // Summarize — don't dump the full payload for high-frequency deltas.
    const summary: Record<string, unknown> = { type }

    if (sessionId) {
      summary.sessionId = sessionId
    }

    // For message.delta, just note it happened (they fire 30×/s during a turn).
    // For everything else, include the payload for inspection.
    if (type === 'message.delta') {
      summary.note = 'delta (streaming)'
    } else {
      summary.payload = event
    }

    debugTrace('gateway-event', type, summary)
  })

  // --- Gateway connection state ---
  let prevGatewayState: string | undefined
  $gatewayState.subscribe(state => {
    if (state !== prevGatewayState) {
      debugTrace('connection', `gatewayState ${prevGatewayState ?? 'undefined'} → ${state}`)
      prevGatewayState = state
    }
  })

  // --- Connection (mode/baseUrl/profile) ---
  let prevConnMode: string | undefined
  let prevConnProfile: string | undefined
  $connection.subscribe(conn => {
    const mode = conn?.mode ?? 'null'
    const profile = conn?.profile ?? 'null'

    if (mode !== prevConnMode || profile !== prevConnProfile) {
      debugTrace('connection', `connection mode=${mode} profile=${profile} baseUrl=${conn?.baseUrl ?? 'null'}`)
      prevConnMode = mode
      prevConnProfile = profile
    }
  })

  // --- Profile switches ---
  let prevProfile: string | undefined
  $activeGatewayProfile.subscribe(profile => {
    if (profile !== prevProfile) {
      debugTrace('profile-switch', `${prevProfile ?? 'undefined'} → ${profile}`)
      prevProfile = profile
    }
  })

  // --- Resume failures + exhaustion ---
  let prevResumeFailed: string | null = null
  $resumeFailedSessionId.subscribe(id => {
    if (id !== prevResumeFailed) {
      debugTrace('resume', `resumeFailedSessionId ${prevResumeFailed ?? 'null'} → ${id ?? 'null'}`)
      prevResumeFailed = id
    }
  })

  let prevResumeExhausted: string | null = null
  $resumeExhaustedSessionId.subscribe(id => {
    if (id !== prevResumeExhausted) {
      debugTrace('resume', `resumeExhaustedSessionId ${prevResumeExhausted ?? 'null'} → ${id ?? 'null'}`)
      prevResumeExhausted = id
    }
  })

  // --- Busy / awaitingResponse edges ---
  let prevBusy = false
  $busy.subscribe(busy => {
    if (busy !== prevBusy) {
      debugTrace('busy', `busy ${prevBusy} → ${busy}`, { activeSessionId: $activeSessionId.get() })
      prevBusy = busy
    }
  })

  let prevAwaiting = false
  $awaitingResponse.subscribe(awaiting => {
    if (awaiting !== prevAwaiting) {
      debugTrace('busy', `awaitingResponse ${prevAwaiting} → ${awaiting}`, { activeSessionId: $activeSessionId.get() })
      prevAwaiting = awaiting
    }
  })

  // --- Message array length changes ---
  // Not per-token (that'd be insane) — only when the count changes, which
  // captures: new message added, transcript cleared, session switched,
  // reconciliation replaced the array.
  let prevMessageCount = -1
  $messages.subscribe(messages => {
    const count = messages.length

    if (count !== prevMessageCount) {
      debugTrace('message', `messages count ${prevMessageCount} → ${count}`, {
        activeSessionId: $activeSessionId.get()
      })
      prevMessageCount = count
    }
  })

  // --- Compression id rotation ---
  // Fires when auto-compaction rotates the active session's stored id mid-turn.
  // One of the spookiest bug classes — the route / pin / draft key silently
  // changes under the user.
  $activeSessionStoredIdRotation.subscribe(rotation => {
    if (rotation) {
      debugTrace('session-switch', `compression id rotation`, {
        prev: rotation.previousStoredSessionId,
        next: rotation.nextStoredSessionId,
        runtime: rotation.runtimeSessionId,
        isActive: rotation.runtimeSessionId === $activeSessionId.get()
      })
    }
  })

  // --- Blocking prompts (clarify / approval / sudo / secret) ---
  // When these appear/disappear, the chat is blocked. Tracing the edges
  // catches "agent silently stalled" bugs.
  let prevClarify: unknown = null
  $clarifyRequest.subscribe(req => {
    if (req !== prevClarify) {
      debugTrace('prompt', `clarify ${prevClarify ? 'cleared' : 'raised'}`, {
        sessionId: $activeSessionId.get(),
        requestId: req ? 'present' : 'null'
      })
      prevClarify = req
    }
  })

  let prevApproval: unknown = null
  $approvalRequest.subscribe(req => {
    if (req !== prevApproval) {
      debugTrace('prompt', `approval ${prevApproval ? 'cleared' : 'raised'}`, {
        sessionId: $activeSessionId.get()
      })
      prevApproval = req
    }
  })

  let prevSudo: unknown = null
  $sudoRequest.subscribe(req => {
    if (req !== prevSudo) {
      debugTrace('prompt', `sudo ${prevSudo ? 'cleared' : 'raised'}`, {
        sessionId: $activeSessionId.get()
      })
      prevSudo = req
    }
  })

  let prevSecret: unknown = null
  $secretRequest.subscribe(req => {
    if (req !== prevSecret) {
      debugTrace('prompt', `secret ${prevSecret ? 'cleared' : 'raised'}`, {
        sessionId: $activeSessionId.get()
      })
      prevSecret = req
    }
  })

  // --- Sessions list length changes ---
  // Captures: new session created, session archived/deleted, sidebar merge
  // kept/dropped a row. Not per-field — just the count edge.
  let prevSessionsCount = -1
  $sessions.subscribe(list => {
    const count = list.length

    if (count !== prevSessionsCount) {
      debugTrace('sessions-list', `sessions count ${prevSessionsCount} → ${count}`)
      prevSessionsCount = count
    }
  })
}
