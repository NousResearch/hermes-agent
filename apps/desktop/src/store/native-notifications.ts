import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

import { $gateway } from './gateway'
import { clearApprovalRequest } from './prompts'
import { $activeSessionId } from './session'
import { $splitPaneRuntimeSessionId, mainPaneRuntimeSessionId } from './split'

// Native OS notifications (Electron `Notification`), separate from the in-app
// toast feed in `notifications.ts`. Each kind toggles independently.
export type NativeNotificationKind = 'approval' | 'backgroundDone' | 'input' | 'turnDone' | 'turnError'

export const NATIVE_NOTIFICATION_KINDS: readonly NativeNotificationKind[] = [
  'approval',
  'input',
  'turnDone',
  'turnError',
  'backgroundDone'
]

// Blocking prompts — surface even while focused if they're for another session.
const ATTENTION_KINDS = new Set<NativeNotificationKind>(['approval', 'input'])

export interface NativeNotificationPrefs {
  enabled: boolean
  kinds: Record<NativeNotificationKind, boolean>
}

const STORAGE_KEY = 'hermes:native-notifications'

const DEFAULT_PREFS: NativeNotificationPrefs = {
  enabled: true,
  kinds: { approval: true, backgroundDone: true, input: true, turnDone: true, turnError: true }
}

function readPrefs(): NativeNotificationPrefs {
  const raw = storedString(STORAGE_KEY)

  if (!raw) {
    return DEFAULT_PREFS
  }

  try {
    const parsed = JSON.parse(raw) as Partial<NativeNotificationPrefs>
    const kinds = { ...DEFAULT_PREFS.kinds }

    for (const kind of NATIVE_NOTIFICATION_KINDS) {
      const value = parsed.kinds?.[kind]

      if (typeof value === 'boolean') {
        kinds[kind] = value
      }
    }

    return {
      enabled: typeof parsed.enabled === 'boolean' ? parsed.enabled : DEFAULT_PREFS.enabled,
      kinds
    }
  } catch {
    return DEFAULT_PREFS
  }
}

export const $nativeNotifyPrefs = atom<NativeNotificationPrefs>(readPrefs())

function writePrefs(next: NativeNotificationPrefs) {
  $nativeNotifyPrefs.set(next)
  persistString(STORAGE_KEY, JSON.stringify(next))
}

export function setNativeNotifyEnabled(enabled: boolean) {
  writePrefs({ ...$nativeNotifyPrefs.get(), enabled })
}

export function setNativeNotifyKind(kind: NativeNotificationKind, on: boolean) {
  const prev = $nativeNotifyPrefs.get()
  writePrefs({ ...prev, kinds: { ...prev.kinds, [kind]: on } })
}

// De-dupe replayed events for the same kind+session. Self-evicting: entries
// older than the window are pruned on every dispatch, so the map can't grow.
const THROTTLE_MS = 1000
const lastFiredAt = new Map<string, number>()

function throttled(key: string, now: number): boolean {
  for (const [k, at] of lastFiredAt) {
    if (now - at >= THROTTLE_MS) {
      lastFiredAt.delete(k)
    }
  }

  if (lastFiredAt.has(key)) {
    return true
  }

  lastFiredAt.set(key, now)

  return false
}

// "Backgrounded" = the user isn't on Hermes. `document.hidden` only flips when
// minimized/occluded; an alt-tabbed window is visible-but-unfocused, so we also
// check `document.hasFocus()`.
function isBackgrounded(): boolean {
  if (typeof document === 'undefined') {
    return false
  }

  if (document.hidden) {
    return true
  }

  return typeof document.hasFocus === 'function' && !document.hasFocus()
}

// A session shown in either chat pane counts as foreground: the split pane's
// transcript is exactly as on-screen as the main one. With the split closed
// ($splitPaneRuntimeSessionId null, no divergent main-pane id) this reduces to
// the pre-split active check. While the split is FOCUSED, $activeSessionId
// mirrors the SPLIT's session — the main pane's on-screen transcript is then
// only reachable through the controller-registered unmirrored getter, so both
// pane ids are consulted.
function isPaneVisibleSession(sessionId: null | string | undefined): boolean {
  if (!sessionId) {
    return false
  }

  if (sessionId === $activeSessionId.get()) {
    return true
  }

  const splitRuntimeId = $splitPaneRuntimeSessionId.get()

  if (splitRuntimeId !== null && sessionId === splitRuntimeId) {
    return true
  }

  const mainRuntimeId = mainPaneRuntimeSessionId()

  return mainRuntimeId !== null && sessionId === mainRuntimeId
}

function shouldFire(kind: NativeNotificationKind, sessionId?: null | string, global = false): boolean {
  // Global notifications aren't tied to a chat session (e.g. pet generation,
  // which runs from the command center with no active conversation). They fire
  // whenever the user is away, with no session-match requirement — otherwise a
  // background run started without an open session would be silently dropped.
  if (global) {
    return isBackgrounded()
  }

  // Attention kinds break through for an off-screen session even while focused.
  if (ATTENTION_KINDS.has(kind)) {
    return isBackgrounded() || (Boolean(sessionId) && !isPaneVisibleSession(sessionId))
  }

  // Completion kinds: only a pane-visible session, only while away — so a busy
  // gateway (messaging, kanban, cron) can't spam a toast per background session.
  return isBackgrounded() && isPaneVisibleSession(sessionId)
}

export interface NativeNotificationAction {
  id: string
  text: string
}

export interface NativeNotificationInput {
  kind: NativeNotificationKind
  title: string
  body?: string
  sessionId?: null | string
  /**
   * Not tied to a chat session (e.g. pet generation). Fires whenever the user
   * is away, bypassing the session-match gate that completion kinds normally
   * require.
   */
  global?: boolean
  silent?: boolean
  actions?: NativeNotificationAction[]
}

export function dispatchNativeNotification(input: NativeNotificationInput): void {
  const prefs = $nativeNotifyPrefs.get()

  if (!prefs.enabled || !prefs.kinds[input.kind]) {
    return
  }

  if (!shouldFire(input.kind, input.sessionId, input.global)) {
    return
  }

  if (throttled(`${input.kind}:${input.sessionId ?? (input.global ? 'global' : '')}`, Date.now())) {
    return
  }

  void window.hermesDesktop?.notify({
    actions: input.actions,
    body: input.body,
    kind: input.kind,
    sessionId: input.sessionId ?? undefined,
    silent: input.silent,
    title: input.title
  })
}

// Resolve a pending approval from a notification button, mirroring the in-app
// Run/Reject bar. Keyed by session id — a background approval has no local guard.
export async function respondToApprovalAction(sessionId: null | string, actionId: string): Promise<void> {
  const choice = actionId === 'approve' ? 'once' : actionId === 'reject' ? 'deny' : null

  if (!choice) {
    return
  }

  const gateway = $gateway.get()

  if (!gateway) {
    return
  }

  try {
    await gateway.request('approval.respond', { choice, session_id: sessionId ?? undefined })
    clearApprovalRequest(sessionId)
  } catch {
    // Leave the prompt parked so the user can still resolve it in-app.
  }
}

// Settings "send test" — bypasses gating. Returns whether the OS accepted it so
// the panel can flag a silent permission failure instead of looking dead.
export async function sendTestNativeNotification(title: string, body: string): Promise<boolean> {
  const bridge = window.hermesDesktop

  if (!bridge?.notify) {
    return false
  }

  try {
    return await bridge.notify({ body, kind: 'turnDone', title })
  } catch {
    return false
  }
}
