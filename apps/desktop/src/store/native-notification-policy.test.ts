import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type NotificationLevel, registerNotificationLevelResolver } from './native-notification-policy'
import { dispatchNativeNotification, NATIVE_NOTIFICATION_KINDS, setNativeNotifyEnabled, setNativeNotifyKind } from './native-notifications'
import { __resetNativeNotifyBaselineForTests } from './notify-baseline'
import { setActiveSessionId } from './session'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const notify = vi.fn().mockResolvedValue(true)
let counter = 0
let unregister: () => void = () => undefined

function fresh(): string {
  counter += 1

  return `policy-session-${counter}`
}

/** A plugin that owns exactly one session and gives it `level`. */
function ownSession(sessionId: string, level: NotificationLevel) {
  unregister = registerNotificationLevelResolver(query => (query.runtimeSessionId === sessionId ? level : null))
}

beforeEach(() => {
  notify.mockClear()
  desktopWindow.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
  setNativeNotifyEnabled(true)

  for (const kind of NATIVE_NOTIFICATION_KINDS) {
    setNativeNotifyKind(kind, true)
  }

  // Away from the app, so every kind would fire for the active session.
  Object.defineProperty(document, 'hidden', { configurable: true, value: true })
  __resetNativeNotifyBaselineForTests()
})

afterEach(() => {
  unregister()
  desktopWindow.hermesDesktop = initialHermesDesktop
})

describe('per-owner notification level', () => {
  it("'needs-you' keeps blocking prompts and failures, drops completions", () => {
    const sid = fresh()
    setActiveSessionId(sid)
    ownSession(sid, 'needs-you')

    expect(dispatchNativeNotification({ kind: 'turnDone', sessionId: sid, title: 'done' })).toBe(false)
    expect(dispatchNativeNotification({ kind: 'approval', sessionId: sid, title: 'approve' })).toBe(true)
    expect(dispatchNativeNotification({ kind: 'turnError', sessionId: sid, title: 'failed' })).toBe(true)
  })

  it("'off' silences the owner's sessions, even a blocking prompt", () => {
    const sid = fresh()
    setActiveSessionId(sid)
    ownSession(sid, 'off')

    expect(dispatchNativeNotification({ kind: 'approval', sessionId: sid, title: 'approve' })).toBe(false)
    expect(notify).not.toHaveBeenCalled()
  })

  it("an owner the plugin doesn't claim, or an unregistered plugin, keeps today's behaviour", () => {
    const mine = fresh()
    const other = fresh()
    ownSession(mine, 'off')
    setActiveSessionId(other)

    expect(dispatchNativeNotification({ kind: 'turnDone', sessionId: other, title: 'done' })).toBe(true)

    unregister()
    setActiveSessionId(mine)
    expect(dispatchNativeNotification({ kind: 'turnDone', sessionId: mine, title: 'done' })).toBe(true)
  })
})
