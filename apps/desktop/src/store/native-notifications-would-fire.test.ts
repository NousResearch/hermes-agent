// The presence reporter claims "this client can alert" from turnDoneNotificationWouldFire. That
// claim is only honest while the predicate and the real dispatch agree on every guard that is
// knowable before the turn ends.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type BrowserNotifyPayload, showBrowserNotification } from '@/lib/browser-notifications'

import {
  dispatchNativeNotification,
  setNativeNotifyEnabled,
  setNativeNotifyKind,
  turnDoneNotificationWouldFire
} from './native-notifications'
import { __resetNativeNotifyBaselineForTests, markNativeNotifyBaseline } from './notify-baseline'
import { setActiveSessionId } from './session'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const shown: string[] = []

class FakeNotification {
  static permission: NotificationPermission = 'granted'
  onclick: (() => void) | null = null
  constructor(title: string) {
    shown.push(title)
  }
  close() {}
}

let counter = 0

function setWindow({ focused, hidden }: { focused: boolean; hidden: boolean }) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'hasFocus', { configurable: true, value: () => focused })
}

beforeEach(() => {
  shown.length = 0
  vi.stubGlobal('Notification', FakeNotification)
  vi.stubGlobal('isSecureContext', true)
  setNativeNotifyEnabled(true)
  setNativeNotifyKind('turnDone', true)
  __resetNativeNotifyBaselineForTests()
})

afterEach(() => {
  delete desktopWindow.hermesDesktop
  delete document.documentElement.dataset.hermesDesktopHost
  setActiveSessionId(null)
  vi.unstubAllGlobals()
})

const cases = [
  // enabled, turnDone, quiet interval, focused, hidden, session is the active one
  [true, true, false, false, true, true],
  [true, true, false, false, false, true],
  [true, true, false, true, false, true],
  [true, true, false, false, true, false],
  [true, true, true, false, true, true],
  [false, true, false, false, true, true],
  [true, false, false, false, true, true]
] as const

describe('turnDoneNotificationWouldFire', () => {
  it.each(cases)(
    'Desktop: enabled %s, turnDone %s, quiet %s, focused %s, hidden %s, active %s agrees with dispatch',
    async (enabled, turnDone, quiet, focused, hidden, active) => {
      const supported = Promise.resolve(true)
      desktopWindow.hermesDesktop = {
        notify: vi.fn(async () => true),
        notifySupported: () => supported
      } as unknown as Window['hermesDesktop']
      const sessionId = `session-${(counter += 1)}`
      setNativeNotifyEnabled(enabled)
      setNativeNotifyKind('turnDone', turnDone)
      setActiveSessionId(active ? sessionId : 'another')
      setWindow({ focused, hidden })

      if (quiet) {
        markNativeNotifyBaseline()
      }

      // Main's answer arrives once per window; ask once so it is known.
      turnDoneNotificationWouldFire(sessionId)
      await supported
      const predicted = turnDoneNotificationWouldFire(sessionId)

      expect(dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })).toBe(predicted)
    }
  )

  it.each([
    ['supported', true],
    ['unsupported', false],
    ['not known yet', false]
  ] as const)('Desktop: host notifications %s -> %s', async (state, expected) => {
    let answer = (_supported: boolean) => undefined as void
    const reply = new Promise<boolean>(resolve => (answer = resolve))
    const notifySupported = vi.fn(() => reply)
    desktopWindow.hermesDesktop = {
      notify: vi.fn(async () => true),
      notifySupported
    } as unknown as Window['hermesDesktop']
    const sessionId = `session-${(counter += 1)}`
    setActiveSessionId(sessionId)
    setWindow({ focused: false, hidden: true })

    turnDoneNotificationWouldFire(sessionId)

    if (state !== 'not known yet') {
      answer(state === 'supported')
      await reply
    }

    expect(turnDoneNotificationWouldFire(sessionId)).toBe(expected)
    // Asked once per window, not once per prediction.
    expect(notifySupported).toHaveBeenCalledTimes(1)
  })

  it('without a bridge nothing can show the notification', () => {
    const sessionId = `session-${(counter += 1)}`
    setActiveSessionId(sessionId)
    setWindow({ focused: false, hidden: true })

    expect(turnDoneNotificationWouldFire(sessionId)).toBe(false)
  })

  it.each(['granted', 'denied', 'default'] as const)(
    'Webapp: live permission %s agrees with what the browser shows',
    permission => {
      document.documentElement.dataset.hermesDesktopHost = 'browser'
      desktopWindow.hermesDesktop = {
        notify: async (payload: BrowserNotifyPayload) => showBrowserNotification(payload)
      } as unknown as Window['hermesDesktop']
      FakeNotification.permission = permission
      const sessionId = `session-${(counter += 1)}`
      setActiveSessionId(sessionId)
      setWindow({ focused: false, hidden: true })

      const predicted = turnDoneNotificationWouldFire(sessionId)
      dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })

      expect(predicted).toBe(permission === 'granted')
      expect(shown.length > 0).toBe(predicted)
    }
  )
})
