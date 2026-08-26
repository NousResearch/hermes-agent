import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from './gateway'
import {
  clearPluginNotifyHandlers,
  dispatchNativeNotification,
  dispatchPluginNativeNotification,
  invokePluginNotifyAction,
  invokePluginNotifyActivate,
  NATIVE_NOTIFICATION_KINDS,
  respondToApprovalAction,
  sendTestNativeNotification,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from './native-notifications'
import { __resetNativeNotifyBaselineForTests, markNativeNotifyBaseline } from './notify-baseline'
import { $approvalRequest, setApprovalRequest } from './prompts'
import { $activeSessionId, setActiveSessionId } from './session'
import { markSessionGone, resetBackgroundPollingGuard } from './runtime-gone'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const notify = vi.fn().mockResolvedValue(true)

function setWindowState({ focused = true, hidden = false }: { focused?: boolean; hidden?: boolean }) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'hasFocus', { configurable: true, value: () => focused })
}

let counter = 0

// Unique session id per call dodges the per-(kind,session) throttle so each
// assertion starts clean.
function freshSession(): string {
  counter += 1

  return `session-${counter}`
}

beforeEach(() => {
  notify.mockClear()
  desktopWindow.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
  setNativeNotifyEnabled(true)

  for (const kind of NATIVE_NOTIFICATION_KINDS) {
    setNativeNotifyKind(kind, true)
  }

  setActiveSessionId(null)
  resetBackgroundPollingGuard()
  setWindowState({ focused: false, hidden: true })
  __resetNativeNotifyBaselineForTests()
})

afterEach(() => {
  clearPluginNotifyHandlers()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }

  resetBackgroundPollingGuard()
})

describe('dispatchNativeNotification focus gating', () => {
  it('fires a completion notification for the active session when the window is hidden', () => {
    const sessionId = freshSession()
    setActiveSessionId(sessionId)
    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })
    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('fires a completion notification when the window is visible but unfocused (alt-tab)', () => {
    const sessionId = freshSession()
    setActiveSessionId(sessionId)
    setWindowState({ focused: false, hidden: false })
    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })
    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('suppresses a completion notification when the window is focused', () => {
    const sessionId = freshSession()
    setActiveSessionId(sessionId)
    setWindowState({ focused: true, hidden: false })
    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('suppresses a completion notification for a non-active background session (no gateway spam)', () => {
    setActiveSessionId('on-screen')
    dispatchNativeNotification({ kind: 'turnDone', sessionId: 'busy-bot-session', title: 'done' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('fires an attention notification for an off-screen session even when focused', () => {
    setWindowState({ focused: true, hidden: false })
    setActiveSessionId('on-screen')
    dispatchNativeNotification({ kind: 'approval', sessionId: 'background', title: 'approve' })
    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('suppresses an attention notification for the active session when focused', () => {
    setWindowState({ focused: true, hidden: false })
    setActiveSessionId('on-screen')
    dispatchNativeNotification({ kind: 'approval', sessionId: 'on-screen', title: 'approve' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('fires a global completion notification while away with no active session (pet gen)', () => {
    setActiveSessionId(null)
    dispatchNativeNotification({ global: true, kind: 'backgroundDone', title: 'Your pet hatched' })
    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('suppresses a global notification when the window is focused', () => {
    setWindowState({ focused: true, hidden: false })
    setActiveSessionId(null)
    dispatchNativeNotification({ global: true, kind: 'backgroundDone', title: 'Your pet hatched' })
    expect(notify).not.toHaveBeenCalled()
  })
})

describe('dispatchNativeNotification preferences', () => {
  it('suppresses everything when the master switch is off', () => {
    setNativeNotifyEnabled(false)
    dispatchNativeNotification({ kind: 'approval', sessionId: freshSession(), title: 'approve' })
    dispatchNativeNotification({ kind: 'turnDone', sessionId: freshSession(), title: 'done' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('suppresses only the disabled kind', () => {
    const sessionId = freshSession()
    setActiveSessionId(sessionId)
    setNativeNotifyKind('turnDone', false)
    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })
    expect(notify).not.toHaveBeenCalled()

    dispatchNativeNotification({ kind: 'turnError', sessionId, title: 'boom' })
    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('forwards kind and sessionId to the bridge', () => {
    setActiveSessionId('abc')
    dispatchNativeNotification({ body: 'hi', kind: 'turnError', sessionId: 'abc', title: 'boom' })
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ body: 'hi', kind: 'turnError', sessionId: 'abc', title: 'boom' })
    )
  })
})

describe('dispatchNativeNotification post-connect baseline', () => {
  it('suppresses a prompt replayed right after a socket opens', () => {
    markNativeNotifyBaseline()
    dispatchNativeNotification({ kind: 'approval', sessionId: freshSession(), title: 'approve' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('suppresses a completion replayed right after a socket opens', () => {
    const sessionId = freshSession()
    setActiveSessionId(sessionId)
    markNativeNotifyBaseline()
    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('fires again once the window has passed', () => {
    vi.useFakeTimers()

    try {
      markNativeNotifyBaseline()
      vi.advanceTimersByTime(5000)
      dispatchNativeNotification({ kind: 'approval', sessionId: freshSession(), title: 'approve' })
      expect(notify).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('dispatchPluginNativeNotification', () => {
  it('fires while the user is away and tags the plugin id for dedupe', () => {
    dispatchPluginNativeNotification('index-network', { body: 'New match', title: 'Opportunity' })
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ body: 'New match', kind: 'plugin', tag: 'index-network', title: 'Opportunity' })
    )
  })

  it('suppresses while the window is focused (the in-app toast covers foreground)', () => {
    setWindowState({ focused: true, hidden: false })
    dispatchPluginNativeNotification('focused-plugin', { title: 'Opportunity' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('is gated by the "plugin" kind preference', () => {
    setNativeNotifyKind('plugin', false)
    dispatchPluginNativeNotification('muted-plugin', { title: 'Opportunity' })
    expect(notify).not.toHaveBeenCalled()
  })

  it('throttles per plugin, so two plugins cannot collapse each other', () => {
    dispatchPluginNativeNotification('plugin-a', { title: 'a' })
    dispatchPluginNativeNotification('plugin-a', { title: 'a again' })
    dispatchPluginNativeNotification('plugin-b', { title: 'b' })
    expect(notify).toHaveBeenCalledTimes(2)
  })

  it('does not register handlers for throttled or suppressed notifications', () => {
    const onActivate = vi.fn()

    // First fires and registers; the immediate repeat is throttled per plugin id.
    dispatchPluginNativeNotification('leak-plugin', { onActivate: () => undefined, title: 'first' })
    dispatchPluginNativeNotification('leak-plugin', { onActivate, title: 'throttled' })
    expect(notify).toHaveBeenCalledTimes(1)

    // The throttled call must not have registered anything: no notifyId ever
    // reached the OS, so its handlers would leak. Invoking with the only
    // minted id (from the first call) must not hit the throttled callback.
    const payload = notify.mock.calls[0]?.[0] as { notifyId?: string }
    invokePluginNotifyActivate(payload.notifyId)
    expect(onActivate).not.toHaveBeenCalled()

    // Fully suppressed (kind disabled): nothing registered either.
    setNativeNotifyKind('plugin', false)
    const suppressed = vi.fn()
    dispatchPluginNativeNotification('other-plugin', { onActivate: suppressed, title: 'muted' })
    expect(notify).toHaveBeenCalledTimes(1)
    invokePluginNotifyActivate(payload.notifyId)
    expect(suppressed).not.toHaveBeenCalled()
  })

  it('forwards icon, resolved activate path, and action buttons (deeplink-compatible)', () => {
    // Unique tag (throttle is per plugin id); activate still uses the plugin deep link.
    dispatchPluginNativeNotification('index-network-alerts', {
      actions: [
        { id: 'open', label: 'Open', activate: 'hermes://index-network/intent/1' },
        { id: 'dismiss', label: 'Dismiss', onAction: () => undefined }
      ],
      activate: 'hermes://index-network/intent/1',
      body: 'New match',
      icon: '/tmp/index-network.png',
      title: 'Opportunity'
    })

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        activate: '/index-network/intent/1',
        actions: [
          { activate: '/index-network/intent/1', id: 'open', text: 'Open' },
          { activate: undefined, id: 'dismiss', text: 'Dismiss' }
        ],
        icon: '/tmp/index-network.png',
        kind: 'plugin',
        notifyId: expect.stringMatching(/^index-network-alerts:/),
        tag: 'index-network-alerts',
        title: 'Opportunity'
      })
    )
  })

  it('registers onActivate / onAction handlers keyed by notifyId', () => {
    const onActivate = vi.fn()
    const onAction = vi.fn()

    dispatchPluginNativeNotification('handlers-plugin', {
      activate: 'hermes://index-network/intent/1',
      onActivate,
      actions: [{ id: 'dismiss', label: 'Dismiss', onAction }],
      title: 'Opportunity'
    })

    const payload = notify.mock.calls[0]?.[0] as { notifyId?: string }
    expect(payload.notifyId).toBeTruthy()
    invokePluginNotifyActivate(payload.notifyId)
    expect(onActivate).toHaveBeenCalledTimes(1)
    expect(invokePluginNotifyAction(payload.notifyId, 'dismiss')).toBe(true)
    expect(onAction).toHaveBeenCalledTimes(1)
  })
})

describe('dispatchNativeNotification throttle', () => {
  it('collapses duplicate kind+session within the throttle window', () => {
    const sessionId = freshSession()
    setActiveSessionId(sessionId)
    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })
    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done again' })
    expect(notify).toHaveBeenCalledTimes(1)
  })
})

describe('sendTestNativeNotification', () => {
  it('fires regardless of focus or active session', () => {
    setWindowState({ focused: true, hidden: false })
    setActiveSessionId('on-screen')
    sendTestNativeNotification('Hermes', 'works')
    expect(notify).toHaveBeenCalledTimes(1)
  })
})

describe('$activeSessionId wiring', () => {
  it('reflects the setter used for gating', () => {
    setActiveSessionId('xyz')
    expect($activeSessionId.get()).toBe('xyz')
  })
})

describe('respondToApprovalAction', () => {
  const request = vi.fn().mockResolvedValue({ resolved: true })

  beforeEach(() => {
    request.mockClear()
    $gateway.set({ request } as unknown as ReturnType<typeof $gateway.get>)
  })

  afterEach(() => {
    $gateway.set(null)
  })

  it('approves via approval.respond {choice: "once"} and clears the prompt', async () => {
    setActiveSessionId('bg')
    setApprovalRequest({ command: 'rm -rf /', description: 'dangerous', sessionId: 'bg' })

    await respondToApprovalAction('bg', 'approve')

    expect(request).toHaveBeenCalledWith('approval.respond', { choice: 'once', session_id: 'bg' })
    expect($approvalRequest.get()).toBeNull()
  })

  it('rejects via approval.respond {choice: "deny"}', async () => {
    await respondToApprovalAction('bg', 'reject')
    expect(request).toHaveBeenCalledWith('approval.respond', { choice: 'deny', session_id: 'bg' })
  })

  it('ignores unknown action ids', async () => {
    await respondToApprovalAction('bg', 'snooze')
    expect(request).not.toHaveBeenCalled()
  })

  it('no-ops without a gateway', async () => {
    $gateway.set(null)
    await respondToApprovalAction('bg', 'approve')
    expect(request).not.toHaveBeenCalled()
  })

  it('does not retry an approval action for a runtime already marked gone', async () => {
    markSessionGone('bg')

    await respondToApprovalAction('bg', 'approve')

    expect(request).not.toHaveBeenCalled()
  })
})
