import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as BrowserNotificationsModule from './browser-notifications'

type BrowserNotifications = typeof BrowserNotificationsModule

const created: { onclick: (() => void) | null; options?: NotificationOptions; title: string }[] = []

/** Chrome for Android's refusal, word for word. */
const MOBILE_REFUSAL =
  "Failed to construct 'Notification': Illegal constructor. Use ServiceWorkerRegistration.showNotification() instead."

class FakeNotification {
  static permission: NotificationPermission = 'default'
  static requestPermission = vi.fn(async () => FakeNotification.permission)
  /** Mobile browsers throw from the constructor. */
  static refuses = false
  onclick: (() => void) | null = null
  constructor(
    public title: string,
    public options?: NotificationOptions
  ) {
    if (FakeNotification.refuses) {
      throw new TypeError(MOBILE_REFUSAL)
    }

    created.push(this)
  }
  close() {}
}

class FakeWorker extends EventTarget {
  /** How many `statechange` listeners the page attached. */
  watchers = 0

  constructor(public state: ServiceWorkerState) {
    super()
  }

  override addEventListener(...args: Parameters<EventTarget['addEventListener']>) {
    this.watchers += args[0] === 'statechange' ? 1 : 0
    super.addEventListener(...args)
  }

  moveTo(state: ServiceWorkerState) {
    this.state = state
    this.dispatchEvent(new Event('statechange'))
  }
}

class FakeRegistration {
  active: FakeWorker | null = new FakeWorker('activated')
  installing: FakeWorker | null = null
  waiting: FakeWorker | null = null
  shown: { options?: NotificationOptions; title: string }[] = []

  async showNotification(title: string, options?: NotificationOptions) {
    // The browser's own rule: a registration shows nothing until a worker of it is active.
    if (!this.active) {
      throw new TypeError('No active registration available on the ServiceWorkerRegistration.')
    }

    this.shown.push({ options, title })
  }
}

/** A browser's `navigator.serviceWorker`, whose `register` answers with *registration*. */
function stubServiceWorker(registration: FakeRegistration) {
  const container = Object.assign(new EventTarget(), { register: vi.fn(async () => registration) })
  Object.defineProperty(window.navigator, 'serviceWorker', { configurable: true, value: container })

  return container
}

let notifications: BrowserNotifications

async function freshModule() {
  FakeNotification.permission = 'default'
  FakeNotification.refuses = false
  FakeNotification.requestPermission.mockClear()
  created.length = 0
  vi.stubGlobal('Notification', FakeNotification)
  vi.stubGlobal('isSecureContext', true)
  vi.resetModules()
  notifications = await import('./browser-notifications')
}

function restoreGlobals() {
  vi.unstubAllGlobals()
  Reflect.deleteProperty(window.navigator, 'serviceWorker')
}

describe('showBrowserNotification', () => {
  beforeEach(freshModule)
  afterEach(restoreGlobals)

  it('never asks for permission from an event (only a click in Settings may)', async () => {
    const container = stubServiceWorker(new FakeRegistration())

    expect(
      await notifications.showBrowserNotification({ kind: 'approval', sessionId: 's1', title: 'Approval needed' })
    ).toBe(false)
    expect(FakeNotification.requestPermission).not.toHaveBeenCalled()
    expect(container.register).not.toHaveBeenCalled()
    expect(created).toHaveLength(0)
  })

  it('shows the notification through the service worker where the constructor throws, as on mobile', async () => {
    FakeNotification.permission = 'granted'
    FakeNotification.refuses = true
    const registration = new FakeRegistration()
    const container = stubServiceWorker(registration)

    expect(
      await notifications.showBrowserNotification({
        body: 'rm -rf build',
        focusSessionId: 'stored-1',
        kind: 'approval',
        sessionId: 'rt-1',
        title: 'Approval needed'
      })
    ).toBe(true)

    expect(container.register).toHaveBeenCalledWith('/notification-worker.js', { scope: '/' })
    expect(registration.shown).toEqual([
      {
        options: { body: 'rm -rf build', data: { focusSessionId: 'stored-1' }, tag: 'hermes:approval:stored-1' },
        title: 'Approval needed'
      }
    ])
  })

  it('reports false, and never throws, where neither a worker nor the constructor can show it', async () => {
    FakeNotification.permission = 'granted'
    FakeNotification.refuses = true

    expect(await notifications.showBrowserNotification({ kind: 'turnDone', title: 'Done' })).toBe(false)
  })

  it('waits for a just-registered worker to activate before showing', async () => {
    FakeNotification.permission = 'granted'
    const registration = new FakeRegistration()
    const installing = new FakeWorker('installing')
    registration.active = null
    registration.installing = installing
    stubServiceWorker(registration)

    const shown = notifications.showBrowserNotification({ kind: 'turnDone', sessionId: 's1', title: 'Done' })
    await vi.waitFor(() => expect(installing.watchers).toBe(1))
    installing.moveTo('installed')
    registration.active = installing
    registration.installing = null
    installing.moveTo('activating')
    installing.moveTo('activated')

    expect(await shown).toBe(true)
    expect(registration.shown.map(entry => entry.options?.tag)).toEqual(['hermes:turnDone:s1'])
  })

  it('registers the worker again on the next event after a failed registration', async () => {
    FakeNotification.permission = 'granted'
    FakeNotification.refuses = true
    const registration = new FakeRegistration()
    const container = stubServiceWorker(registration)
    container.register.mockRejectedValueOnce(new TypeError('Failed to register a ServiceWorker: network error'))

    expect(await notifications.showBrowserNotification({ kind: 'turnDone', title: 'First' })).toBe(false)
    expect(await notifications.showBrowserNotification({ kind: 'turnDone', title: 'Second' })).toBe(true)
    expect(container.register).toHaveBeenCalledTimes(2)
    expect(registration.shown.map(entry => entry.title)).toEqual(['Second'])
  })

  it('falls back to the Notification constructor in a browser that runs no service worker', async () => {
    FakeNotification.permission = 'granted'
    const opened: string[] = []
    const off = notifications.onBrowserFocusSession(id => opened.push(id))
    const focus = vi.spyOn(window, 'focus').mockImplementation(() => undefined)

    expect(
      await notifications.showBrowserNotification({
        focusSessionId: 'stored-1',
        kind: 'approval',
        sessionId: 'rt-1',
        title: 'Approval needed'
      })
    ).toBe(true)
    expect(created[0].options?.tag).toBe('hermes:approval:stored-1')
    created[0].onclick?.()

    expect(focus).toHaveBeenCalled()
    expect(opened).toEqual(['stored-1'])
    off()
  })
})

describe('installBrowserNotifications', () => {
  beforeEach(freshModule)
  afterEach(restoreGlobals)

  it('registers the worker at the Webapp base path once notifications are allowed, not before', async () => {
    const container = stubServiceWorker(new FakeRegistration())

    notifications.installBrowserNotifications('/hermes')
    expect(container.register).not.toHaveBeenCalled()

    FakeNotification.permission = 'granted'
    expect(await notifications.requestBrowserNotificationPermission()).toBe('granted')
    expect(container.register).toHaveBeenCalledWith('/hermes/notification-worker.js', { scope: '/hermes/' })
  })

  it('registers the worker at boot when the user allowed notifications earlier', () => {
    FakeNotification.permission = 'granted'
    const container = stubServiceWorker(new FakeRegistration())

    notifications.installBrowserNotifications('')

    expect(container.register).toHaveBeenCalledWith('/notification-worker.js', { scope: '/' })
  })

  it('opens the session a notification click routes back through the worker', () => {
    const container = stubServiceWorker(new FakeRegistration())
    const opened: string[] = []
    const off = notifications.onBrowserFocusSession(id => opened.push(id))

    notifications.installBrowserNotifications('')
    container.dispatchEvent(
      new MessageEvent('message', { data: { sessionId: 'stored-1', type: 'hermes:focus-session' } })
    )
    container.dispatchEvent(new MessageEvent('message', { data: { sessionId: 'stored-2', type: 'other' } }))
    container.dispatchEvent(new MessageEvent('message', { data: { type: 'hermes:focus-session' } }))

    expect(opened).toEqual(['stored-1'])
    off()
  })
})
