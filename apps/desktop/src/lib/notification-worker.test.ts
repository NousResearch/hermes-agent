import { afterEach, describe, expect, it, vi } from 'vitest'

/**
 * The notification worker ships as a classic script at the root of the Webapp's dist
 * (apps/desktop/public). These tests run it the way a browser does, against a stand-in for its
 * ServiceWorkerGlobalScope, and click the notifications it shows.
 */
const WORKER_SCRIPT = '../../public/notification-worker.js'
const SCOPE = 'https://hermes.example/hermes/'

class FakeWindowClient {
  focused = 0
  messages: unknown[] = []

  constructor(public url: string) {}

  async focus() {
    this.focused += 1

    return this
  }

  postMessage(message: unknown) {
    this.messages.push(message)
  }
}

type Listener = (event: unknown) => void

async function startWorker(windows: FakeWindowClient[]) {
  const listeners = new Map<string, Listener>()

  const clients = {
    matchAll: vi.fn(async () => windows),
    openWindow: vi.fn(async () => null)
  }

  vi.stubGlobal('self', {
    addEventListener: (type: string, listener: Listener) => listeners.set(type, listener),
    clients,
    registration: { scope: SCOPE },
    skipWaiting: vi.fn(async () => undefined)
  })
  vi.resetModules()
  await import(/* @vite-ignore */ WORKER_SCRIPT)

  /** The browser's dispatch of a click on a notification carrying *data*. */
  const click = async (data?: unknown) => {
    const pending: Promise<unknown>[] = []
    const notification = { close: vi.fn(), data }
    const event = { notification, waitUntil: (promise: Promise<unknown>) => pending.push(promise) }

    listeners.get('notificationclick')?.(event)
    await Promise.all(pending)

    return notification
  }

  return { clients, click }
}

afterEach(() => {
  vi.unstubAllGlobals()
  Reflect.deleteProperty(window.navigator, 'serviceWorker')
})

describe('notification worker', () => {
  it('brings the Webapp tab used last forward and asks it to open the session', async () => {
    const otherApp = new FakeWindowClient('https://hermes.example/other/')
    const last = new FakeWindowClient('https://hermes.example/hermes/?profile=p2#/abc')
    const earlier = new FakeWindowClient('https://hermes.example/hermes/')
    const worker = await startWorker([otherApp, last, earlier])

    const notification = await worker.click({ focusSessionId: 'stored-1' })

    expect(notification.close).toHaveBeenCalled()
    // A tab opened before the worker registered is not controlled by it and must still count.
    expect(worker.clients.matchAll).toHaveBeenCalledWith({ includeUncontrolled: true, type: 'window' })
    expect(last.focused).toBe(1)
    expect(last.messages).toEqual([{ sessionId: 'stored-1', type: 'hermes:focus-session' }])
    expect([otherApp.focused, earlier.focused, otherApp.messages, earlier.messages]).toEqual([0, 0, [], []])
    expect(worker.clients.openWindow).not.toHaveBeenCalled()
  })

  it.each([
    ['https://hermes.example/hermes', true],
    ['https://hermes.example/hermes?profile=p2', true],
    ['https://hermes.example/hermes/login', true],
    ['https://hermes.example/hermesx/', false],
    ['https://hermes.example/', false]
  ])('counts %s as a Webapp tab: %s', async (url, webapp) => {
    const tab = new FakeWindowClient(url)
    const worker = await startWorker([tab])

    await worker.click({ focusSessionId: 'stored-1' })

    expect(tab.focused).toBe(webapp ? 1 : 0)
    expect(worker.clients.openWindow).toHaveBeenCalledTimes(webapp ? 0 : 1)
  })

  it("opens the Webapp at the notification's chat when none of its tabs is open", async () => {
    const otherApp = new FakeWindowClient('https://hermes.example/other/')
    const worker = await startWorker([otherApp])

    await worker.click({ focusSessionId: 'stored 1' })

    expect(worker.clients.openWindow).toHaveBeenCalledWith(`${SCOPE}#/stored%201`)
    expect(otherApp.focused).toBe(0)
  })

  it('opens the Webapp at its base path when no tab is open and the notification names no session', async () => {
    const worker = await startWorker([])

    await worker.click({})

    expect(worker.clients.openWindow).toHaveBeenCalledWith(SCOPE)
  })

  it('only brings the tab forward when the notification names no session', async () => {
    const tab = new FakeWindowClient(SCOPE)
    const worker = await startWorker([tab])

    await worker.click({ focusSessionId: undefined })
    await worker.click(undefined)

    expect(tab.focused).toBe(2)
    expect(tab.messages).toEqual([])
  })

  it("opens the clicked notification's session in the Webapp tab", async () => {
    vi.stubGlobal('Notification', { permission: 'default' })
    vi.stubGlobal('isSecureContext', true)
    const container = Object.assign(new EventTarget(), { register: vi.fn() })
    Object.defineProperty(window.navigator, 'serviceWorker', { configurable: true, value: container })
    const tab = new FakeWindowClient(SCOPE)
    tab.postMessage = message => container.dispatchEvent(new MessageEvent('message', { data: message }))
    const worker = await startWorker([tab])
    const page = await import('./browser-notifications')
    const opened: string[] = []
    const off = page.onBrowserFocusSession(id => opened.push(id))
    page.installBrowserNotifications('/hermes')

    await worker.click({ focusSessionId: 'stored-1' })

    expect(opened).toEqual(['stored-1'])
    off()
  })
})
