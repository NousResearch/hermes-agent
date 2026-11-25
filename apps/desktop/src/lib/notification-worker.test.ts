import { afterEach, describe, expect, it, vi } from 'vitest'

/**
 * The notification worker ships as a classic script at the root of the Webapp's dist
 * (apps/desktop/public). These tests run it the way a browser does, against a stand-in for its
 * ServiceWorkerGlobalScope, and click the notifications it shows.
 */
const WORKER_SCRIPT = '../../public/notification-worker.js'
const SCOPE = 'https://hermes.example/hermes/'
const TAB_QUERY = 'hermes:tab-query'

class FakeWindowClient {
  focused = 0
  messages: unknown[] = []
  queries = 0

  /** *id*: what the tab answers when the worker asks for the id its notifications carry; null never answers. */
  constructor(
    public url: string,
    public id: null | string = null
  ) {}

  async focus() {
    this.focused += 1

    return this
  }

  postMessage(message: unknown, ports: readonly MessagePort[] = []) {
    if ((message as { type?: unknown }).type === TAB_QUERY) {
      this.queries += 1

      if (this.id !== null) {
        ports[0]?.postMessage(this.id)
      }

      return
    }

    this.messages.push(message)
  }
}

/** A message from the worker as the page's container receives it, transferred ports included. */
function workerMessage(data: unknown, ports: readonly MessagePort[] = []): Event {
  const event = new Event('message')
  Object.defineProperties(event, { data: { value: data }, ports: { value: ports } })

  return event
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
    ['https://hermes.example/hermes/index.html?profile=p2', true],
    ['https://hermes.example/hermes/login', false],
    ['https://hermes.example/hermes/webapp/launch', false],
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

  it('passes over a sign-in page used since for the Webapp tab', async () => {
    const signIn = new FakeWindowClient('https://hermes.example/hermes/login?next=%2Fhermes%2F')
    const tab = new FakeWindowClient(SCOPE)
    const worker = await startWorker([signIn, tab])

    await worker.click({ focusSessionId: 'stored-1' })

    expect([tab.focused, tab.messages]).toEqual([1, [{ sessionId: 'stored-1', type: 'hermes:focus-session' }]])
    expect([signIn.focused, signIn.messages]).toEqual([0, []])
  })

  it('asks each open tab which one showed the notification, and opens the session there', async () => {
    // Both on the same profile by their addresses, and one used since: only the tabs can tell.
    const other = new FakeWindowClient(SCOPE, 'tab-b')
    const notifier = new FakeWindowClient(SCOPE, 'tab-a')
    const worker = await startWorker([other, notifier])

    await worker.click({ focusSessionId: 'stored-1', profile: 'research', tab: 'tab-a' })

    expect([notifier.focused, notifier.messages]).toEqual([
      1,
      [{ sessionId: 'stored-1', type: 'hermes:focus-session' }]
    ])
    expect([other.focused, other.messages]).toEqual([0, []])
    expect(worker.clients.openWindow).not.toHaveBeenCalled()
  })

  it('prefers a tab that does not answer, which may have shown it, then the tab used last', async () => {
    const other = new FakeWindowClient(SCOPE, 'tab-b')
    const frozen = new FakeWindowClient(SCOPE, null)
    const third = new FakeWindowClient(SCOPE, 'tab-c')
    const worker = await startWorker([other, frozen, third])

    await worker.click({ focusSessionId: 'stored-1', profile: 'research', tab: 'tab-a' })
    frozen.id = 'tab-d'
    await worker.click({ focusSessionId: 'stored-2', profile: 'research', tab: 'tab-a' })

    expect(frozen.messages).toEqual([{ sessionId: 'stored-1', type: 'hermes:focus-session' }])
    expect(other.messages).toEqual([{ sessionId: 'stored-2', type: 'hermes:focus-session' }])
    expect([third.focused, third.messages]).toEqual([0, []])
    expect(worker.clients.openWindow).not.toHaveBeenCalled()
  })

  it('asks no tab when there is no choice to make: a lone tab, or a click that names no session', async () => {
    const lone = new FakeWindowClient(SCOPE, 'tab-b')
    const worker = await startWorker([lone])

    await worker.click({ focusSessionId: 'stored-1', profile: 'research', tab: 'tab-a' })

    expect([lone.queries, lone.messages]).toEqual([0, [{ sessionId: 'stored-1', type: 'hermes:focus-session' }]])

    const first = new FakeWindowClient(SCOPE, 'tab-b')
    const second = new FakeWindowClient(SCOPE, 'tab-a')
    const sessionless = await startWorker([first, second])

    await sessionless.click({ profile: 'research', tab: 'tab-a' })

    expect([first.focused, first.queries, second.focused, second.queries]).toEqual([1, 0, 0, 0])
  })

  it("opens the Webapp on the notification's profile when none of its tabs is open", async () => {
    const worker = await startWorker([new FakeWindowClient('https://hermes.example/other/')])

    await worker.click({ focusSessionId: 'stored 1', profile: 'research' })
    await worker.click({ profile: 'default' })

    expect(worker.clients.openWindow.mock.calls).toEqual([
      [`${SCOPE}?profile=research#/stored%201`],
      [`${SCOPE}?profile=default`]
    ])
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

  it('brings a click back to the tab that showed the notification', async () => {
    vi.stubGlobal('Notification', { permission: 'granted' })
    vi.stubGlobal('isSecureContext', true)
    const shown: NotificationOptions[] = []

    const registration = {
      active: {},
      showNotification: async (_title: string, options: NotificationOptions) => void shown.push(options)
    }

    const container = Object.assign(new EventTarget(), { register: vi.fn(async () => registration) })
    Object.defineProperty(window.navigator, 'serviceWorker', { configurable: true, value: container })
    // This tab, and another Webapp tab used since.
    const other = new FakeWindowClient(SCOPE, 'another-tab')
    const tab = new FakeWindowClient(SCOPE)
    tab.postMessage = (message, ports = []) => container.dispatchEvent(workerMessage(message, ports))
    const worker = await startWorker([other, tab])
    const page = await import('./browser-notifications')
    const opened: string[] = []
    const off = page.onBrowserFocusSession(id => opened.push(id))
    page.installBrowserNotifications('/hermes', () => 'research')

    expect(
      await page.showBrowserNotification({ focusSessionId: 'stored-1', kind: 'approval', title: 'Approval' })
    ).toBe(true)
    await worker.click(shown[0].data)

    expect(opened).toEqual(['stored-1'])
    expect([other.focused, other.messages]).toEqual([0, []])
    off()
  })
})
