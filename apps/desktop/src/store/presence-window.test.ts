// The window's reporter through its real seams: the HashRouter route and session stores decide what
// is in view, the composer's owner ladder and the gateway registry decide which connection owns it,
// and real HermesGateway clients over in-memory sockets carry the frames.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/api/client'
import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { captureSubmissionDestination } from '@/app/session/hooks/use-prompt-actions/submission-destination'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'

import { $browserNotificationPermission } from './browser-notification-permission'
import { $gateway, setPrimaryGateway, setPrimaryGatewayConnectionId } from './gateway'
import { setNativeNotifyEnabled, setNativeNotifyKind } from './native-notifications'
import { __resetNativeNotifyBaselineForTests, markNativeNotifyBaseline } from './notify-baseline'
import { createPresenceReporter, startPresenceReporter } from './presence'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  $sessions,
  forgetSessionOwnerHintsForConnection,
  setSessionOwnerHint
} from './session'
import { $sessionStates, clearAllSessionStates, recordSessionEventScope } from './session-states'

interface Frame {
  id?: string
  method?: string
  params?: Record<string, unknown>
}

class Sock extends EventTarget {
  static OPEN = 1
  static CLOSED = 3
  static sockets: Sock[] = []
  static answers: Record<string, unknown>[] = []
  readyState = 0
  sent: Frame[] = []
  readonly answer: Record<string, unknown>

  constructor() {
    super()
    this.answer = Sock.answers[Sock.sockets.length] ?? {}
    Sock.sockets.push(this)
  }

  open() {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
    this.frame({ jsonrpc: '2.0', method: 'event', params: { payload: {}, type: 'gateway.ready' } })
  }

  close() {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }

  frame(frame: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }

  send(data: string) {
    const frame = JSON.parse(data) as Frame
    this.sent.push(frame)

    if (frame.id) {
      queueMicrotask(() => this.frame({ id: frame.id, jsonrpc: '2.0', result: this.answer }))
    }
  }

  methods() {
    return this.sent.map(frame => frame.method)
  }
}

const URL_ = 'ws://example.invalid/api/ws'

async function openGateway(answer: Record<string, unknown> = { client_activity: true }) {
  Sock.answers[Sock.sockets.length] = answer
  const gateway = new HermesGateway()
  const connected = gateway.connect(URL_)
  const socket = Sock.sockets.at(-1)!
  socket.open()
  await connected
  await settle()

  return { gateway, socket }
}

const settle = (ms = 0) => vi.advanceTimersByTimeAsync(ms)

/** The primary pane presents `stored` on its own route, bound to the runtime `runtime`. */
function showChat(stored = 'stored-1', runtime = 'rt-1') {
  window.location.hash = `#/${stored}`
  $selectedStoredSessionId.set(stored)
  $activeSessionId.set(runtime)
  $sessionStates.set({ [runtime]: createClientSessionState(stored) })
}

/** A host that can show notifications; resolves once main's answer is known. */
function notifyingHost() {
  const supported = Promise.resolve(true)
  window.hermesDesktop = { notify: vi.fn(async () => true), notifySupported: () => supported } as never

  return supported
}

const reports = (socket: Sock) => socket.sent.filter(frame => frame.method === 'client.activity').map(frame => frame.params!)

let stop: (() => void) | null = null

beforeEach(() => {
  vi.useFakeTimers()
  vi.stubGlobal('WebSocket', Sock)
  Sock.sockets = []
  Sock.answers = []
  setNativeNotifyEnabled(true)
  setNativeNotifyKind('turnDone', true)
  __resetNativeNotifyBaselineForTests()
})

afterEach(() => {
  stop?.()
  stop = null
  Sock.sockets.forEach(socket => socket.close())
  setPrimaryGateway(null)
  $gateway.set(null)
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  clearAllSessionStates()
  $sessions.set([])
  $layoutTree.set(null)
  forgetSessionOwnerHintsForConnection('conn-a')
  window.location.hash = ''
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  delete document.documentElement.dataset.hermesDesktopHost
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('the report goes where a submit for the session in view goes', () => {
  it('two backends: the owner gets client.activity and prompt.submit, the ambient one gets neither', async () => {
    const owner = await openGateway()
    const ambient = await openGateway()
    setPrimaryGateway(owner.gateway, 'default')
    setPrimaryGatewayConnectionId('conn-a', 'remote')
    $gateway.set(ambient.gateway)
    setSessionOwnerHint('stored-1', { connectionId: 'conn-a', profile: 'default' })
    showChat()

    stop = startPresenceReporter()
    document.dispatchEvent(new Event('keydown'))
    await settle()
    await captureSubmissionDestination('stored-1', (method, params) => ambient.gateway.request(method, params)).requestGateway(
      'prompt.submit',
      { session_id: 'rt-1', text: 'hi' }
    )
    await settle()

    expect(reports(owner.socket)).toEqual([expect.objectContaining({ sessions: ['rt-1'] })])
    expect(owner.socket.methods()).toContain('prompt.submit')
    expect(ambient.socket.methods()).not.toContain('client.activity')
    expect(ambient.socket.methods()).not.toContain('prompt.submit')
  })

  describe('a session whose row names only a profile', () => {
    /** The composer's submit for `stored-1` on the window's real session dispatcher. */
    async function submit(ambient: HermesGateway) {
      const dispatch = createSessionRpcDispatcher({
        ambientRequest: (method, params) => ambient.request(method, params),
        runtimeIdByStoredSessionIdRef: { current: new Map([['stored-1', 'rt-1']]) },
        selectedStoredSessionIdRef: { current: 'stored-1' },
        sessionStateByRuntimeIdRef: { current: new Map([['rt-1', createClientSessionState('stored-1')]]) }
      })

      await captureSubmissionDestination('stored-1', dispatch)
        .requestGateway('prompt.submit', { session_id: 'rt-1', text: 'hi' })
        .catch(() => undefined)
      await settle()
    }

    async function start() {
      const primary = await openGateway()
      setPrimaryGateway(primary.gateway, 'work')
      $gateway.set(primary.gateway)
      $sessions.set([{ id: 'stored-1', profile: 'work' } as never])
      showChat()

      return primary
    }

    it('reports on the socket the submit uses', async () => {
      const primary = await start()

      stop = startPresenceReporter()
      document.dispatchEvent(new Event('keydown'))
      await settle()
      await submit(primary.gateway)

      expect(primary.socket.methods()).toContain('prompt.submit')
      expect(reports(primary.socket)).toEqual([expect.objectContaining({ sessions: ['rt-1'] })])
    })

    it('reports nowhere the submit would not go when its events came from another connection', async () => {
      const primary = await start()
      // The runtime's own events proved its exact owner: another connection serving the same profile.
      recordSessionEventScope({ connectionId: 'conn-b', profile: 'work', session_id: 'rt-1' })

      stop = startPresenceReporter()
      document.dispatchEvent(new Event('keydown'))
      await settle()
      await submit(primary.gateway)

      expect(primary.socket.methods()).not.toContain('prompt.submit')
      expect(reports(primary.socket)).toEqual([])
    })
  })

  it('reports nothing for a draft, and never creates or resumes a session to report', async () => {
    const { gateway, socket } = await openGateway()
    $gateway.set(gateway)
    window.location.hash = '#/'
    $activeSessionId.set('rt-draft')

    stop = startPresenceReporter()
    document.dispatchEvent(new Event('keydown'))
    await settle(60_000)

    expect(socket.methods().filter(method => /^(client\.activity|session\.)/.test(method ?? ''))).toEqual([])
  })

  it('withdraws when the window moves from the chat to the Settings route, focus and all', async () => {
    const { gateway, socket } = await openGateway()
    $gateway.set(gateway)
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    setNativeNotifyEnabled(false)
    showChat()

    stop = startPresenceReporter()
    document.dispatchEvent(new Event('keydown'))
    await settle()
    window.location.hash = '#/settings/notifications'
    window.dispatchEvent(new HashChangeEvent('hashchange'))
    document.dispatchEvent(new Event('keydown'))
    await settle(60_000)

    expect(reports(socket)).toEqual([
      expect.objectContaining({ can_alert: true, sessions: ['rt-1'] }),
      expect.objectContaining({ sessions: [] })
    ])
  })
})

describe('a reply counts as on screen only while the main chat is the tab in front', () => {
  const zone = (active: string) => group(['workspace', 'session-tile:stored-2', 'files'], { active, id: 'main' })

  it('another tab in front of the main chat gives can_alert false at once, and true again when it is back', async () => {
    const { gateway, socket } = await openGateway()
    $gateway.set(gateway)
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    $layoutTree.set(zone('workspace'))
    showChat()

    stop = startPresenceReporter()
    document.dispatchEvent(new Event('keydown'))
    await settle()
    // A second conversation opened as a tab in the same zone: the route and the main session stay.
    $layoutTree.set(zone('session-tile:stored-2'))
    await settle()
    $layoutTree.set(zone('workspace'))
    await settle()
    // A tool tab covers it the same way.
    $layoutTree.set(zone('files'))
    await settle()

    expect(reports(socket).map(report => [report.sessions, report.can_alert])).toEqual([
      [['rt-1'], true],
      [['rt-1'], false],
      [['rt-1'], true],
      [['rt-1'], false]
    ])
  })

  it('a covered main chat still counts when this app would notify for it', async () => {
    const { gateway, socket } = await openGateway()
    $gateway.set(gateway)
    // In the background with "Response ready" on: the notification reaches the user.
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    $layoutTree.set(zone('session-tile:stored-2'))
    showChat()
    const known = notifyingHost()

    stop = startPresenceReporter()
    await known
    document.dispatchEvent(new Event('keydown'))
    await settle()

    expect(reports(socket)).toEqual([expect.objectContaining({ can_alert: true, sessions: ['rt-1'] })])
  })

  it('a minimized zone hides the main chat too', async () => {
    const { gateway, socket } = await openGateway()
    $gateway.set(gateway)
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    $layoutTree.set(zone('workspace'))
    showChat()

    stop = startPresenceReporter()
    document.dispatchEvent(new Event('keydown'))
    await settle()
    $layoutTree.set({ ...zone('workspace'), minimized: true })
    await settle()

    expect(reports(socket).map(report => [report.sessions, report.can_alert])).toEqual([
      [['rt-1'], true],
      [['rt-1'], false]
    ])
  })
})

describe('client.activity only on a connection that advertised it, now', () => {
  it('a redial to a backend that never advertised client_activity gets no report', async () => {
    const first = await openGateway()
    const client = first.gateway
    expect(client.clientActivity).toBe(true)
    client.close()

    Sock.answers[Sock.sockets.length] = {}
    const reconnected = client.connect(URL_)
    const second = Sock.sockets.at(-1)!
    const target = new EventTarget()

    const reporter = createPresenceReporter({
      canAlert: () => true,
      focused: () => true,
      inputTarget: target,
      view: () => ({ channel: client, kind: 'session', sessionId: 's' }),
      visible: () => true
    })

    target.dispatchEvent(new Event('keydown'))
    await settle(5)
    expect(client.clientActivity).toBe(false)
    second.open()
    await reconnected
    await settle(20_000)
    reporter.stop()

    expect(second.methods()).not.toContain('client.activity')
  })
})

describe('can_alert by notification uses the check that sends the notification', () => {
  it('a browser permission revoked after start gives can_alert false', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.stubGlobal('isSecureContext', true)
    const permission = { value: 'granted' }
    vi.stubGlobal('Notification', { get permission() { return permission.value } })
    $browserNotificationPermission.set('granted')
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    const { gateway, socket } = await openGateway()
    $gateway.set(gateway)
    showChat()
    notifyingHost()

    stop = startPresenceReporter()
    document.dispatchEvent(new Event('keydown'))
    await settle()
    permission.value = 'denied'
    // Nothing announces a revoked permission; the next evaluation reads it live.
    window.dispatchEvent(new Event('blur'))
    await settle()

    expect(reports(socket).map(report => report.can_alert)).toEqual([true, false])
  })

  it('inside the post-connect quiet interval a background session gives can_alert false', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    const { gateway, socket } = await openGateway()
    $gateway.set(gateway)
    showChat()
    markNativeNotifyBaseline()

    stop = startPresenceReporter()
    document.dispatchEvent(new Event('keydown'))
    await settle()

    expect(reports(socket)).toEqual([expect.objectContaining({ can_alert: false, sessions: ['rt-1'] })])
  })
})
