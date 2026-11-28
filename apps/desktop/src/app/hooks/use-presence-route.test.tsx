// A move made by the real router reaches the window's presence reporter at once. HashRouter pushes
// history without a `hashchange` event, so nothing else would tell the reporter before its next tick.
import { act, cleanup, render } from '@testing-library/react'
import { HashRouter, type NavigateFunction, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/api/client'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $gateway } from '@/store/gateway'
import { setNativeNotifyEnabled, setNativeNotifyKind } from '@/store/native-notifications'
import { __resetNativeNotifyBaselineForTests } from '@/store/notify-baseline'
import { startPresenceReporter } from '@/store/presence'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { usePresenceRoute } from './use-presence-route'

interface Frame {
  id?: string
  method?: string
  params?: Record<string, unknown>
}

class Sock extends EventTarget {
  static OPEN = 1
  static CLOSED = 3
  static sockets: Sock[] = []
  readyState = 0
  sent: Frame[] = []

  constructor() {
    super()
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
      queueMicrotask(() => this.frame({ id: frame.id, jsonrpc: '2.0', result: { client_activity: true } }))
    }
  }
}

const settle = (ms = 0) => vi.advanceTimersByTimeAsync(ms)

const reports = (socket: Sock) =>
  socket.sent.filter(frame => frame.method === 'client.activity').map(frame => frame.params!)

let stop: (() => void) | null = null
let go: NavigateFunction = () => undefined

function Routed({ told }: { told: boolean }) {
  go = useNavigate()

  return told ? <Told /> : null
}

function Told() {
  usePresenceRoute()

  return null
}

async function chatOpenWithInput(told: boolean) {
  const gateway = new HermesGateway()
  const connected = gateway.connect('ws://example.invalid/api/ws')
  const socket = Sock.sockets.at(-1)!
  socket.open()
  await connected
  await settle()
  $gateway.set(gateway)

  window.location.hash = '#/stored-1'
  $selectedStoredSessionId.set('stored-1')
  $activeSessionId.set('rt-1')
  $sessionStates.set({ 'rt-1': createClientSessionState('stored-1') })

  render(
    <HashRouter useTransitions={false}>
      <Routed told={told} />
    </HashRouter>
  )
  stop = startPresenceReporter()
  window.document.dispatchEvent(new Event('keydown'))
  await settle()

  return socket
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.stubGlobal('WebSocket', Sock)
  Sock.sockets = []
  // Notifications off: only the conversation on screen in this focused window can alert.
  setNativeNotifyEnabled(false)
  setNativeNotifyKind('turnDone', true)
  __resetNativeNotifyBaselineForTests()
  vi.spyOn(window.document, 'hasFocus').mockReturnValue(true)
})

afterEach(() => {
  stop?.()
  stop = null
  cleanup()
  Sock.sockets.forEach(socket => socket.close())
  $gateway.set(null)
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
  window.location.hash = ''
  setNativeNotifyEnabled(true)
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('a router move reaches the presence reporter', () => {
  it('withdraws the moment the router leaves the chat, and claims again the moment it is back', async () => {
    const socket = await chatOpenWithInput(true)
    const hashchanges: Event[] = []
    const heard = (event: Event) => hashchanges.push(event)
    window.addEventListener('hashchange', heard)

    expect(reports(socket)).toEqual([expect.objectContaining({ can_alert: true, sessions: ['rt-1'] })])

    await act(async () => {
      await go('/settings?page=main')
    })
    await settle()

    expect(window.location.hash).toBe('#/settings?page=main')
    expect(reports(socket)).toEqual([
      expect.objectContaining({ can_alert: true, sessions: ['rt-1'] }),
      expect.objectContaining({ sessions: [] })
    ])

    await act(async () => {
      await go('/stored-1')
    })
    await settle()

    expect(reports(socket)).toHaveLength(3)
    expect(reports(socket)[2]).toEqual(expect.objectContaining({ can_alert: true, sessions: ['rt-1'] }))
    // The router said nothing a hashchange listener could hear, and no reporter tick has run.
    expect(hashchanges).toEqual([])
    window.removeEventListener('hashchange', heard)
  })

  it('control: the reporter alone does not hear a router move before its next tick', async () => {
    const socket = await chatOpenWithInput(false)

    await act(async () => {
      await go('/settings?page=main')
    })
    await settle()

    expect(window.location.hash).toBe('#/settings?page=main')
    expect(reports(socket)).toEqual([expect.objectContaining({ can_alert: true, sessions: ['rt-1'] })])
  })
})
