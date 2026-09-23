import { act, fireEvent, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type * as ScreenConnection from './screen-connection'
import type { RosterRow } from './types'

const sockets = vi.hoisted(
  () => [] as Array<{ closeCodes: number[]; closed: boolean; close: (code?: number) => void; serverClose: (code: number) => void }>
)

const rfbs = vi.hoisted(() => [] as Array<{ emit: (type: string, detail?: unknown) => void; viewOnly?: boolean }>)
const retention = vi.hoisted(() => ({ held: 0 }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button {...props}>{children}</button>
    ),
    Codicon: () => null,
    GlyphSpinner: () => null,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>,
    EmptyState: () => null,
    useValue: useStore,
    host: {
      onEvent: onGatewayEvent,
      retainProfile: async () => {
        retention.held += 1

        return () => {
          retention.held -= 1
        }
      }
    }
  }
})
vi.mock('./routing', () => {
  const route = { connectionId: 'host-a', mode: 'remote', profile: 'default', targetProfile: 'default' }

  return { botConnectionRoute: () => route, resolveBotConnectionRoute: () => ({ status: 'resolved', route }) }
})
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      title: 'Screen',
      controlTaken: 'Another viewer took control',
      youControl: 'You control',
      handBack: 'Hand back',
      takeOver: 'Take over',
      reconnect: 'Reconnect',
      streamLost: 'Stream lost'
    }
  })
}))
vi.mock('./screen-connection', async importActual => ({
  // Real pure helpers (viewerHash / leaseHeldBy); only the gateway legs are faked.
  ...(await importActual<typeof ScreenConnection>()),
  displayRequest: vi.fn(),
  resolveScreenWsUrl: vi.fn(async () => 'ws://localhost/api/display/ws'),
  isEventForBotScreen: () => false
}))
vi.mock('@novnc/novnc', () => ({
  default: class {
    private listeners = new Map<string, Array<(event: { detail?: unknown }) => void>>()
    viewOnly = true
    constructor(
      _target: HTMLElement,
      private socket: { close: () => void }
    ) {
      rfbs.push(this)
    }
    emit(type: string, detail?: unknown) {
      for (const listener of this.listeners.get(type) ?? []) {
        listener({ detail })
      }
    }
    addEventListener(type: string, callback: (event: { detail?: unknown }) => void) {
      this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback])

      if (type === 'connect') {
        queueMicrotask(() => callback({}))
      }
    }
    // noVNC 1.7 disconnects its WebSocket without a close code.
    disconnect() {
      this.socket.close()
    }
    focus() {}
  }
}))

import { displayRequest } from './screen-connection'
import { BotScreenPane } from './screen-pane'
import { $screenState } from './screen-state'

const bot: RosterRow = { name: 'default' }

const status: DisplayStatus = {
  profile: 'default',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: true,
  missing: [],
  running: true,
  pid: 42,
  display: ':20',
  socket: '/tmp/rfb.sock',
  geometry: '1440x900',
  install_command: null,
  lease: { holder: 'human', viewer_id: null, viewer_hash: 'e0f9a555d558', since: 1, reason: '', epoch: 1 }
}

beforeEach(() => {
  $screenState.set({})
  sockets.length = 0
  rfbs.length = 0
  retention.held = 0
  vi.mocked(displayRequest)
    .mockReset()
    .mockResolvedValue({ ...status, ticket: 'test-ticket', viewer_id: 'this-viewer' })
  vi.stubGlobal(
    'WebSocket',
    class {
      closeCodes: number[] = []
      closed = false
      private onClose: Array<(event: { code: number }) => void> = []
      constructor() {
        sockets.push(this)
      }
      addEventListener(type: string, listener: (event: { code: number }) => void) {
        if (type === 'close') {
          this.onClose.push(listener)
        }
      }
      // The bridge closing us: the raw close frame reaches our listener, then noVNC
      // reports a statusless `disconnect` — the code is only on the socket event.
      serverClose(code: number) {
        this.closed = true

        for (const listener of this.onClose) {
          listener({ code })
        }
      }
      close(code?: number) {
        // Subsequent close calls cannot replace the frame already sent to the server.
        if (this.closed) {
          return
        }

        this.closed = true
        this.closeCodes.push(code ?? 1005)
      }
    }
  )
})

afterEach(() => vi.unstubAllGlobals())

it('sends an intentional close before noVNC can send its statusless close on pane unmount', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(sockets).toHaveLength(1))
  await act(async () => {})
  view.unmount()
  expect(sockets[0].closeCodes).toEqual([1000])
})

it('pins the bot socket for the attach lifetime and lets go on unmount', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(sockets).toHaveLength(1))
  await act(async () => {})
  expect(retention.held).toBe(1)
  fireEvent.click(view.getByLabelText('Reconnect'))
  await waitFor(() => expect(sockets).toHaveLength(2))
  expect(retention.held).toBe(1)
  view.unmount()
  expect(retention.held).toBe(0)
})

it('does not hand back while replacing a stream to reconnect the same viewer', async () => {
  // The server mints a fresh id for a bare observe; only an observe that re-presents the minted id
  // keeps it. Model that, so a pane that forgets its id visibly loses the lease it holds.
  let minted = 0
  vi.mocked(displayRequest).mockImplementation(async (_bot, method, params) => {
    if (method !== 'display.observe') {
      return { ...status, ticket: 'test-ticket', viewer_id: 'this-viewer' }
    }

    const presented = (params as { viewer_id?: string } | undefined)?.viewer_id
    const viewer_id = presented === 'this-viewer' ? 'this-viewer' : minted++ === 0 ? 'this-viewer' : 'replacement-viewer'

    return { ...status, ticket: 'test-ticket', viewer_id }
  })
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(sockets).toHaveLength(1))
  await act(async () => {})
  fireEvent.click(view.getByLabelText('Reconnect'))
  await waitFor(() => expect(sockets).toHaveLength(2))
  expect(sockets[0].closeCodes).toEqual([1005])
  expect(sockets[1].closed).toBe(false)
  // Same viewer after the reconnect: the human's lease still belongs to this pane and it can hand back.
  expect(rfbs[1].viewOnly).toBe(false)
  view.unmount()
})

it('re-attaches in watch mode after the bridge evicts us with 4000, with a bounded retry budget', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(sockets).toHaveLength(1))
  await act(async () => {})
  const observes = () => vi.mocked(displayRequest).mock.calls.filter(([, method]) => method === 'display.observe').length
  expect(observes()).toBe(1)

  act(() => {
    sockets[0].serverClose(4000)
    rfbs[0].emit('disconnect', { clean: true })
  })
  // The caption is informational; the stream itself must come back on a fresh ticket,
  // not sit frozen on a dead socket until the user finds Reconnect.
  expect(view.getByText('Another viewer took control')).toBeTruthy()
  await waitFor(() => expect(sockets).toHaveLength(2))
  expect(observes()).toBe(2)
  await act(async () => {})
  expect(view.queryByText('Another viewer took control')).toBeNull()

  // A bridge that evicts every fresh attach must not become a tight observe loop: after a
  // bounded run of rapid evictions the pane lands in the error state with Reconnect.
  const evictLatest = async () => {
    await waitFor(() => expect(sockets.at(-1)?.closed).toBe(false))
    const index = sockets.length - 1
    act(() => {
      sockets[index].serverClose(4000)
      rfbs[index].emit('disconnect', { clean: true })
    })
    await act(async () => {})
  }

  for (let round = 0; round < 3; round += 1) {
    await evictLatest()
  }

  await act(async () => new Promise(resolve => setTimeout(resolve, 20)))
  expect(observes()).toBe(4)
  expect(sockets.at(-1)?.closed).toBe(true)
  expect(view.getByLabelText('Reconnect').closest('button')?.disabled).toBe(false)
  view.unmount()
})
