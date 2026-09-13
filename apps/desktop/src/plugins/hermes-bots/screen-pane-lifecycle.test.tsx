import { act, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type * as ScreenConnection from './screen-connection'
import type { RosterRow } from './types'

const sockets = vi.hoisted(() => [] as Array<{ closeCodes: number[]; closed: boolean; close: (code?: number) => void }>)
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
vi.mock('./routing', () => ({
  botConnectionRoute: () => ({ connectionId: 'host-a', profile: 'default', targetProfile: 'default' })
}))
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      title: 'Screen',
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
    constructor(
      _target: HTMLElement,
      private socket: { close: () => void }
    ) {}
    addEventListener(type: string, callback: () => void) {
      if (type === 'connect') {
        queueMicrotask(callback)
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
  lease: { holder: 'human', viewer_id: 'this-viewer', pending_handoff: null, since: 1, reason: '' }
}

beforeEach(() => {
  $screenState.set({})
  sockets.length = 0
  retention.held = 0
  vi.mocked(displayRequest)
    .mockReset()
    .mockResolvedValue({ ...status, ticket: 'test-ticket', viewer_id: 'this-viewer' })
  vi.stubGlobal(
    'WebSocket',
    class {
      closeCodes: number[] = []
      closed = false
      constructor() {
        sockets.push(this)
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
  fireEvent.click(view.getByTitle('Reconnect'))
  await waitFor(() => expect(sockets).toHaveLength(2))
  expect(retention.held).toBe(1)
  view.unmount()
  expect(retention.held).toBe(0)
})

it('does not hand back while replacing a stream to reconnect the same viewer', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(sockets).toHaveLength(1))
  await act(async () => {})
  fireEvent.click(view.getByTitle('Reconnect'))
  await waitFor(() => expect(sockets).toHaveLength(2))
  expect(sockets[0].closeCodes).toEqual([1005])
  expect(sockets[1].closed).toBe(false)
  view.unmount()
})
