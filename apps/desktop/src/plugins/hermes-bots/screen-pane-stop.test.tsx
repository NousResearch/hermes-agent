/**
 * RED: a running Bot Screen pane must offer an end-session control. The
 * backend owns `display.stop` (plus `force` while a human holds the lease)
 * but the pane only ever calls `display.start` — once running there is no
 * way from Desktop to end the session, so the Bots view sticks on
 * "Screen · bot in control" and reopening the tab re-attaches to the same
 * live runtime.
 */

import { fireEvent, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type * as ScreenConnection from './screen-connection'
import type { RosterRow } from './types'

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
    host: { onEvent: onGatewayEvent }
  }
})
vi.mock('./routing', () => {
  const route = { connectionId: 'host-a', mode: 'remote', profile: 'ops', targetProfile: 'ops' }

  return { botConnectionRoute: () => route, resolveBotConnectionRoute: () => ({ status: 'resolved', route }) }
})
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      title: 'Screen',
      stoppedTitle: 'Stopped',
      stoppedBody: 'Body',
      start: 'Start',
      stop: 'Stop screen',
      stopForce: 'Stop screen (force)',
      stopForceHint: 'Stop even though another viewer is in control',
      agentControls: 'Bot controls',
      takeOver: 'Take over',
      reconnect: 'Reconnect',
      streamLost: 'Stream lost'
    }
  })
}))
vi.mock('./screen-connection', async importActual => ({
  ...(await importActual<typeof ScreenConnection>()),
  displayRequest: vi.fn(),
  resolveScreenWsUrl: vi.fn(async () => 'ws://localhost/api/display/ws')
}))
vi.mock('@novnc/novnc', () => ({
  default: class {
    viewOnly = true
    constructor(_target: HTMLElement, _socket: unknown) {}
    addEventListener() {}
    disconnect() {}
    focus() {}
  }
}))

import { displayRequest } from './screen-connection'
import { BotScreenPane } from './screen-pane'
import { $screenState } from './screen-state'

const bot: RosterRow = { name: 'ops', sourceScoped: true, connectionId: 'host-a', connectionKind: 'remote' }

const stopped: DisplayStatus = {
  profile: 'ops',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: true,
  missing: [],
  running: false,
  pid: null,
  display: null,
  socket: null,
  geometry: '1440x900',
  install_command: null,
  lease: { holder: 'agent', viewer_id: null, viewer_hash: null, since: 1, reason: '', epoch: 0 }
}

const agentLease = {
  holder: 'agent',
  viewer_id: null,
  viewer_hash: null,
  since: 1,
  reason: '',
  epoch: 1
} as const

const running: DisplayStatus = { ...stopped, running: true, pid: 42, display: ':20', lease: { ...agentLease } }

beforeEach(() => {
  $screenState.set({})
  vi.mocked(displayRequest)
    .mockReset()
    .mockImplementation(async (_bot, method) => {
      if (method === 'display.observe') {
        return { ...running, ticket: 't', viewer_id: 'v1' }
      }

      if (method === 'display.stop') {
        return { ...stopped, lease: { holder: 'agent', viewer_id: null, viewer_hash: null, since: 2, reason: '', epoch: 2 } }
      }

      return { ...running }
    })
  vi.stubGlobal(
    'WebSocket',
    class {
      binaryType = ''
      closed = false
      addEventListener() {}
      close() {
        this.closed = true
      }
    }
  )
})

afterEach(() => vi.unstubAllGlobals())

it('a running pane offers Stop screen, which ends the session and paints the stopped state', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  const stop = await view.findByText('Stop screen')
  fireEvent.click(stop)
  await waitFor(() => expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.stop', {}))
  await waitFor(() => expect(view.getByText('Stopped')).toBeTruthy())
  view.unmount()
})

it('stopping while another viewer holds the lease forces the stop so the session can still end', async () => {
  const humanLease = {
    holder: 'human',
    viewer_id: null,
    viewer_hash: 'someone-else',
    since: 1,
    reason: '',
    epoch: 1
  } as const
  vi.mocked(displayRequest).mockImplementation(async (_bot, method) => {
    if (method === 'display.observe') {
      // The backend snapshot is internally consistent: observe carries the same lease as status.
      return { ...running, lease: { ...humanLease }, ticket: 't', viewer_id: 'v1' }
    }

    if (method === 'display.stop') {
      return { ...stopped }
    }

    // A lease held by a different viewer: this pane's minted id never matches it.
    return { ...running, lease: { ...humanLease } }
  })
  const view = render(<BotScreenPane bot={bot} />)
  const stopForce = await view.findByText('Stop screen (force)')
  fireEvent.click(stopForce)
  await waitFor(() => expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.stop', { force: true }))
  await waitFor(() => expect(view.getByText('Stopped')).toBeTruthy())
  view.unmount()
})
