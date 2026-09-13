/**
 * Viewer identity is SERVER-MINTED: `display.observe` returns the id this attach
 * is known by, and lease payloads name the holder by `viewer_hash`
 * (sha256(viewer_id)[:12]) rather than the raw id. "I hold" must be derived
 * from the minted id, never from a client-generated constant — otherwise a
 * Desktop reload could claim (or lose) control it does not have.
 */

import { act, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type * as ScreenConnection from './screen-connection'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
    Codicon: () => null,
    GlyphSpinner: () => null,
    EmptyState: () => null,
    useValue: useStore,
    host: { onEvent: onGatewayEvent }
  }
})
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      title: 'Screen',
      youControl: 'You control',
      otherControls: 'Other controls',
      agentControls: 'Bot controls',
      handBack: 'Hand back',
      handBackForce: 'Hand back (force)',
      handBackForceHint: 'Force',
      takeOver: 'Take over',
      reconnect: 'Reconnect',
      streamLost: 'Stream lost'
    }
  })
}))
vi.mock('./screen-connection', async importActual => ({
  ...(await importActual<typeof ScreenConnection>()),
  displayRequest: vi.fn(),
  resolveScreenWsUrl: vi.fn(async () => 'ws://localhost/api/display/ws'),
  isEventForBotScreen: () => true
}))
vi.mock('@novnc/novnc', () => ({
  default: class {
    addEventListener() {}
    disconnect() {}
    focus() {}
  }
}))

// Real event bus, so the pane's listener path is the one under test.
// eslint-disable-next-line no-restricted-imports
import { emitGatewayEvent } from '../../contrib/events'

import { displayRequest, viewerHash } from './screen-connection'
import { BotScreenPane } from './screen-pane'
import { $screenState } from './screen-state'

const bot: RosterRow = { name: 'default' }
const MINTED = 'srv-viewer-0001'

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
  lease: { holder: 'agent', viewer_id: null, viewer_hash: null, pending_handoff: null, since: 1, reason: '' }
}

beforeEach(() => {
  $screenState.set({})
  vi.mocked(displayRequest)
    .mockReset()
    .mockImplementation(async (_bot, method) =>
      method === 'display.observe' ? { ...status, ticket: 'test-ticket', viewer_id: MINTED } : status
    )
  vi.stubGlobal(
    'WebSocket',
    class {
      binaryType = ''
      close() {}
    }
  )
})

afterEach(() => vi.unstubAllGlobals())

const emitLease = (viewer_hash: string) =>
  act(() =>
    emitGatewayEvent({
      type: 'display.lease',
      payload: { profile_key: status.profile_key, lease: { ...status.lease, holder: 'human', viewer_hash } }
    })
  )

it('holds control when the lease names the hash of the server-minted viewer id, not when it names another', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.observe'))
  await act(async () => {})

  emitLease(await viewerHash('someone-else'))
  expect(view.queryByText('You control')).toBeNull()
  expect(view.getByText('Other controls')).toBeTruthy()

  emitLease(await viewerHash(MINTED))
  expect(view.getByText('You control')).toBeTruthy()
  view.unmount()
})

it('hands back with the minted id, never a client-generated one', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.observe'))
  await act(async () => {})
  emitLease(await viewerHash(MINTED))

  await act(async () => {
    view.getByText('Hand back').click()
  })
  expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.lease.release', { viewer_id: MINTED })
  view.unmount()
})

it('offers a forced hand-back for a human lease this window does not hold, sending {force: true} and no viewer id', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.observe'))
  await act(async () => {})

  emitLease(await viewerHash(MINTED))
  expect(view.queryByText('Hand back (force)')).toBeNull()

  emitLease(await viewerHash('viewer-from-before-the-reload'))
  await act(async () => {
    view.getByText('Hand back (force)').click()
  })
  expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.lease.release', { force: true })
  view.unmount()
})
