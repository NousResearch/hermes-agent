/**
 * Explicit-paste clipboard bridge (#123089): a native `paste` gesture over the
 * live screen must forward the client's text clipboard to noVNC's
 * `clipboardPasteFrom`, but only while this viewer holds the lease — never on
 * a watch-only stream, and never via polling.
 */

import { fireEvent, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type * as ScreenConnection from './screen-connection'
import type { RosterRow } from './types'

const rfbs = vi.hoisted(
  () => [] as Array<{ target: HTMLElement; viewOnly: boolean; clipboardPasteFrom: (text: string) => void }>
)

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
    host: { onEvent: onGatewayEvent, retainProfile: async () => () => {} }
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
      youControl: 'You control',
      handBack: 'Hand back',
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
  isEventForBotScreen: () => false
}))
vi.mock('@novnc/novnc', () => ({
  default: class {
    viewOnly = true
    clipboardPasteFrom = vi.fn()
    constructor(target: HTMLElement) {
      rfbs.push(this as never)
      ;(this as unknown as { target: HTMLElement }).target = target
    }
    addEventListener(type: string, callback: (event: { detail?: unknown }) => void) {
      if (type === 'connect') {
        queueMicrotask(() => callback({}))
      }
    }
    disconnect() {}
    focus() {}
  }
}))

import { displayRequest } from './screen-connection'
import { BotScreenPane } from './screen-pane'
import { $screenState } from './screen-state'

const bot: RosterRow = { name: 'default' }

// `viewer_hash` is the first 12 hex of sha256("this-viewer"); leaseHeldBy() matches on it,
// which is what makes this viewer the lease holder (`iHold`) and puts `client.viewOnly = false`.
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
  rfbs.length = 0
  vi.mocked(displayRequest)
    .mockReset()
    .mockResolvedValue({ ...status, ticket: 'test-ticket', viewer_id: 'this-viewer' })
  vi.stubGlobal(
    'WebSocket',
    class {
      binaryType = ''
      addEventListener() {}
      close() {}
    }
  )
})

afterEach(() => vi.unstubAllGlobals())

it('forwards an explicit paste over the canvas to noVNC while holding the lease', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(false))

  fireEvent.paste(rfbs[0].target, { clipboardData: { getData: () => 'clipboard-test-123' } })

  expect(rfbs[0].clipboardPasteFrom).toHaveBeenCalledWith('clipboard-test-123')
  view.unmount()
})

it('drops a paste while this viewer only watches (no lease)', async () => {
  vi.mocked(displayRequest).mockResolvedValue({
    ...status,
    lease: { ...status.lease, viewer_hash: 'someone-else' },
    ticket: 'test-ticket',
    viewer_id: 'this-viewer'
  })

  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(true))

  fireEvent.paste(rfbs[0].target, { clipboardData: { getData: () => 'clipboard-test-123' } })

  expect(rfbs[0].clipboardPasteFrom).not.toHaveBeenCalled()
  view.unmount()
})

it('drops a paste over the bridge\'s 256 KiB cut-text cap instead of forwarding it', async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(false))

  const oversized = 'a'.repeat(256 * 1024 + 1)
  fireEvent.paste(rfbs[0].target, { clipboardData: { getData: () => oversized } })

  expect(rfbs[0].clipboardPasteFrom).not.toHaveBeenCalled()
  view.unmount()
})
