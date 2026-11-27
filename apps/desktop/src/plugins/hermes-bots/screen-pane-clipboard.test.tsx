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

const writes = vi.hoisted(() => vi.fn(async (_text: string) => {}))

const rfbs = vi.hoisted(
  () =>
    [] as Array<{
      target: HTMLElement
      canvas: HTMLCanvasElement
      viewOnly: boolean
      clipboardPasteFrom: (text: string) => void
      sendKey: (keysym: number, code: string, down?: boolean) => void
      noVncKeys: string[]
    }>
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
    writeClipboardText: writes,
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
      streamLost: 'Stream lost',
      copyFromScreen: 'Copy',
      copyFromScreenHint: 'Copy hint',
      copied: (chars: number) => `Copied ${chars}`,
      copyEmpty: 'Empty',
      copyTooLarge: 'Too large',
      copyUnavailable: 'Unavailable',
      pasteSent: 'Sent - press Ctrl+V',
      pasteTooLarge: 'Paste too large'
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
    sendKey = vi.fn()
    noVncKeys: string[] = []
    canvas: HTMLCanvasElement
    constructor(target: HTMLElement) {
      rfbs.push(this as never)
      ;(this as unknown as { target: HTMLElement }).target = target
      // Stands in for noVNC's Keyboard: it owns a focusable canvas and cancels every keydown it sees.
      this.canvas = document.createElement('canvas')
      this.canvas.tabIndex = -1
      this.canvas.addEventListener('keydown', event => {
        this.noVncKeys.push(event.code)
        event.stopPropagation()
        event.preventDefault()
      })
      target.appendChild(this.canvas)
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
  writes.mockClear()
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

let platformSpy: { mockRestore: () => void } | null = null

afterEach(() => {
  vi.unstubAllGlobals()
  platformSpy?.mockRestore()
  platformSpy = null
})

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

it("drops a paste over the bridge's 256 KiB cut-text cap instead of forwarding it", async () => {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(false))

  const oversized = 'a'.repeat(256 * 1024 + 1)
  fireEvent.paste(rfbs[0].target, { clipboardData: { getData: () => oversized } })

  expect(rfbs[0].clipboardPasteFrom).not.toHaveBeenCalled()
  view.unmount()
})

const withPlatform = (platform: string) => {
  platformSpy = vi.spyOn(navigator, 'platform', 'get').mockReturnValue(platform)
}

it("keeps Cmd+V away from noVNC on macOS so the browser's native paste can fire", async () => {
  withPlatform('MacIntel')
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(false))

  const notCancelled = fireEvent.keyDown(rfbs[0].canvas, { code: 'KeyV', key: 'v', metaKey: true })
  expect(notCancelled).toBe(true)
  expect(rfbs[0].noVncKeys).toEqual([])

  fireEvent.paste(rfbs[0].canvas, { clipboardData: { getData: () => 'mac-paste' } })
  expect(rfbs[0].clipboardPasteFrom).toHaveBeenCalledWith('mac-paste')
  expect(rfbs[0].sendKey).not.toHaveBeenCalled()
  view.unmount()
})

it('loads the clipboard, then replays the V so Ctrl+V pastes in the remote app', async () => {
  withPlatform('Linux x86_64')
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(false))

  expect(fireEvent.keyDown(rfbs[0].canvas, { code: 'KeyV', key: 'v', ctrlKey: true })).toBe(true)
  fireEvent.paste(rfbs[0].canvas, { clipboardData: { getData: () => 'linux-paste' } })

  expect(rfbs[0].clipboardPasteFrom).toHaveBeenCalledWith('linux-paste')
  expect(rfbs[0].sendKey).toHaveBeenCalledWith(0x0076, 'KeyV')
  expect(vi.mocked(rfbs[0].clipboardPasteFrom).mock.invocationCallOrder[0]).toBeLessThan(
    vi.mocked(rfbs[0].sendKey).mock.invocationCallOrder[0]
  )
  view.unmount()
})

it('leaves other keys, and every key while watching, to noVNC', async () => {
  withPlatform('MacIntel')
  vi.mocked(displayRequest).mockResolvedValue({
    ...status,
    lease: { ...status.lease, viewer_hash: 'someone-else' },
    ticket: 'test-ticket',
    viewer_id: 'this-viewer'
  })
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(true))

  fireEvent.keyDown(rfbs[0].canvas, { code: 'KeyV', key: 'v', metaKey: true })
  fireEvent.keyDown(rfbs[0].canvas, { code: 'KeyC', key: 'c', metaKey: true })
  expect(rfbs[0].noVncKeys).toEqual(['KeyV', 'KeyC'])
  view.unmount()
})

/** display.clipboard.read answers `clip`; every other display call answers the holding status. */
function serveClipboard(clip: { text?: null | string; reason?: null | string }) {
  vi.mocked(displayRequest).mockImplementation(async (_bot, method) =>
    method === 'display.clipboard.read' ? clip : { ...status, ticket: 'test-ticket', viewer_id: 'this-viewer' }
  )
}

it("copies the screen's clipboard to this computer from the toolbar, for this viewer only", async () => {
  serveClipboard({ text: 'from-the-bot' })
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs[0]?.viewOnly).toBe(false))

  fireEvent.click(view.getByText('Copy'))

  await waitFor(() => expect(writes).toHaveBeenCalledWith('from-the-bot'))
  expect(vi.mocked(displayRequest)).toHaveBeenCalledWith(bot, 'display.clipboard.read', { viewer_id: 'this-viewer' })
  await view.findByText('Copied 12')
  view.unmount()
})

it('copies with Cmd+C over the screen on macOS, without sending it to the remote app', async () => {
  withPlatform('MacIntel')
  serveClipboard({ text: 'cmd-c' })
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs[0]?.viewOnly).toBe(false))

  fireEvent.keyDown(rfbs[0].canvas, { code: 'KeyC', key: 'c', metaKey: true })

  await waitFor(() => expect(writes).toHaveBeenCalledWith('cmd-c'))
  expect(rfbs[0].noVncKeys).toEqual([])
  view.unmount()
})

it('says why nothing was copied and writes nothing', async () => {
  serveClipboard({ text: null, reason: 'empty' })
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs[0]?.viewOnly).toBe(false))

  fireEvent.click(view.getByText('Copy'))

  await view.findByText('Empty')
  expect(writes).not.toHaveBeenCalled()
  view.unmount()
})

it('offers no copy to a viewer that only watches', async () => {
  vi.mocked(displayRequest).mockResolvedValue({
    ...status,
    lease: { ...status.lease, viewer_hash: 'someone-else' },
    ticket: 'test-ticket',
    viewer_id: 'this-viewer'
  })
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs[0]?.viewOnly).toBe(true))

  expect(view.queryByText('Copy')).toBeNull()
  view.unmount()
})

it('tells a Mac user the paste is waiting for Ctrl+V in the app', async () => {
  withPlatform('MacIntel')
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs[0]?.viewOnly).toBe(false))

  fireEvent.keyDown(rfbs[0].canvas, { code: 'KeyV', key: 'v', metaKey: true })
  fireEvent.paste(rfbs[0].canvas, { clipboardData: { getData: () => 'mac-paste' } })

  await view.findByText('Sent - press Ctrl+V')
  view.unmount()
})
