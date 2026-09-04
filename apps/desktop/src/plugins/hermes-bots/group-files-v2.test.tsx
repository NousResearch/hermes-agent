import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GroupAttachmentDownload } from './group-attachment-download'
import { $groupChats } from './group-chat'
import { parseGroupFilesPage } from './group-files-client'
import { deferred, FILE_ROOM, fileItem, filePage, parsedFilePage } from './group-files-test-fixtures'
import { SharedFilesControl, SharedFilesDialog } from './group-files-view'
import { $hostedRoomCapabilities } from './hosted-room-capability-state'
import { classifyHostedRoomCapability } from './hosted-room-client'
import { translateBots } from './i18n-test-helper'
import type { Attachment, GroupMessage } from './types'

const mocks = vi.hoisted(() => ({ read: vi.fn(), request: vi.fn(), route: vi.fn(), notify: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, notify: mocks.notify }, usePluginI18n: () => translateBots }
})
vi.mock('./hosted-room-runtime', () => ({
  hostedRouteForRoom: mocks.route,
  readHostedGroupChatAttachment: mocks.read,
  requestHostedConnection: mocks.request
}))

const props = {
  availability: 'available' as const,
  group: 'Core',
  latestSeq: 20,
  onClose: vi.fn(),
  open: true,
  roomId: 'room-1'
}

const capable = () =>
  classifyHostedRoomCapability(
    {
      driver: true,
      persistent_process: true,
      authority_gateway_id: 'install:home',
      features: ['attachment_metadata_catalog']
    },
    { connectionId: 'gateway-a' }
  )

beforeEach(() => {
  vi.clearAllMocks()
  $groupChats.set({ Core: FILE_ROOM })
  $hostedRoomCapabilities.set({ 'gateway-a': capable() })
  mocks.route.mockResolvedValue({
    connectionId: 'gateway-a',
    mode: 'remote',
    profile: 'default',
    targetProfile: 'default'
  })
  mocks.request.mockResolvedValue(filePage())
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('Desktop Files v2 acceptance regressions', () => {
  // The real transport already bounds reads at 30s; this tests the tighter v2 UX budget.
  it('releases a download row at the tighter v2 deadline', async () => {
    vi.useFakeTimers()
    const held = deferred<Attachment>()
    mocks.read.mockReturnValue(held.promise)
    render(
      <GroupAttachmentDownload
        attachment={parsedFilePage().items[0].attachment}
        group="Core"
        message={{ eventId: 'event-20', roomId: 'room-1' } as GroupMessage}
      />
    )
    const button = screen.getByRole('button', { name: 'Download file-20.txt' }) as HTMLButtonElement
    fireEvent.click(button)
    expect(button.disabled).toBe(true)
    await act(async () => vi.advanceTimersByTimeAsync(10_001))
    expect(button.disabled).toBe(false)
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('offline Retry directly consumes its successful page, independent of capability side effects', async () => {
    const loadPage = vi.fn().mockResolvedValue(parsedFilePage())
    render(<SharedFilesDialog {...props} availability="offline" loadPage={loadPage} />)
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(1))
    await act(async () => Promise.resolve())
    expect(screen.queryByText('file-20.txt')).not.toBeNull()
  })

  it('C7 a capability flap keeps an older snapshot and does not refetch on recovery', async () => {
    const loadPage = vi
      .fn()
      .mockResolvedValueOnce(parsedFilePage([fileItem(20)], true))
      .mockResolvedValueOnce(parsedFilePage([fileItem(19, 'older.txt')]))
      .mockResolvedValue(parsedFilePage())

    const view = render(<SharedFilesDialog {...props} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: /Older/ }))
    await screen.findByText('older.txt')
    view.rerender(<SharedFilesDialog {...props} availability="offline" loadPage={loadPage} />)
    expect(screen.queryByText('older.txt')).not.toBeNull()
    view.rerender(<SharedFilesDialog {...props} loadPage={loadPage} />)
    await act(async () => new Promise(resolve => setTimeout(resolve, 20)))
    expect(loadPage).toHaveBeenCalledTimes(2)
    expect(screen.queryByText('older.txt')).not.toBeNull()
  })

  it.each(['classic', 'older host'])('keeps the Files entry visible for %s', mode => {
    const room = mode === 'classic' ? { ...FILE_ROOM, hosted: null, continuityMode: 'desktop' as const } : FILE_ROOM
    $hostedRoomCapabilities.set({})
    render(<SharedFilesControl group="Core" room={room} />)
    expect(screen.queryByRole('button', { name: 'Files' })).not.toBeNull()
  })

  it('focuses search when the dialog opens, before results arrive', () => {
    render(<SharedFilesDialog {...props} loadPage={() => new Promise(() => undefined)} />)
    const input = screen.queryByRole('textbox')
    expect(input).not.toBeNull()
    expect(globalThis.document.activeElement).toBe(input)
  })

  it('ordinary room messages never offer Show latest', async () => {
    const view = render(<SharedFilesControl group="Core" room={FILE_ROOM} />)
    fireEvent.click(screen.getByRole('button', { name: /[Ff]iles/ }))
    await screen.findByText('file-20.txt')
    view.rerender(<SharedFilesControl group="Core" room={{ ...FILE_ROOM, hostedSeq: 21 }} />)
    expect(screen.queryByRole('button', { name: 'Show latest' })).toBeNull()
  })

  it('accepts a bounded 4096-character opaque cursor from the coordinated contract', () => {
    expect(() => parseGroupFilesPage({ ...filePage([], true), next_cursor: 'x'.repeat(4096) })).not.toThrow()
  })

  it('does not confuse delimiter-containing room scopes or keep the old query', async () => {
    const loadPage = vi.fn().mockResolvedValue(parsedFilePage())
    const view = render(<SharedFilesDialog {...props} group="A:B" loadPage={loadPage} roomId="C" />)
    await screen.findByText('file-20.txt')
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'old-query' } })
    view.rerender(<SharedFilesDialog {...props} group="A" loadPage={loadPage} roomId="B:C" />)
    expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe('')
    await waitFor(() => expect(loadPage).toHaveBeenLastCalledWith('A', { limit: 8 }, expect.any(AbortSignal)))
  })

  it('rejects a repeated sparse cursor instead of allowing an Older loop', async () => {
    const first = { ...parsedFilePage([fileItem(20)], true), nextCursor: 'A' }
    const empty = { ...parsedFilePage([], true), nextCursor: 'B' }
    const cycle = { ...parsedFilePage([], true), nextCursor: 'A' }
    const loadPage = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(empty).mockResolvedValueOnce(cycle)
    render(<SharedFilesDialog {...props} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await screen.findByText('No files on this page')
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await screen.findAllByText('Refresh the file list to continue.')
    expect(screen.getAllByRole('button', { name: 'Show latest' }).length).toBeGreaterThan(0)
  })
})
