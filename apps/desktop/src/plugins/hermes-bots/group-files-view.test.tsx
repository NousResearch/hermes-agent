import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $groupChats } from './group-chat'
import type { GroupFilesPage } from './group-files-client'
import { deferred, FILE_ROOM, fileItem, filePage, parsedFilePage } from './group-files-test-fixtures'
import { SharedFilesControl, SharedFilesDialog } from './group-files-view'
import { $hostedRoomCapabilities } from './hosted-room-capability-state'
import { classifyHostedRoomCapability } from './hosted-room-client'
import { translateBots } from './i18n-test-helper'

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

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  vi.clearAllMocks()
  $hostedRoomCapabilities.set({})
  $groupChats.set({ Core: FILE_ROOM })
  mocks.route.mockResolvedValue({
    connectionId: 'gateway-a',
    profile: 'default',
    targetProfile: 'default',
    mode: 'remote'
  })
  mocks.request.mockResolvedValue(filePage())
})

afterEach(() => cleanup())

const dialogProps = {
  availability: 'available' as const,
  group: 'Core',
  latestSeq: 20,
  onClose: vi.fn(),
  open: true,
  roomId: 'room-1'
}

const files = () => screen.queryAllByRole('listitem').map(row => row.querySelector('[title]')?.textContent)

describe('shared-files states and recovery', () => {
  it('uses the real Dialog loading/empty states and names its Group Chat', async () => {
    const response = deferred<GroupFilesPage>()
    const loadPage = vi.fn(() => response.promise)
    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)

    expect(screen.getByRole('dialog').textContent).toContain('Core')
    expect(screen.getByRole('status', { name: 'Loading files' })).toBeTruthy()
    expect(screen.queryByText('No files shared yet.')).toBeNull()
    await waitFor(() => expect(loadPage).toHaveBeenCalledWith('Core', { limit: 8 }, expect.any(AbortSignal)))
    await act(async () => response.resolve(parsedFilePage([])))
    expect(screen.getByText('No files shared yet.')).toBeTruthy()
    expect(screen.queryByRole('status')).toBeNull()
    expect(screen.getByRole('textbox', { name: 'Search files' })).toBeTruthy()
  })

  it('recovers a failed first-page request through Retry', async () => {
    const loadPage = vi.fn().mockRejectedValueOnce(new Error('read failed')).mockResolvedValueOnce(parsedFilePage())
    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('Files could not be loaded.')
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByText('file-20.txt')
    expect(loadPage).toHaveBeenCalledTimes(2)
  })

  it('distinguishes offline from unavailable and makes offline Retry probe again', async () => {
    const response = deferred<GroupFilesPage>()
    const loadPage = vi.fn(() => response.promise)
    const view = render(<SharedFilesDialog {...dialogProps} availability="offline" loadPage={loadPage} />)
    expect(screen.getByText('Files are temporarily unavailable.')).toBeTruthy()
    expect(loadPage).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(loadPage).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('status', { name: 'Loading files' })).toBeTruthy()
    await act(async () => response.reject(new Error('still offline')))
    expect(screen.getByText('Files are temporarily unavailable.')).toBeTruthy()
    view.rerender(<SharedFilesDialog {...dialogProps} availability="unavailable" loadPage={loadPage} />)
    expect(screen.getByText("File browsing isn't available for this Group Chat yet.")).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect(loadPage).toHaveBeenCalledTimes(1)
  })

  it('distinguishes an exhausted search from an empty Group Chat', async () => {
    const loadPage = vi.fn().mockResolvedValueOnce(parsedFilePage()).mockResolvedValueOnce(parsedFilePage([]))
    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    fireEvent.change(screen.getByRole('textbox', { name: 'Search files' }), { target: { value: 'missing' } })
    await screen.findByText('No matching files.')
    expect(screen.queryByText('No files shared yet.')).toBeNull()
    expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe('missing')
    expect(loadPage).toHaveBeenLastCalledWith('Core', { limit: 8, query: 'missing' }, expect.any(AbortSignal))
  })

  it('retains navigation on an empty bounded scan page', async () => {
    const loadPage = vi
      .fn()
      .mockResolvedValueOnce(parsedFilePage([], true))
      .mockResolvedValueOnce(parsedFilePage([fileItem(10)]))

    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('No files on this page')
    expect(screen.queryByText('No files shared yet.')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await screen.findByText('file-10.txt')
    expect(loadPage).toHaveBeenLastCalledWith('Core', { limit: 8, cursor: 'cursor-after-20' }, expect.any(AbortSignal))
  })
})

describe('query and room request fences', () => {
  it('ignores an Older response after a new search replaces the page state', async () => {
    const older = deferred<GroupFilesPage>()

    const loadPage = vi
      .fn()
      .mockResolvedValueOnce(parsedFilePage([fileItem()], true))
      .mockReturnValueOnce(older.promise)
      .mockResolvedValueOnce(parsedFilePage([fileItem(18, 'found.txt')]))

    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(2))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'found' } })
    await screen.findByText('found.txt')
    await act(async () => older.resolve(parsedFilePage([fileItem(19, 'stale-page.txt')])))
    expect(files()).toEqual(['found.txt'])
    expect((screen.getByRole('button', { name: 'Newer' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('debounces typing and ignores an older search response', async () => {
    const old = deferred<GroupFilesPage>()
    const current = deferred<GroupFilesPage>()

    const loadPage = vi
      .fn()
      .mockResolvedValueOnce(parsedFilePage())
      .mockReturnValueOnce(old.promise)
      .mockReturnValueOnce(current.promise)

    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    const input = screen.getByRole('textbox')
    fireEvent.change(input, { target: { value: 'o' } })
    fireEvent.change(input, { target: { value: 'ol' } })
    fireEvent.change(input, { target: { value: 'old' } })
    expect(loadPage).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(2))
    fireEvent.change(input, { target: { value: 'new' } })
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(3))
    await act(async () => current.resolve(parsedFilePage([fileItem(18, 'new.txt')])))
    await act(async () => old.resolve(parsedFilePage([fileItem(19, 'old.txt')])))
    expect(files()).toEqual(['new.txt'])
    expect((input as HTMLInputElement).value).toBe('new')
  })

  it('clearing a search during debounce cannot strand the loading state', async () => {
    const loadPage = vi.fn().mockResolvedValue(parsedFilePage())
    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    const input = screen.getByRole('textbox')
    fireEvent.change(input, { target: { value: 'unfinished' } })
    fireEvent.change(input, { target: { value: '' } })
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByRole('list').getAttribute('aria-busy')).toBe('false'))
  })

  it('ignores a prior room response and clears its query/page state', async () => {
    const old = deferred<GroupFilesPage>()
    const next = deferred<GroupFilesPage>()
    const loadPage = vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise)
    const view = render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(1))
    view.rerender(<SharedFilesDialog {...dialogProps} group="Other" loadPage={loadPage} roomId="room-2" />)
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(2))
    await act(async () => next.resolve(parsedFilePage([fileItem(20, 'other.txt')])))
    await act(async () => old.resolve(parsedFilePage([fileItem(20, 'old-room.txt')])))
    expect(files()).toEqual(['other.txt'])
    expect(screen.getByRole('dialog').textContent).toContain('Other')
  })

  it('ignores work after close and starts fresh when reopened', async () => {
    const old = deferred<GroupFilesPage>()

    const loadPage = vi
      .fn()
      .mockReturnValueOnce(old.promise)
      .mockResolvedValueOnce(parsedFilePage([fileItem(20, 'fresh.txt')]))

    const view = render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(1))
    view.rerender(<SharedFilesDialog {...dialogProps} loadPage={loadPage} open={false} />)
    await act(async () => old.resolve(parsedFilePage([fileItem(20, 'closed.txt')])))
    expect(screen.queryByRole('dialog')).toBeNull()
    view.rerender(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('fresh.txt')
    expect(screen.queryByText('closed.txt')).toBeNull()
  })

  it('bounds Unicode search text to 255 code points', async () => {
    const loadPage = vi.fn().mockResolvedValue(parsedFilePage())
    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '研'.repeat(300) } })
    await waitFor(() => expect(loadPage).toHaveBeenCalledTimes(2))
    expect(loadPage.mock.calls[1][1].query).toBe('研'.repeat(255))
  })
})

describe('snapshot navigation and restart recovery', () => {
  it('keeps same-name versions distinct and Newer stable across a concurrent arrival', async () => {
    const initial = parsedFilePage(
      Array.from({ length: 8 }, (_, index) => fileItem(20 - index, 'report.txt')),
      true
    )

    const older = parsedFilePage([fileItem(12, 'report.txt')])
    const latest = parsedFilePage([fileItem(21, 'new-arrival.txt')], false, 21)
    const loadPage = vi.fn().mockResolvedValueOnce(initial).mockResolvedValueOnce(older).mockResolvedValueOnce(latest)
    const view = render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await waitFor(() => expect(files()).toHaveLength(8))
    expect(screen.getAllByRole('button', { name: 'Download report.txt' })).toHaveLength(8)
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await waitFor(() => expect(files()).toHaveLength(1))
    view.rerender(<SharedFilesDialog {...dialogProps} latestSeq={21} loadPage={loadPage} />)
    expect(screen.getByRole('button', { name: 'Show latest' })).toBeTruthy()
    expect(files()).toEqual(['report.txt'])
    fireEvent.click(screen.getByRole('button', { name: 'Newer' }))
    expect(files()).toHaveLength(8)
    expect(loadPage).toHaveBeenCalledTimes(2)
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    expect(files()).toHaveLength(1)
    expect(loadPage).toHaveBeenCalledTimes(2)
    fireEvent.click(screen.getByRole('button', { name: 'Show latest' }))
    await screen.findByText('new-arrival.txt')
    expect(loadPage).toHaveBeenLastCalledWith('Core', { limit: 8 }, expect.any(AbortSignal))
  })

  it('offers Show latest after a restart invalidates an Older cursor', async () => {
    const loadPage = vi
      .fn()
      .mockResolvedValueOnce(parsedFilePage([fileItem()], true))
      .mockRejectedValueOnce(new Error('attachment list cursor is invalid'))
      .mockResolvedValueOnce(parsedFilePage([fileItem(20, 'fresh-snapshot.txt')]))

    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await screen.findByText('Refresh the file list to continue.')
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show latest' }))
    await screen.findByText('fresh-snapshot.txt')
    expect(loadPage.mock.calls.map(call => call[1]?.cursor)).toEqual([undefined, 'cursor-after-20', undefined])
  })

  it('offers new arrivals without replacing a focused row or navigating', async () => {
    const loadPage = vi.fn().mockResolvedValueOnce(parsedFilePage())
    const onClose = vi.fn()
    const view = render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} onClose={onClose} />)
    await screen.findByText('file-20.txt')
    const download = screen.getByRole('button', { name: 'Download file-20.txt' })
    download.focus()
    view.rerender(<SharedFilesDialog {...dialogProps} latestSeq={21} loadPage={loadPage} onClose={onClose} />)
    expect(loadPage).toHaveBeenCalledTimes(1)
    expect(globalThis.document.activeElement).toBe(download)
    expect(screen.getByRole('button', { name: 'Show latest' })).toBeTruthy()
    expect(files()).toEqual(['file-20.txt'])
    expect(onClose).not.toHaveBeenCalled()
  })
})

describe('capability, keyboard and narrow layout', () => {
  it('invalidates an open pending list when browsing capability disappears', async () => {
    const pending = deferred<ReturnType<typeof filePage>>()
    mocks.request.mockReturnValue(pending.promise)

    const capability = classifyHostedRoomCapability(
      {
        authority_gateway_id: 'install:home',
        driver: true,
        persistent_process: true,
        features: ['attachment_metadata_catalog']
      },
      { connectionId: 'gateway-a' }
    )

    $hostedRoomCapabilities.set({ 'gateway-a': capability })
    render(<SharedFilesControl group="Core" room={FILE_ROOM} />)
    fireEvent.click(screen.getByRole('button', { name: 'Files' }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(1))
    act(() =>
      $hostedRoomCapabilities.set({
        'gateway-a': { ...capability, limits: { ...capability.limits, attachmentList: false } }
      })
    )
    expect(screen.getByText("File browsing isn't available for this Group Chat yet.")).toBeTruthy()
    await act(async () => pending.resolve(filePage()))
    expect(screen.queryByText('file-20.txt')).toBeNull()
  })

  it('keeps the Files entry for classic rooms and attachment-only older gateways', () => {
    const capability = classifyHostedRoomCapability(
      {
        authority_gateway_id: 'install:home',
        driver: true,
        persistent_process: true,
        methods: ['groups.attachment.read', 'groups.attachment.put']
      },
      { connectionId: 'gateway-a' }
    )

    $hostedRoomCapabilities.set({ 'gateway-a': capability })
    const view = render(<SharedFilesControl group="Core" room={FILE_ROOM} />)
    expect(screen.getByRole('button', { name: 'Files' })).toBeTruthy()
    act(() =>
      $hostedRoomCapabilities.set({
        'gateway-a': classifyHostedRoomCapability(
          {
            authority_gateway_id: 'install:home',
            driver: true,
            persistent_process: true,
            features: ['attachment_metadata_catalog']
          },
          { connectionId: 'gateway-a' }
        )
      })
    )
    expect(screen.getByRole('button', { name: 'Files' })).toBeTruthy()
    view.rerender(
      <SharedFilesControl group="Classic" room={{ ...FILE_ROOM, hosted: null, continuityMode: 'desktop' }} />
    )
    expect(screen.getByRole('button', { name: 'Files' })).toBeTruthy()
  })

  it('opens only from the header action and Escape closes the actual Dialog once', async () => {
    $hostedRoomCapabilities.set({
      'gateway-a': classifyHostedRoomCapability(
        {
          authority_gateway_id: 'install:home',
          driver: true,
          persistent_process: true,
          features: ['attachment_metadata_catalog']
        },
        { connectionId: 'gateway-a' }
      )
    })
    render(<SharedFilesControl group="Core" room={FILE_ROOM} />)
    expect(screen.queryByRole('dialog')).toBeNull()
    const trigger = screen.getByRole('button', { name: 'Files' })
    expect(trigger.tagName).toBe('BUTTON')
    expect(trigger.getAttribute('data-size')).toBe('icon-sm')
    fireEvent.click(trigger)
    await screen.findByText('file-20.txt')
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape', code: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })

  it('keeps long Unicode names inside flat, bounded rows with native filename titles', async () => {
    const name = '資料é'.repeat(70) + '.pdf'

    const loadPage = vi
      .fn()
      .mockResolvedValue(parsedFilePage([{ ...fileItem(20, name), kind: 'pdf', mime: 'application/pdf' }]))

    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    const filename = await screen.findByText(name)
    const row = screen.getByRole('listitem')
    const action = screen.getByRole('button', { name: `Download ${name}` })
    expect(filename.getAttribute('title')).toBe(name)
    expect(filename.className).toContain('truncate')
    expect(filename.parentElement?.className).toContain('min-w-0')
    expect(row.className).toContain('min-w-0')
    expect(row.className).toContain('min-h-12')
    expect(row.className).not.toMatch(/rounded|border|shadow|bg-/)
    expect(action.getAttribute('data-size')).toBe('icon-xs')
    expect(action.getAttribute('title')).toBeNull()
    expect(screen.getByRole('dialog').className).toContain('max-w-xl')
    expect(screen.getByRole('dialog').className).toContain('h-[min(36rem,85vh)]')
    expect(screen.getByRole('list').parentElement?.className).toContain('flex-1')
    expect(screen.getByRole('list').className).toContain('flex-1')
    expect(screen.getByRole('button', { name: 'Older' }).closest('[class*="min-h-8"]')?.previousElementSibling).toBe(
      screen.getByRole('list')
    )
    expect(row.querySelector('.codicon-file-pdf')).toBeTruthy()
  })

  it('keeps keyboard focus inside the real Dialog at both Tab boundaries', async () => {
    const loadPage = vi.fn().mockResolvedValue(parsedFilePage())
    render(<SharedFilesDialog {...dialogProps} loadPage={loadPage} />)
    await screen.findByText('file-20.txt')
    const input = screen.getByRole('textbox')
    const close = screen.getByRole('button', { name: 'Close' })
    close.focus()
    fireEvent.keyDown(close, { key: 'Tab', code: 'Tab' })
    expect(globalThis.document.activeElement).toBe(input)
    fireEvent.keyDown(input, { key: 'Tab', code: 'Tab', shiftKey: true })
    expect(globalThis.document.activeElement).toBe(close)
  })
})
