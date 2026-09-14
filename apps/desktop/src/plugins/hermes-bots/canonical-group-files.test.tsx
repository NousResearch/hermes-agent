import { webcrypto } from 'node:crypto'

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { expectDownloaded, observeDownloads } from './canonical-download-test-utils'
import { deferred, FILE_BINDING, fileItem, filePage } from './canonical-files-test-fixtures'
import { CanonicalGroupFiles } from './canonical-group-files'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return { ...sdk, host: { ...sdk.host, requestProfile: request }, useI18n: () => ({ locale: 'en', t: en }) }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return {
    useCanonicalGroupLabels: () => ({
      ...CANONICAL_GROUP_LOCALES.en,
      back: 'Back',
      refresh: 'Refresh',
      retry: 'Retry',
      send: 'Send',
      stop: 'Stop',
      download: 'Download',
      discard: 'Discard',
      cancel: 'Cancel'
    })
  }
})

const props = { binding: FILE_BINDING, name: 'Review room', authority: { gatewayId: 'install:home', epoch: 1 } }
const originalDesktop = window.hermesDesktop
const save = vi.fn()
let observed: ReturnType<typeof observeDownloads>
beforeEach(() => {
  request.mockReset()
  save.mockReset().mockResolvedValue(undefined)
  vi.stubGlobal('crypto', webcrypto)
  observed = observeDownloads()
  window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
  window.hermesDesktop = originalDesktop
  localStorage.clear()
})

function openFiles() {
  fireEvent.click(screen.getByRole('button', { name: 'Files' }))
}

function rows() {
  return screen.queryAllByRole('listitem')
}

async function downloadReceipt(item = fileItem()) {
  const bytes = new Uint8Array([65])

  return {
    ...item,
    room_id: 'room-1',
    authority: { gateway_id: 'install:home', epoch: 1 },
    data_base64: 'QQ==',
    sha256: Buffer.from(await webcrypto.subtle.digest('SHA-256', bytes)).toString('hex')
  }
}

it('preserves donor newest-first versions, cached pages and current page across observations', async () => {
  request
    .mockResolvedValueOnce(filePage([fileItem(20, 'same.txt', 1), fileItem(19, 'same.txt', 2)], true))
    .mockResolvedValueOnce(filePage([fileItem(18, 'older.txt', 3)]))
  const view = render(<CanonicalGroupFiles {...props} />)
  openFiles()
  await waitFor(() => expect(rows()).toHaveLength(2))
  expect(screen.getByRole('dialog').textContent).toContain('Review room')
  expect(rows().map(row => row.querySelector('bdi')?.textContent)).toEqual(['same.txt', 'same.txt'])
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await screen.findByText('older.txt')
  expect(request.mock.calls[1][2]).toMatchObject({
    cursor: 'cursor-after-19',
    authority_gateway_id: 'install:home',
    authority_epoch: 1
  })
  view.rerender(
    <CanonicalGroupFiles
      {...props}
      authority={{ ...props.authority }}
      binding={{ ...FILE_BINDING }}
      latestFileSeq={21}
    />
  )
  expect(screen.getByText('older.txt')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Newer' }))
  expect(rows()).toHaveLength(2)
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  expect(screen.getByText('older.txt')).toBeTruthy()
  expect(request).toHaveBeenCalledTimes(2)
  request.mockResolvedValueOnce(filePage([fileItem(21, 'latest.txt')], false, 21))
  fireEvent.click(screen.getByRole('button', { name: 'Show latest' }))
  await screen.findByText('latest.txt')
  expect(request.mock.calls[2][2]).not.toHaveProperty('cursor')
})

it('retires old queries immediately and ignores their late success or denial', async () => {
  const old = deferred<unknown>()
  request.mockReturnValueOnce(old.promise).mockResolvedValue(filePage([fileItem(18, 'new-query.txt')]))
  render(<CanonicalGroupFiles {...props} />)
  openFiles()
  await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
  fireEvent.change(screen.getByRole('textbox', { name: 'Search files' }), { target: { value: 'new' } })
  await screen.findByText('new-query.txt')
  await act(async () => old.reject({ code: 4001, data: { reason: 'permission_denied' } }))
  expect(screen.getByText('new-query.txt')).toBeTruthy()
  expect(request.mock.calls[1][2]).toMatchObject({ query: 'new', room_id: 'room-1', profile: 'reviewer' })
  expect(request.mock.calls[1][2]).not.toHaveProperty('cursor')
})

it('passes empty continuation pages without losing the snapshot and rejects looping cursors', async () => {
  request
    .mockResolvedValueOnce(filePage([fileItem()], true))
    .mockResolvedValueOnce({ ...filePage([], true), next_cursor: 'scanned-empty' })
    .mockResolvedValueOnce({ ...filePage([fileItem(19)], true), next_cursor: 'cursor-after-20' })
  render(<CanonicalGroupFiles {...props} />)
  openFiles()
  await screen.findByText('file-20.txt')
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await screen.findByText('No files on this page')
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await screen.findByText('Refresh the file list to continue.')
  expect(screen.queryByText('file-19.txt')).toBeNull()
  expect(request).toHaveBeenCalledTimes(3)
})

it('clears all cached pages on access denial and never delivers a concurrent late download', async () => {
  const download = deferred<unknown>()
  const response = await downloadReceipt()
  request.mockResolvedValueOnce(filePage([fileItem()], true)).mockImplementation((_route, method) => {
    if (method === 'groups.attachment.download') {
      return download.promise
    }

    return Promise.reject({ code: 4001, data: { reason: 'permission_denied' } })
  })
  render(<CanonicalGroupFiles {...props} />)
  openFiles()
  fireEvent.click(await screen.findByRole('button', { name: 'Download: file-20.txt' }))
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await screen.findByText('Files are unavailable for this Group Chat.')
  expect(rows()).toHaveLength(0)
  await act(async () => download.resolve(response))
  expect(save).not.toHaveBeenCalled()
  expect(observed.create).not.toHaveBeenCalled()
})

it.each(['close', 'profile', 'room', 'authority', 'denied'])(
  'retires native save intent when the dialog is %s',
  async change => {
    const pending = deferred<unknown>()
    const response = await downloadReceipt()
    request.mockResolvedValueOnce(filePage()).mockReturnValue(pending.promise)
    const view = render(<CanonicalGroupFiles {...props} />)
    openFiles()
    fireEvent.click(await screen.findByRole('button', { name: 'Download: file-20.txt' }))
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2))

    if (change === 'close') {
      fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    } else if (change === 'profile') {
      view.rerender(<CanonicalGroupFiles {...props} binding={{ ...FILE_BINDING, profile: 'other' }} />)
    } else if (change === 'room') {
      view.rerender(<CanonicalGroupFiles {...props} binding={{ ...FILE_BINDING, roomId: 'other' }} />)
    } else if (change === 'authority') {
      view.rerender(<CanonicalGroupFiles {...props} authority={{ ...props.authority, epoch: 2 }} />)
    } else {
      view.rerender(<CanonicalGroupFiles {...props} accessDenied />)
    }

    await act(async () => pending.resolve(response))
    expect(save).not.toHaveBeenCalled()
    expect(observed.create).not.toHaveBeenCalled()
    expect(rows()).toHaveLength(0)
  }
)

it('keeps a transiently offline snapshot until explicit Retry and does not reset its page', async () => {
  request
    .mockResolvedValueOnce(filePage([fileItem()], true))
    .mockRejectedValueOnce(new Error('connection lost'))
    .mockResolvedValueOnce(filePage([fileItem(21)], false, 21))
  const view = render(<CanonicalGroupFiles {...props} />)
  openFiles()
  await screen.findByText('file-20.txt')
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await screen.findByText('Files are temporarily unavailable.')
  expect(screen.getByText('file-20.txt')).toBeTruthy()
  view.rerender(<CanonicalGroupFiles {...props} latestFileSeq={21} />)
  expect(request).toHaveBeenCalledTimes(2)
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByText('Reconnected')
  expect(screen.getByText('file-20.txt')).toBeTruthy()
  expect(screen.queryByText('file-21.txt')).toBeNull()
  expect(request).toHaveBeenCalledTimes(3)
})

it('distinguishes exhausted search and supports clearing it through the real SearchField', async () => {
  request.mockResolvedValueOnce(filePage()).mockResolvedValueOnce(filePage([])).mockResolvedValue(filePage())
  render(<CanonicalGroupFiles {...props} />)
  openFiles()
  await screen.findByText('file-20.txt')
  fireEvent.change(screen.getByRole('textbox', { name: 'Search files' }), { target: { value: 'missing' } })
  await screen.findByText('No matching files.')
  expect(screen.queryByText('No files shared yet.')).toBeNull()
  fireEvent.click(screen.getByText('Clear search'))
  await screen.findByText('file-20.txt')
  expect((screen.getByRole('textbox', { name: 'Search files' }) as HTMLInputElement).value).toBe('')
})

it('uses the selected same-name version and keeps an individual missing file separate from room access', async () => {
  const first = fileItem(20, 'report.txt', 1)
  const second = fileItem(19, 'report.txt', 2)
  request
    .mockResolvedValueOnce(filePage([first, second]))
    .mockRejectedValueOnce({ code: 4001, data: { reason: 'attachment_not_found' } })
    .mockResolvedValueOnce(await downloadReceipt(second))
  render(<CanonicalGroupFiles {...props} />)
  openFiles()
  await waitFor(() => expect(rows()).toHaveLength(2))
  fireEvent.click(within(rows()[0]).getByRole('button'))
  await screen.findByText('This file is no longer available.')
  expect(rows()).toHaveLength(2)
  fireEvent.click(within(rows()[1]).getByRole('button'))
  await waitFor(() => expect(observed.downloads).toHaveLength(1))
  await expectDownloaded(observed, new Uint8Array([65]), 'report.txt', 'text/plain')
  expect(save).not.toHaveBeenCalled()
  expect(request.mock.calls[2][2]).toMatchObject({
    attachment_id: second.attachment_id,
    event_id: second.event_id,
    authority_epoch: 1
  })
})

it('opens usable Files from a canonical workspace without an execution driver', async () => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {
      return { room: { name: 'Read-only room', authority_gateway_id: 'install:home', authority_epoch: 1 } }
    }

    if (method === 'groups.log') {
      return { events: [] }
    }

    if (method === 'groups.attachment.list') {
      return filePage()
    }

    throw new Error(`Unexpected method ${method}`)
  })
  render(<CanonicalGroupWorkspace binding={FILE_BINDING} />)
  await screen.findByRole('heading', { name: 'Read-only room' })
  openFiles()
  await screen.findByText('file-20.txt')
  expect(
    request.mock.calls.every(call => ['groups.state', 'groups.log', 'groups.attachment.list'].includes(call[1]))
  ).toBe(true)
})
