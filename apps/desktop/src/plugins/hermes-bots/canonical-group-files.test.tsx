import { webcrypto } from 'node:crypto'

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { expectDownloaded, observeDownloads } from './canonical-download-test-utils'
import { deferred, FILE_BINDING, fileItem, filePage } from './canonical-files-test-fixtures'
import backend from './canonical-files-test-fixtures.backend.json'
import { CanonicalGroupFiles } from './canonical-group-files'
import { registerCanonicalGroup } from './canonical-group-registry'
import { GroupChatWorkspace } from './group-chat-view'

const { request, activation } = vi.hoisted(() => ({ request: vi.fn(), activation: { epoch: 1 } }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return { ...sdk, gatewayActivationEpoch: () => activation.epoch, host: { ...sdk.host, requestProfile: request }, useI18n: () => ({ locale: 'en', t: en }) }
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
  activation.epoch = 1
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

it('keeps cached file selections fenced to the activation that listed them', async () => {
  request.mockResolvedValueOnce(filePage()).mockResolvedValue(await downloadReceipt())
  render(<CanonicalGroupFiles {...props} />)
  openFiles()
  const button = await screen.findByRole('button', { name: 'Download: file-20.txt' })
  activation.epoch++
  fireEvent.click(button)
  await screen.findByText('Refresh the file list to continue.')
  expect(request).toHaveBeenCalledTimes(1)
  expect(observed.create).not.toHaveBeenCalled()
})

it.each([
  [{ code: -32601 }, "File browsing isn't available for this Group Chat yet."],
  [{ data: { reason: 'runtime_coordination_required' } }, "File browsing isn't available for this Group Chat yet."],
  [new Error('failed read'), 'Files could not be loaded.']
])('shows unavailable/error with explicit Retry and no alternate owner', async (error, message) => {
  const held = deferred<unknown>()
  request.mockReturnValueOnce(held.promise).mockResolvedValue(filePage([]))
  render(<CanonicalGroupFiles {...props} />)
  openFiles()
  await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
  expect(screen.getByRole('textbox', { name: 'Search files' })).toBeTruthy()
  await act(async () => held.reject(error))
  await screen.findByText(message as string)
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByText('No files shared yet.')
  expect(request.mock.calls.every(call => call[1] === 'groups.attachment.list')).toBe(true)
})

// Captured from the exact coherent backend's handler/catalog/store, using inert
// synthetic user/member publications. No model, output producer or peer grant ran.
it('browses current backend metadata through registry, real client and named Save without a driver', async () => {
  const binding = { ...FILE_BINDING, roomId: backend.first.room_id }
  const selected = backend.downloads.find(file => file.event_id === 'event-2')!
  let appended = false
  let latest = false
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {
      return { room: { name: 'Read-only room', authority_gateway_id: backend.first.authority.gateway_id, authority_epoch: 1 } }
    }

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.attachment.list') {
      if (params.query === 'report') {return backend.filename}

      if (params.query === 'Analyst') {return backend.sharer}

      if (params.cursor) {
        expect(params.cursor).toBe(backend.first.next_cursor)

        return appended ? backend.continuation_after_append : backend.older
      }

      return latest ? backend.latest : backend.first
    }

    if (method === 'groups.attachment.download') {
      expect(params).toEqual({ room_id: binding.roomId, event_id: selected.event_id,
        attachment_id: selected.attachment_id, profile: binding.profile,
        authority_gateway_id: selected.authority.gateway_id, authority_epoch: selected.authority.epoch })

      return selected
    }

    throw new Error(`Unexpected method ${method}`)
  })
  const group = registerCanonicalGroup(binding, { room_id: binding.roomId, name: 'Read-only room', members: [] })
  render(<GroupChatWorkspace group={group} members={[]} />)
  await screen.findByRole('heading', { name: 'Read-only room' })
  openFiles()
  await waitFor(() => expect(rows()).toHaveLength(backend.first.items.length))
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await waitFor(() => expect(rows()).toHaveLength(backend.older.items.length))
  expect(rows()[0].textContent).toContain(backend.older.items[0].name)
  fireEvent.click(screen.getByRole('button', { name: 'Newer' }))
  expect(rows()).toHaveLength(backend.first.items.length)
  fireEvent.change(screen.getByRole('textbox', { name: 'Search files' }), { target: { value: 'report' } })
  await waitFor(() => expect(rows()).toHaveLength(backend.filename.items.length))
  expect(rows().every(row => row.textContent?.includes('report.txt'))).toBe(true)
  fireEvent.change(screen.getByRole('textbox', { name: 'Search files' }), { target: { value: 'Analyst' } })
  await screen.findByText('notes-2-1.txt')
  expect(rows()).toHaveLength(backend.sharer.items.length)
  expect(rows().every(row => row.textContent?.includes('Analyst'))).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: 'Download: report.txt' }))
  await waitFor(() => expect(observed.downloads).toHaveLength(1))
  await expectDownloaded(observed, Uint8Array.from(new TextEncoder().encode('synthetic version 2, file 0\n')), selected.name, selected.mime)
  expect(save).not.toHaveBeenCalled()

  appended = true
  fireEvent.change(screen.getByRole('textbox', { name: 'Search files' }), { target: { value: '' } })
  await waitFor(() => expect(rows()).toHaveLength(backend.first.items.length))
  fireEvent.click(screen.getByRole('button', { name: 'Older' }))
  await screen.findByRole('button', { name: 'Show latest' })
  expect(rows()[0].textContent).toContain(backend.older.items[0].name)
  latest = true
  fireEvent.click(screen.getByRole('button', { name: 'Show latest' }))
  await screen.findByText('notes-4-1.txt')
  const listCalls = request.mock.calls.filter(call => call[1] === 'groups.attachment.list')
  expect(listCalls.at(-1)![2]).not.toHaveProperty('cursor')
  expect(request.mock.calls.every(call => ['groups.state', 'groups.log', 'groups.attachment.list', 'groups.attachment.download'].includes(call[1]))).toBe(true)
  expect(request.mock.calls.every(call => call[0].connectionId === binding.connectionId && call[0].targetProfile === binding.profile)).toBe(true)
})

it.each(['authority', 'denied', 'missing', 'malformed', 'unchanged', 'stale-authority', 'stale-denied'] as const)(
  'the current workspace writer fences Save before React commit: %s', async change => {
    const digest = deferred<ArrayBuffer>()
    const hash = await webcrypto.subtle.digest('SHA-256', new Uint8Array([65]))
    const receipt = await downloadReceipt()
    let epoch = 1
    let gatewayId: string | undefined = 'install:home'
    let denied = false
    let name = 'Before poll'
    let staleState: Promise<unknown> | undefined
    const viable = change === 'unchanged' || change.startsWith('stale-')
    // This is the decoded-RPC seam, not a mock of the workspace writer or React.
    request.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        if (staleState) {
          const result = staleState
          staleState = undefined

          return result
        }

        if (denied) { throw { code: 401, data: { reason: 'permission_denied' } } }

        return { room: { name, authority_gateway_id: gatewayId, authority_epoch: epoch } }
      }

      if (method === 'groups.log') { return { events: [] } }

      if (method === 'groups.attachment.list') { return { ...filePage(), authority: { gateway_id: gatewayId, epoch } } }

      if (method === 'groups.attachment.download') { return { ...receipt, authority: { gateway_id: gatewayId, epoch } } }
      throw new Error(`Unexpected method ${method}`)
    })
    // Observe the real poll's next scheduling edge: it happens only AFTER the
    // current refresh writer has settled. Hold React's act commit, not the writer.
    const timers = globalThis.setTimeout
    let poll: (() => void) | undefined
    let settled = deferred<void>()
    vi.spyOn(globalThis, 'setTimeout').mockImplementation(((callback: () => void, delay?: number, ...args: unknown[]) => {
      if (delay === 2000) {
        poll = callback
        settled.resolve()

        return 0 as unknown as ReturnType<typeof setTimeout>
      }

      return timers(callback, delay, ...args)
    }) as typeof setTimeout)
    const group = registerCanonicalGroup(FILE_BINDING, { room_id: FILE_BINDING.roomId, name, members: [] })
    render(<GroupChatWorkspace group={group} members={[]} />)
    await screen.findByRole('heading', { name })
    openFiles()
    const digestSpy = vi.spyOn(crypto.subtle, 'digest').mockReturnValueOnce(digest.promise)
    fireEvent.click(await screen.findByRole('button', { name: 'Download: file-20.txt' }))
    await waitFor(() => expect(digestSpy).toHaveBeenCalledOnce())
    const oldDialog = screen.getByRole('dialog')

    name = 'After poll'

    if (change === 'authority') { epoch = 2 }

    if (change === 'denied') { denied = true }

    if (change === 'missing') { gatewayId = undefined }

    if (change === 'malformed') { epoch = -1 }
    await act(async () => {
      const oldPoll = deferred<unknown>()

      if (change.startsWith('stale-')) {
        staleState = oldPoll.promise
        poll!()
      }

      settled = deferred<void>()
      poll!()
      await settled.promise

      if (change.startsWith('stale-')) {
        settled = deferred<void>()

        if (change === 'stale-denied') {oldPoll.reject({ code: 401, data: { reason: 'permission_denied' } })}
        else {oldPoll.resolve({ room: { name: 'Obsolete poll', authority_gateway_id: 'install:other', authority_epoch: 2 } })}

        await settled.promise
      }

      expect(screen.getByRole('dialog')).toBe(oldDialog)
      expect(oldDialog.textContent).toContain('Before poll')
      expect(oldDialog.textContent).not.toContain('After poll')
      digest.resolve(hash)
      // Drain the real digest/read/Save continuation while act still owns the
      // uncommitted render queue. The unchanged-authority row proves this drains Save.
      await new Promise<void>(resolve => timers(resolve, 0))
      expect(screen.getByRole('dialog')).toBe(oldDialog)
      expect(observed.create).toHaveBeenCalledTimes(viable ? 1 : 0)
      expect(observed.click).toHaveBeenCalledTimes(viable ? 1 : 0)
    })

    if (!viable) {
      // Recover through the same real writer, then select under its NEW authority.
      epoch = change === 'denied' ? 1 : 3
      gatewayId = 'install:home'
      denied = false
      await act(async () => {
        settled = deferred<void>()
        poll!()
        await settled.promise
      })
      expect(screen.queryByRole('dialog')).toBeNull()
      openFiles()
      fireEvent.click(await screen.findByRole('button', { name: 'Download: file-20.txt' }))
      await waitFor(() => expect(observed.click).toHaveBeenCalledOnce())
      expect(request.mock.calls.filter(call => call[1] === 'groups.attachment.download').at(-1)![2]).toMatchObject({ authority_epoch: epoch })
    }

    expect(save).not.toHaveBeenCalled()
  }
)

it.each(['list', 'download', 'digest'] as const)(
  'retires an old %s on a same-room authoritative epoch snapshot without affecting the replacement', async boundary => {
    const held = deferred<unknown>()
    const digest = deferred<ArrayBuffer>()
    const receipt = await downloadReceipt()
    const hash = await webcrypto.subtle.digest('SHA-256', new Uint8Array([65]))
    let epoch = 1
    request.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return { room: { name: 'Same room', authority_gateway_id: 'install:home', authority_epoch: epoch } }
      }

      if (method === 'groups.log') {return { events: [] }}

      if (method === 'groups.attachment.list') {
        if (epoch === 1 && boundary === 'list') {return held.promise}

        return { ...filePage(), authority: { gateway_id: 'install:home', epoch } }
      }

      if (method === 'groups.attachment.download') {
        return boundary === 'download' ? held.promise : receipt
      }

      throw new Error(`Unexpected method ${method}`)
    })
    const group = registerCanonicalGroup(FILE_BINDING, { room_id: FILE_BINDING.roomId, name: 'Same room', members: [] })
    render(<GroupChatWorkspace group={group} members={[]} />)
    await screen.findByRole('heading', { name: 'Same room' })
    openFiles()
    await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.attachment.list')).toBe(true))

    if (boundary !== 'list') {
      if (boundary === 'digest') {vi.spyOn(crypto.subtle, 'digest').mockReturnValueOnce(digest.promise)}
      fireEvent.click(await screen.findByRole('button', { name: 'Download: file-20.txt' }))
      await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.attachment.download')).toBe(true))

      if (boundary === 'digest') {await waitFor(() => expect(crypto.subtle.digest).toHaveBeenCalled())}
    }

    // The real workspace poll commits a replacement authority for the SAME room.
    epoch = 2
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull(), { timeout: 4000 })
    openFiles()
    await screen.findByText('file-20.txt')
    await act(async () => {
      if (boundary === 'digest') {digest.resolve(hash)} else {
        // A stale canonical denial/401 must not clear the replacement's catalog.
        held.reject({ code: 401, data: { reason: 'permission_denied' } })
      }
    })
    expect(screen.getByText('file-20.txt')).toBeTruthy()
    expect(screen.queryByText('Files are unavailable for this Group Chat.')).toBeNull()
    expect((screen.getByRole('button', { name: 'Download: file-20.txt' }) as HTMLButtonElement).disabled).toBe(false)
    expect(observed.create).not.toHaveBeenCalled()
    expect(save).not.toHaveBeenCalled()
    expect(request.mock.calls.filter(call => call[1] === 'groups.attachment.list').at(-1)![2]).toMatchObject({ authority_epoch: 2 })
  }
)
