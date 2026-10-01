import { webcrypto } from 'node:crypto'

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { expectDownloaded, observeDownloads } from './canonical-download-test-utils'

const { request, activation } = vi.hoisted(() => ({ request: vi.fn(), activation: { epoch: 0 } }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')
  const { captureGroupRequests } = await import('./group-test-utils')

  return { ...sdk, gatewayActivationEpoch: () => activation.epoch, useI18n: () => ({ locale: 'en', t: en }),
    host: { ...sdk.host, requestProfile: captureGroupRequests(request).request } }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, locale: 'en', retry: 'Retry',
    download: 'Download', cancel: 'Cancel', save: 'Save', refresh: 'Refresh', stop: 'Stop', send: 'Send' }) }
})
import { CanonicalGroupRoomActions } from './canonical-group-header'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { CANONICAL_GROUP_CAPABILITIES } from './group-test-utils'

const binding = { connectionId: 'local', profile: 'reviewer', roomId: 'room-1' }
const authority = { gateway_id: 'install:home', epoch: 1 }

const file = (seq: number, name = `file-${seq}.txt`, index = 0) => ({
  event_id: `user:${seq}`, attachment_id: `att_${(seq * 8 + index).toString(16).padStart(32, '0')}`, seq,
  manifest_index: index, kind: 'file', name, mime: 'text/plain', size: 1,
  producer: { kind: 'user', id: 'desktop', label: 'You' }, shared_at: 1_700_000_000 + seq
})

const page = (items: ReturnType<typeof file>[], cursor: null | string = null, snapshot = 20) =>
  ({ room_id: binding.roomId, authority, snapshot_seq: snapshot, items, next_cursor: cursor, has_more: cursor !== null })

const advertised = { ...CANONICAL_GROUP_CAPABILITIES, methods: [...CANONICAL_GROUP_CAPABILITIES.methods, 'groups.attachment.list'] }

let observed: ReturnType<typeof observeDownloads>
let gateway: Record<string, (params: Record<string, unknown>) => unknown>
beforeEach(() => {
  // Each test is a fresh gateway activation, so no capability read is shared between tests.
  activation.epoch++
  vi.stubGlobal('crypto', webcrypto)
  observed = observeDownloads()
  gateway = { 'groups.capabilities': () => advertised }
  request.mockImplementation(async (_route, method: string, params: Record<string, unknown>) => {
    if (!gateway[method]) {throw new Error(`Unexpected method ${method}`)}

    return gateway[method](params)
  })
})
afterEach(() => { cleanup(); request.mockReset(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

const calls = (method: string) => request.mock.calls.filter(call => call[1] === method).map(call => call[2])
const rows = () => screen.queryAllByRole('listitem')

async function openFiles(latestFileSeq = 0) {
  render(<CanonicalGroupRoomActions binding={binding} latestFileSeq={latestFileSeq} name="Review room"
    onChanged={() => undefined} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Files' }))

  return within(await screen.findByRole('dialog'))
}

async function downloadReply(item: ReturnType<typeof file>, data = 'A') {
  const bytes = new TextEncoder().encode(data)
  const digest = Buffer.from(await webcrypto.subtle.digest('SHA-256', bytes)).toString('hex')

  return { ...item, state: 'committed', sha256: digest, data_base64: Buffer.from(bytes).toString('base64') }
}

it('offers Files only when the gateway advertises the catalog', async () => {
  gateway['groups.capabilities'] = () => CANONICAL_GROUP_CAPABILITIES
  render(<CanonicalGroupRoomActions binding={binding} name="Review room" onChanged={() => undefined} />)
  await waitFor(() => expect(calls('groups.capabilities')).toHaveLength(1))
  expect(screen.queryByRole('button', { name: 'Files' })).toBeNull()
})

it('browses newest first, tells same-name versions apart and pages through one snapshot', async () => {
  gateway['groups.attachment.list'] = params => params.cursor === 'after-19'
    ? page([file(12, 'older.txt')]) : page([file(20, 'same.txt'), file(19, 'same.txt')], 'after-19')
  const dialog = await openFiles(21)
  await waitFor(() => expect(rows()).toHaveLength(2))
  expect(dialog.getByText('Review room')).toBeTruthy()
  const [newest, earlier] = rows().map(row => row.getAttribute('aria-label') ?? '')
  // Same name on one page: each row shows its own time to the second.
  expect(newest).toMatch(/^same\.txt · You · .*\d{1,2}:\d{2}:\d{2}/)
  expect(newest).not.toBe(earlier)
  expect(dialog.getByRole('button', { name: 'Show latest' })).toBeTruthy()
  fireEvent.click(dialog.getByRole('button', { name: 'Older files' }))
  await dialog.findByText('older.txt')
  fireEvent.click(dialog.getByRole('button', { name: 'Newer files' }))
  expect(rows()).toHaveLength(2)
  fireEvent.click(dialog.getByRole('button', { name: 'Older files' }))
  expect(dialog.getByText('older.txt')).toBeTruthy()
  expect(calls('groups.attachment.list')).toEqual([
    { room_id: 'room-1', limit: 8, profile: 'reviewer' },
    { room_id: 'room-1', limit: 8, cursor: 'after-19', profile: 'reviewer' }])
})

it('searches by name or sharer and ignores a superseded reply', async () => {
  let releaseFirst!: (value: unknown) => void
  gateway['groups.attachment.list'] = params => params.query === undefined
    ? new Promise(resolve => { releaseFirst = resolve })
    : params.query === 'Builder' ? page([{ ...file(7, 'plan.md'), producer: { kind: 'member', id: 'builder', label: 'Builder' } }])
      : page([])
  const dialog = await openFiles()
  await waitFor(() => expect(calls('groups.attachment.list')).toHaveLength(1))
  fireEvent.change(dialog.getByRole('textbox', { name: 'Search files' }), { target: { value: 'Builder' } })
  await dialog.findByText('plan.md')
  await act(async () => releaseFirst(page([file(20, 'stale.txt')])))
  expect(dialog.queryByText('stale.txt')).toBeNull()
  fireEvent.change(dialog.getByRole('textbox', { name: 'Search files' }), { target: { value: 'nothing' } })
  await dialog.findByText('No matching files.')
  // The search field has its own clear control; this is the one the empty result offers.
  fireEvent.click(dialog.getAllByRole('button', { name: 'Clear search' }).at(-1)!)
  await waitFor(() => expect(calls('groups.attachment.list').at(-1)).toEqual({ room_id: 'room-1', limit: 8, profile: 'reviewer' }))
  expect(calls('groups.attachment.list').map(params => params.query)).toEqual([undefined, 'Builder', 'nothing', undefined])
})

it('saves the exact listed version through the existing download', async () => {
  const selected = { ...file(19, 'same.txt'), size: 10 }
  gateway['groups.attachment.list'] = () => page([file(20, 'same.txt'), selected])
  gateway['groups.attachment.download'] = async () => downloadReply(selected, 'version 19')
  const dialog = await openFiles()
  await waitFor(() => expect(rows()).toHaveLength(2))
  fireEvent.click(within(rows()[1]).getByRole('button', { name: 'Download: same.txt' }))
  await waitFor(() => expect(observed.downloads).toHaveLength(1))
  await expectDownloaded(observed, new Uint8Array(Array.from('version 19', char => char.charCodeAt(0))), 'same.txt', 'text/plain')
  expect(calls('groups.attachment.download')).toEqual([
    { room_id: 'room-1', event_id: selected.event_id, attachment_id: selected.attachment_id, profile: 'reviewer' }])
  expect(dialog.queryByRole('alert')).toBeNull()
})

it.each([
  ['digest', { sha256: '0'.repeat(64) }],
  ['size', { data_base64: 'QUJD' }],
  ['version', { attachment_id: 'att_' + 'f'.repeat(32) }]
])('saves nothing when the downloaded %s does not match the listed version', async (_case, change) => {
  const selected = { ...file(19), size: 1 }
  gateway['groups.attachment.list'] = () => page([selected])
  gateway['groups.attachment.download'] = async () => ({ ...await downloadReply(selected), ...change })
  const dialog = await openFiles()
  fireEvent.click(await dialog.findByRole('button', { name: 'Download: file-19.txt' }))
  expect((await dialog.findByRole('alert')).textContent).toContain('Nothing was downloaded.')
  expect(observed.create).not.toHaveBeenCalled()
})

it('never saves a download that lands after the dialog closed', async () => {
  const selected = file(19)
  let release!: (value: unknown) => void
  gateway['groups.attachment.list'] = () => page([selected])
  gateway['groups.attachment.download'] = () => new Promise(resolve => { release = resolve })
  const dialog = await openFiles()
  fireEvent.click(await dialog.findByRole('button', { name: 'Download: file-19.txt' }))
  await waitFor(() => expect(calls('groups.attachment.download')).toHaveLength(1))
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await act(async () => release(await downloadReply(selected)))
  expect(observed.create).not.toHaveBeenCalled()
})

it('recovers from a refused cursor by showing the latest files, and retries an unavailable gateway', async () => {
  let attempts = 0

  gateway['groups.attachment.list'] = params => {
    if (params.cursor) {throw { code: 4001, data: { reason: 'attachment_cursor_invalid' } }}

    if (++attempts === 1) {return page([file(20)], 'after-20')}

    if (attempts === 2) {throw { code: 4001, data: { reason: 'runtime_coordination_required' } }}

    return page([file(21)], null, 21)
  }

  const dialog = await openFiles()
  await dialog.findByText('file-20.txt')
  fireEvent.click(dialog.getByRole('button', { name: 'Older files' }))
  expect((await dialog.findByRole('status')).textContent).toContain('This list is out of date.')
  fireEvent.click(within(dialog.getByRole('status')).getByRole('button', { name: 'Show latest' }))
  expect(await dialog.findByText('Files are temporarily unavailable.')).toBeTruthy()
  fireEvent.click(dialog.getByRole('button', { name: 'Retry' }))
  await dialog.findByText('file-21.txt')
  expect(calls('groups.attachment.list').filter(params => !params.cursor)).toHaveLength(3)
})

it('refuses a page whose order cannot be trusted', async () => {
  gateway['groups.attachment.list'] = () => page([file(19), file(20)])
  const dialog = await openFiles()
  expect(await dialog.findByText('Files could not be loaded.')).toBeTruthy()
  expect(rows()).toHaveLength(0)
})

it('tells the header which file the room log shared last', async () => {
  const seen: number[] = []
  request.mockImplementation(async (_route, method: string) => method === 'groups.state'
    ? { room: { name: 'Room', authority_epoch: 1 }, driver_status: {} }
    : { events: [
      { seq: 3, event_id: 'user:3', room_id: binding.roomId, kind: 'message.user',
        payload: { text: 'files', attachments: [{ attachment_id: file(3).attachment_id, kind: 'file', name: 'a.txt', mime: 'text/plain', size: 1 }] } },
      { seq: 4, event_id: 'user:4', room_id: binding.roomId, kind: 'message.user', payload: { text: 'just text' } }] })
  render(<CanonicalGroupWorkspace actions={room => { seen.push(room.latestFileSeq);

 return null }} binding={binding} />)
  await waitFor(() => expect(seen.at(-1)).toBe(3))
})
