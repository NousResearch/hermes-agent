import { webcrypto } from 'node:crypto'

import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { expectDownloaded, observeDownloads } from './canonical-download-test-utils'
import {
  canonicalFilesFailure,
  captureFilesSource,
  listCanonicalFiles,
  readCanonicalFile,
  saveCanonicalFile
} from './canonical-files-client'
import { CANONICAL_FILES_LOCALES } from './canonical-files-labels'
import { deferred, FILE_BINDING, fileItem, filePage } from './canonical-files-test-fixtures'
import { parseGroupFilesPage } from './group-files-parser'

const { request, source } = vi.hoisted(() => ({ request: vi.fn(), source: { connectionId: 'gateway-a', profile: 'reviewer', gateway: 'open', epoch: 1 } }))
vi.mock('@hermes/plugin-sdk', () => ({
  gatewayActivationEpoch: () => source.epoch,
  host: { requestProfile: request, state: {
    connectionId: { get: () => source.connectionId }, profile: { get: () => source.profile }, gateway: { get: () => source.gateway }
  } }
}))
const originalDesktop = window.hermesDesktop
const authority = { gatewayId: 'install:home', epoch: 1 }
const raw = { ...fileItem(20, 'same-name.bin'), size: 4 }
const item = parseGroupFilesPage(filePage([raw])).items[0]
const bytes = new Uint8Array([0, 1, 2, 255])
let observed: ReturnType<typeof observeDownloads>

async function receipt() {
  const hash = await webcrypto.subtle.digest('SHA-256', bytes)

  return {
    ...raw,
    room_id: FILE_BINDING.roomId,
    authority: { gateway_id: authority.gatewayId, epoch: authority.epoch },
    data_base64: 'AAEC/w==',
    sha256: Buffer.from(hash).toString('hex')
  }
}

beforeEach(() => {
  Object.assign(source, { connectionId: 'gateway-a', profile: 'reviewer', gateway: 'open', epoch: 1 })
  request.mockReset()
  vi.stubGlobal('crypto', webcrypto)
  observed = observeDownloads()
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  window.hermesDesktop = originalDesktop
})

it('lists only through the captured canonical profile without legacy viewer parameters', async () => {
  request.mockResolvedValue(filePage())
  await listCanonicalFiles(FILE_BINDING, { limit: 100, cursor: 'opaque', query: 'report' }, authority)
  expect(request).toHaveBeenCalledWith(
    { connectionId: 'gateway-a', profile: 'reviewer', targetProfile: 'reviewer', mode: 'remote' },
    'groups.attachment.list',
    {
      profile: 'reviewer',
      room_id: 'room-1',
      limit: 32,
      cursor: 'opaque',
      query: 'report',
      authority_gateway_id: 'install:home',
      authority_epoch: 1
    }
  )
})

it.each([{ limit: 0 }, { limit: 1.5 }, { query: 'x'.repeat(256) }, { cursor: 'x'.repeat(4097) }])(
  'rejects invalid list input before routing: %j',
  async input => {
    await expect(listCanonicalFiles(FILE_BINDING, input)).rejects.toThrow()
    expect(request).not.toHaveBeenCalled()
  }
)

it('refuses foreign room and authority replies and malformed pages', async () => {
  request.mockResolvedValueOnce({ ...filePage(), room_id: 'foreign' })
  await expect(listCanonicalFiles(FILE_BINDING)).rejects.toMatchObject({ kind: 'scope' })
  request.mockResolvedValueOnce({ ...filePage(), authority: { gateway_id: 'other', epoch: 2 } })
  await expect(listCanonicalFiles(FILE_BINDING, {}, authority)).rejects.toMatchObject({ kind: 'scope' })
  request.mockResolvedValueOnce({ ...filePage(), snapshot_seq: '20' })
  await expect(listCanonicalFiles(FILE_BINDING)).rejects.toMatchObject({ kind: 'verification' })
})

it('downloads the selected event/version with both authority pins and saves verified bytes', async () => {
  const save = vi.fn().mockResolvedValue(undefined)
  window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop
  request.mockResolvedValue(await receipt())
  await saveCanonicalFile(FILE_BINDING, authority, item)
  expect(request).toHaveBeenCalledWith(
    expect.objectContaining({ connectionId: 'gateway-a', targetProfile: 'reviewer' }),
    'groups.attachment.download',
    {
      profile: 'reviewer',
      room_id: 'room-1',
      event_id: raw.event_id,
      attachment_id: raw.attachment_id,
      authority_gateway_id: 'install:home',
      authority_epoch: 1
    }
  )
  expect(observed.downloads).toHaveLength(1)
  await expectDownloaded(observed, bytes, raw.name, raw.mime)
  expect(save).not.toHaveBeenCalled()
})

it('uses the user-facing download workflow even without the composer-cache bridge', async () => {
  window.hermesDesktop = {} as typeof window.hermesDesktop
  request.mockResolvedValue(await receipt())
  await saveCanonicalFile(FILE_BINDING, authority, item)
  expect(observed.downloads).toHaveLength(1)
  await expectDownloaded(observed, bytes, raw.name, raw.mime)
})

it.each(['room_id', 'event_id', 'attachment_id', 'name', 'mime', 'kind', 'size', 'sha256', 'data_base64', 'authority'])(
  'never saves a mismatched or corrupt %s',
  async field => {
    const save = vi.fn()
    window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop
    request.mockResolvedValue({
      ...(await receipt()),
      [field]: field === 'size' ? 5 : field === 'data_base64' ? 'AQEC/w==' : 'foreign'
    })
    await expect(saveCanonicalFile(FILE_BINDING, authority, item)).rejects.toThrow()
    expect(save).not.toHaveBeenCalled()
    expect(observed.create).not.toHaveBeenCalled()
  }
)

it('retires late download replies on intent cancellation and timeout', async () => {
  const save = vi.fn()
  window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop
  const response = await receipt()
  const held = deferred<unknown>()
  request.mockReturnValue(held.promise)
  const controller = new AbortController()
  const saving = saveCanonicalFile(FILE_BINDING, authority, item, controller.signal)
  controller.abort()
  await expect(saving).rejects.toMatchObject({ name: 'AbortError' })
  held.resolve(response)
  await Promise.resolve()
  expect(save).not.toHaveBeenCalled()
  expect(observed.create).not.toHaveBeenCalled()

  vi.useFakeTimers()
  const late = deferred<unknown>()
  request.mockReturnValue(late.promise)
  const timed = readCanonicalFile(FILE_BINDING, authority, item)
  const check = expect(timed).rejects.toMatchObject({ kind: 'timeout' })
  await vi.advanceTimersByTimeAsync(10_000)
  await check
  late.resolve(response)
  expect(save).not.toHaveBeenCalled()
  expect(observed.create).not.toHaveBeenCalled()
})

it.each(['permission_denied', 'profile_mismatch'])(
  'recognizes canonical access denial %s without relying on display prose',
  reason => {
    expect(canonicalFilesFailure({ code: 4001, message: 'RPC failed', data: { reason } })).toBe('access')
  }
)

it.each(['connectionId', 'profile', 'gateway', 'epoch'] as const)('refuses a stale %s after the download await', async field => {
  const response = await receipt()
  const held = deferred<unknown>()
  request.mockReturnValue(held.promise)
  const saving = saveCanonicalFile(FILE_BINDING, authority, item)

  if (field === 'epoch') { source.epoch++ } else { source[field] = 'changed' }
  held.resolve(response)
  await expect(saving).rejects.toMatchObject({ kind: 'scope' })
  expect(observed.create).not.toHaveBeenCalled()
})

it('rechecks activation after digest and freezes the selected metadata through Save', async () => {
  const response = await receipt()
  const held = deferred<ArrayBuffer>()
  const digest = await webcrypto.subtle.digest('SHA-256', bytes)
  vi.spyOn(crypto.subtle, 'digest').mockReturnValueOnce(held.promise)
  request.mockResolvedValue(response)
  const saving = saveCanonicalFile(FILE_BINDING, authority, item)
  await vi.waitFor(() => expect(crypto.subtle.digest).toHaveBeenCalled())
  source.epoch++
  held.resolve(digest)
  await expect(saving).rejects.toMatchObject({ kind: 'scope' })
  expect(observed.create).not.toHaveBeenCalled()
})

it('never retargets a mutable same-name selection while waiting for bytes', async () => {
  const response = await receipt()
  const held = deferred<unknown>()
  request.mockReturnValue(held.promise)
  const selection = structuredClone(item)
  const saving = saveCanonicalFile({ ...FILE_BINDING }, { ...authority }, selection)
  selection.attachment.name = 'another-file.bin'
  selection.eventId = 'another-event'
  held.resolve(response)
  await saving
  await expectDownloaded(observed, bytes, raw.name, raw.mime)
})

it.each(['connectionId', 'profile', 'gateway', 'epoch'] as const)(
  'checks captured %s before list routing and after its decoded RPC body', async field => {
    const current = captureFilesSource()
    const held = deferred<unknown>()
    request.mockReturnValueOnce(held.promise)
    const listing = listCanonicalFiles(FILE_BINDING, {}, authority, undefined, current)

    if (field === 'epoch') {source.epoch++} else {source[field] = 'changed'}
    held.resolve(filePage())
    await expect(listing).rejects.toMatchObject({ kind: 'scope' })
    await expect(listCanonicalFiles(FILE_BINDING, {}, authority, undefined, current)).rejects.toMatchObject({ kind: 'scope' })
    expect(request).toHaveBeenCalledTimes(1)
  }
)

it('freezes authority, effective backend profile and nested selection across digest and the actual Save callback', async () => {
  const response = await receipt()
  const held = deferred<ArrayBuffer>()
  const hash = await webcrypto.subtle.digest('SHA-256', bytes)
  vi.spyOn(crypto.subtle, 'digest').mockReturnValueOnce(held.promise)
  request.mockResolvedValue(response)
  const selected = structuredClone(item)
  const binding = { ...FILE_BINDING }
  const selectedAuthority = { ...authority }
  const saving = saveCanonicalFile(binding, selectedAuthority, selected)
  await vi.waitFor(() => expect(crypto.subtle.digest).toHaveBeenCalled())
  binding.profile = 'other-backend'
  binding.connectionId = 'other-connection'
  binding.roomId = 'other-room'
  selectedAuthority.gatewayId = 'other-authority'
  selectedAuthority.epoch = 2
  Object.assign(selected.attachment, { attachmentId: 'att_' + 'f'.repeat(32), name: 'wrong.bin', mime: 'image/png', size: 9 })
  selected.eventId = 'other-event'
  held.resolve(hash)
  await saving
  expect(request.mock.calls[0][0]).toMatchObject({ profile: FILE_BINDING.profile, targetProfile: FILE_BINDING.profile })
  expect(request.mock.calls[0][2]).toMatchObject({ room_id: FILE_BINDING.roomId, event_id: item.eventId,
    attachment_id: item.attachment.attachmentId, authority_gateway_id: authority.gatewayId, authority_epoch: authority.epoch })
  expect(observed.click).toHaveBeenCalledTimes(1)
  await expectDownloaded(observed, bytes, raw.name, raw.mime)
})

it('checks ownership once more between verified bytes and the real Save helper', async () => {
  request.mockResolvedValue(await receipt())
  const current = captureFilesSource()
  // Queue a source swap after the digest continuation's check, before the
  // saveCanonicalFile continuation. No download sink or Files client is mocked.
  let checks = 0

  const sourceCurrent = () => {
    const valid = current()
    checks++

    if (checks === 3) {queueMicrotask(() => {source.epoch++})}

    return valid
  }

  await expect(saveCanonicalFile(FILE_BINDING, authority, item, undefined, sourceCurrent)).rejects.toMatchObject({ kind: 'scope' })
  expect(observed.create).not.toHaveBeenCalled()
  expect(observed.click).not.toHaveBeenCalled()
})

it('retains the complete donor Files copy in every supported locale', () => {
  const keys = Object.keys(CANONICAL_FILES_LOCALES.en).sort()

  for (const copy of Object.values(CANONICAL_FILES_LOCALES)) {
    expect(Object.keys(copy).sort()).toEqual(keys)
    expect(Object.values(copy).every(text => text.trim().length > 0)).toBe(true)
  }
})
