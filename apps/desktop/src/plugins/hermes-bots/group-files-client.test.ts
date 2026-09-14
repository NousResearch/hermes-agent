import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $groupChats } from './group-chat'
import { listHostedGroupFiles, parseGroupFilesPage, validateGroupFilesContinuation } from './group-files-client'
import { deferred, FILE_ROOM, fileItem, filePage, parsedFilePage } from './group-files-test-fixtures'

const mocks = vi.hoisted(() => ({ route: vi.fn(), request: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock({})
})

vi.mock('./hosted-room-runtime', () => ({
  hostedRouteForRoom: mocks.route,
  requestHostedConnection: mocks.request
}))

const ROUTE = { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' }

beforeEach(() => {
  vi.clearAllMocks()
  $groupChats.set({ Core: FILE_ROOM })
  mocks.route.mockResolvedValue(ROUTE)
  mocks.request.mockResolvedValue(filePage())
})

afterEach(() => vi.useRealTimers())

describe('strict shared-files response contract', () => {
  it('accepts empty and scan-bounded empty continuation pages', () => {
    expect(parseGroupFilesPage(filePage([]))).toMatchObject({ items: [], hasMore: false, nextCursor: null })
    expect(parseGroupFilesPage(filePage([], true))).toMatchObject({ items: [], hasMore: true })
  })

  it('preserves same-name versions, Unicode names, zero-byte files and authoritative order', () => {
    const unicode = '研究報告-é-'.repeat(20) + '.txt'
    const first = { ...fileItem(20, unicode, 1), size: 0 }
    const second = fileItem(20, unicode, 2)
    const page = parseGroupFilesPage(filePage([first, second, fileItem(19, unicode, 3)]))

    expect(page.items).toHaveLength(3)
    expect(page.items.map(item => item.attachment.name)).toEqual([unicode, unicode, unicode])
    expect(page.items.map(item => item.attachment.attachmentId)).toEqual(
      [first, second, fileItem(19, unicode, 3)].map(item => item.attachment_id)
    )
    expect(page.items[0].producer).toEqual({ identity: 'builder', kind: 'member', label: 'Builder' })
    expect(page.items[0].attachment.size).toBe(0)
  })

  it.each([
    ['missing page', null],
    ['array page', []],
    ['error envelope', { error: 'denied' }],
    ['missing items', { ...filePage(), items: undefined }],
    ['string snapshot', { ...filePage(), snapshot_seq: '20' }],
    ['boolean snapshot', { ...filePage(), snapshot_seq: true }],
    ['negative snapshot', { ...filePage(), snapshot_seq: -1 }],
    ['fractional epoch', { ...filePage(), authority: { gateway_id: 'install:home', epoch: 1.5 } }],
    ['string has_more', { ...filePage(), has_more: 'false' }],
    ['missing cursor', { ...filePage(), next_cursor: undefined }],
    ['missing continuation', { ...filePage(), has_more: true }],
    ['unexpected continuation', { ...filePage(), next_cursor: 'cursor' }],
    ['oversized cursor', { ...filePage([], true), next_cursor: 'x'.repeat(4097) }],
    ['too many items', filePage(Array.from({ length: 9 }, (_, index) => fileItem(20 - index)))],
    ['duplicate identity', filePage([fileItem(20, 'a', 1), fileItem(19, 'b', 1)])],
    ['ascending sequence', filePage([fileItem(19), fileItem(20)])],
    ['reversed tie', filePage([fileItem(20, 'a', 2), fileItem(20, 'b', 1)])]
  ])('rejects %s', (_label, value) => {
    expect(() => parseGroupFilesPage(value)).toThrow()
  })

  it.each([
    { attachment_id: 'att_bad' },
    { event_id: '' },
    { kind: 'video' },
    { name: '' },
    { name: 'x'.repeat(256) },
    { mime: 'text/plain; charset=utf8' },
    { size: '1' },
    { size: null },
    { size: -1 },
    { size: 15_000_001 },
    { seq: 21 },
    { seq: '20' },
    { seq: 0 },
    { shared_at: '1700000000' },
    { shared_at: Number.NaN },
    { shared_at: 0 },
    { producer: { identity: 'builder', label: 'Builder' } },
    { producer: { kind: 'gateway', id: 'builder', label: 'Builder' } },
    { producer: { kind: 'member', id: '', label: 'Builder' } },
    { producer: { kind: 'member', id: 'builder', label: '' } }
  ])('rejects malformed item fields %j', override => {
    expect(() =>
      parseGroupFilesPage(filePage([{ ...fileItem(), ...override } as ReturnType<typeof fileItem>]))
    ).toThrow()
  })

  it('rejects a foreign authority or epoch', () => {
    expect(() => parseGroupFilesPage(filePage(), { authorityId: 'install:other' })).toThrow()
    expect(() => parseGroupFilesPage(filePage(), { authorityEpoch: 2 })).toThrow()
  })

  it('strips unrelated transport fields instead of constructing download URLs from metadata', () => {
    const item = { ...fileItem(), path: '/private/file', data: 'data:text/html;base64,YQ==' }
    const parsed = parseGroupFilesPage(filePage([item]))

    expect(parsed.items[0].attachment).not.toHaveProperty('data')
    expect(parsed.items[0].attachment).not.toHaveProperty('path')
  })

  it('accepts only a progressing continuation from the same snapshot and authority', () => {
    const first = parsedFilePage([fileItem(20)], true)
    const next = parsedFilePage([fileItem(19)])

    expect(() => validateGroupFilesContinuation(first, next)).not.toThrow()
    expect(() => validateGroupFilesContinuation(first, { ...next, snapshotSeq: 21 })).toThrow()
    expect(() =>
      validateGroupFilesContinuation(first, { ...next, authority: { epoch: 2, gatewayId: 'install:home' } })
    ).toThrow()
    expect(() => validateGroupFilesContinuation(first, { ...next, nextCursor: first.nextCursor })).toThrow()
    expect(() => validateGroupFilesContinuation(first, parsedFilePage([fileItem(20)]))).toThrow()
  })
})

describe('shared-files request routing', () => {
  it('uses the verified room route and exact viewer request, never the active gateway', async () => {
    await listHostedGroupFiles('Core')
    expect(mocks.route).toHaveBeenCalledWith(FILE_ROOM, 'read', undefined)
    expect(mocks.request).toHaveBeenCalledWith(ROUTE, 'groups.attachment.list', {
      room_id: 'room-1',
      purpose: 'viewer',
      limit: 8
    })
  })

  it('keeps cursor/query exact and clamps the requested maximum to 32', async () => {
    await listHostedGroupFiles('Core', { cursor: 'opaque-cursor', limit: 100, query: '研究' })
    expect(mocks.request).toHaveBeenCalledWith(ROUTE, 'groups.attachment.list', {
      room_id: 'room-1',
      purpose: 'viewer',
      limit: 32,
      cursor: 'opaque-cursor',
      query: '研究'
    })
  })

  it.each([0, -1, 1.5, Number.NaN])('rejects invalid page size %s before routing', async limit => {
    await expect(listHostedGroupFiles('Core', { limit })).rejects.toThrow()
    expect(mocks.route).not.toHaveBeenCalled()
  })

  it('bounds search and cursor input before routing', async () => {
    await expect(listHostedGroupFiles('Core', { query: 'x'.repeat(256) })).rejects.toThrow()
    await expect(listHostedGroupFiles('Core', { cursor: 'x'.repeat(4097) })).rejects.toThrow()
    expect(mocks.route).not.toHaveBeenCalled()
  })

  it('refuses a room replacement while its route is being resolved', async () => {
    const route = deferred<typeof ROUTE>()
    mocks.route.mockReturnValue(route.promise)
    const request = listHostedGroupFiles('Core')
    $groupChats.set({ Core: { ...FILE_ROOM, roomId: 'room-2' } })
    route.resolve(ROUTE)
    await expect(request).rejects.toThrow('unavailable')
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it('bounds a stalled list request and ignores its late completion', async () => {
    vi.useFakeTimers()
    const held = deferred<ReturnType<typeof filePage>>()
    mocks.request.mockReturnValue(held.promise)
    const pending = listHostedGroupFiles('Core')
    const rejection = expect(pending).rejects.toThrow('timed out')
    await vi.advanceTimersByTimeAsync(10_000)
    await rejection
    held.resolve(filePage())
    await Promise.resolve()
  })
})
