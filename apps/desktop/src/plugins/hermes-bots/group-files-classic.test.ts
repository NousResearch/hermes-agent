import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $groupChats } from './group-chat'
import {
  classicLatestFileKey,
  createClassicGroupFilesLoader,
  foldGroupFileSearch,
  latestHostedFileSeq
} from './group-files-classic'
import { validateGroupFilesContinuation } from './group-files-client'
import type { GroupChat, GroupMessage } from './types'

const host = vi.hoisted(() => ({ requestProfile: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const entry = (index: number, name = `file-${index}.txt`, producer = 'Writer'): GroupMessage => ({
  id: `local-message-${index}`,
  at: 1_700_000_000_000 + index,
  from: { kind: 'member', name: producer },
  text: '',
  images: [{ kind: 'file', name, data: 'data:text/plain;base64,YQ==' }]
})

const room = (log: GroupMessage[]): GroupChat => ({
  continuityMode: 'desktop',
  log,
  roomId: 'classic-room',
  watermarks: {}
})

beforeEach(() => {
  vi.clearAllMocks()
  $groupChats.set({})
})

describe('bounded retained classic files', () => {
  it('pages retained data URLs offline without canonical IDs or byte copies', async () => {
    $groupChats.set({ Core: room(Array.from({ length: 9 }, (_, index) => entry(index, 'report.txt'))) })
    const load = createClassicGroupFilesLoader('Core', 'classic-room')
    const first = await load('Core')
    expect(first.authority).toBeNull()
    expect(first.items).toHaveLength(8)
    expect(first.items[0].localMessage?.id).toBe('local-message-8')
    expect(first.items.every(item => item.attachment.data === 'data:text/plain;base64,YQ==')).toBe(true)
    expect(first.items.every(item => !item.attachment.attachmentId)).toBe(true)
    const second = await load('Core', { cursor: first.nextCursor! })
    expect(second.items).toHaveLength(1)
    expect(second.items[0].localMessage?.id).toBe('local-message-0')
    expect(() => validateGroupFilesContinuation(first, second)).not.toThrow()
    expect(host.requestProfile).not.toHaveBeenCalled()
  })

  it('holds the snapshot across arrivals and invalidates old cursors only on explicit refresh', async () => {
    const original = room(Array.from({ length: 9 }, (_, index) => entry(index)))
    $groupChats.set({ Core: original })
    const load = createClassicGroupFilesLoader('Core', 'classic-room')
    const first = await load('Core')
    $groupChats.set({ Core: room([...original.log, entry(10)]) })
    const older = await load('Core', { cursor: first.nextCursor! })
    expect(older.items[0].attachment.name).toBe('file-0.txt')
    const latest = await load('Core')
    expect(latest.items[0].attachment.name).toBe('file-10.txt')
    await expect(load('Core', { cursor: first.nextCursor! })).rejects.toThrow('cursor')
  })

  it('binds cursors to one loader instance and rejects a room replacement', async () => {
    $groupChats.set({ Core: room(Array.from({ length: 9 }, (_, index) => entry(index))) })
    const first = await createClassicGroupFilesLoader('Core', 'classic-room')('Core')
    const other = createClassicGroupFilesLoader('Core', 'classic-room')
    await other('Core')
    await expect(other('Core', { cursor: first.nextCursor! })).rejects.toThrow('cursor')
    $groupChats.set({ Core: { ...room([]), roomId: 'replacement' } })
    await expect(other('Core')).rejects.toThrow()
  })

  it('folds accents and case in both filename and sharer', async () => {
    $groupChats.set({ Core: room([entry(1, 'Re\u0301sume\u0301.pdf', 'José'), entry(2, 'notes.md', 'STRASSE')]) })
    const load = createClassicGroupFilesLoader('Core', 'classic-room')
    expect((await load('Core', { query: 'RESUME' })).items).toHaveLength(1)
    expect((await load('Core', { query: 'jose' })).items).toHaveLength(1)
    expect((await load('Core', { query: 'straße' })).items).toHaveLength(1)
    expect(foldGroupFileSearch('ẞ')).toBe(foldGroupFileSearch('ß'))
    expect(foldGroupFileSearch('ᾈ')).toBe(foldGroupFileSearch('ἀι'))
  })

  it('scans at most the existing 96-entry retained window and eight attachments per entry', async () => {
    const log = Array.from({ length: 100 }, (_, index) => entry(index))
    log[99].images = Array.from({ length: 20 }, (_, index) => ({ ...entry(index).images![0] }))
    $groupChats.set({ Core: room(log) })
    const load = createClassicGroupFilesLoader('Core', 'classic-room')
    let page = await load('Core')
    let count = page.items.length

    while (page.nextCursor) {
      page = await load('Core', { cursor: page.nextCursor })
      count += page.items.length
    }

    expect(count).toBe(95 + 8)
  })

  it('does not invent local bytes from metadata or ordinary text', async () => {
    $groupChats.set({
      Core: room([{ ...entry(1), images: [{ kind: 'file', name: 'remote.txt', attachmentId: 'att_missing' }] }])
    })
    expect((await createClassicGroupFilesLoader('Core', 'classic-room')('Core')).items).toHaveLength(0)
  })

  it('signals only file arrivals, never an ordinary room sequence advance', () => {
    const first = room([entry(1)])
    const text = { ...entry(2), images: [], seq: 100 }
    expect(classicLatestFileKey(room([...first.log, text]))).toBe(classicLatestFileKey(first))
    expect(latestHostedFileSeq(room([{ ...entry(1), seq: 5 }, text]))).toBe(5)
    expect(latestHostedFileSeq(room([{ ...entry(1), seq: 5 }, text, { ...entry(3), seq: 101 }]))).toBe(101)
  })
})
