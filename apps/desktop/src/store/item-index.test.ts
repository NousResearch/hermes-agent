import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type { SessionInfo } from '@/types/hermes'

import {
  $itemIndex,
  _resetItemIndexForTests,
  activeItemIndexOwner,
  ownerIndex,
  ownerItems,
  persistOwner,
  recordChatMessages,
  refreshItemIndex
} from './item-index'
import { $connection } from './session'
import { $removedSessionIds } from './session-removal'

const listAllProfileSessions = vi.fn()
const getSessionMessages = vi.fn()

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: (...args: unknown[]) => listAllProfileSessions(...args),
  getSessionMessages: (...args: unknown[]) => getSessionMessages(...args)
}))

const machine = (connectionId: string) =>
  ({ baseUrl: 'http://127.0.0.1:1', connectionId, isFullscreen: false, mode: 'remote' }) as HermesConnection

const row = (id: string, lastActive = 100, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    id,
    title: `Chat ${id}`,
    profile: 'default',
    last_active: lastActive,
    message_count: 2,
    started_at: 1,
    ...overrides
  }) as SessionInfo

/** Every chat's transcript names one file after its id. */
function serveTranscripts() {
  getSessionMessages.mockImplementation(async (id: string) => ({
    session_id: id,
    messages: [{ role: 'assistant', timestamp: 500, content: `MEDIA:/work/${id}.md` }]
  }))
}

function serveList(rows: SessionInfo[]) {
  listAllProfileSessions.mockResolvedValue({ sessions: rows, limit: 30, offset: 0, total: rows.length })
}

const values = () => ownerItems(ownerIndex(activeItemIndexOwner()).snapshot).map(item => item.value)

/** A fresh window: memory gone, storage kept. */
function reopen() {
  Object.keys($itemIndex.get()).forEach(persistOwner)
  _resetItemIndexForTests()
}

beforeEach(() => {
  window.localStorage.clear()
  _resetItemIndexForTests()
  $connection.set(null)
  serveTranscripts()
})

afterEach(() => {
  vi.clearAllMocks()
  $removedSessionIds.set(new Set())
})

describe('item index', () => {
  it('paints the stored list on a cold open before any request', async () => {
    serveList([row('a'), row('b')])
    await refreshItemIndex()
    reopen()
    listAllProfileSessions.mockClear()
    getSessionMessages.mockClear()

    expect(values().sort()).toEqual(['/work/a.md', '/work/b.md'])
    expect(listAllProfileSessions).not.toHaveBeenCalled()
    expect(getSessionMessages).not.toHaveBeenCalled()
  })

  it('reads only the chats whose activity changed since the last check', async () => {
    const rows = Array.from({ length: 10 }, (_, index) => row(`c${index}`))

    serveList(rows)
    await refreshItemIndex()
    expect(getSessionMessages).toHaveBeenCalledTimes(10)

    getSessionMessages.mockClear()
    serveList(rows.map(entry => (entry.id === 'c4' ? { ...entry, last_active: 200, message_count: 4 } : entry)))
    await refreshItemIndex()

    expect(getSessionMessages.mock.calls.map(call => call[0])).toEqual(['c4'])
  })

  it('records a streamed item without reading the transcript again', async () => {
    serveList([row('live')])
    await refreshItemIndex()
    getSessionMessages.mockClear()

    recordChatMessages({ connectionId: 'local', profile: 'default', sessionId: 'live' }, [
      { role: 'assistant', content: 'Saved MEDIA:/work/report.pdf', timestamp: 900 }
    ])

    expect(values()).toContain('/work/report.pdf')
    expect(getSessionMessages).not.toHaveBeenCalled()
  })

  it('drops a deleted chat, and another window cannot write it back', async () => {
    serveList([row('keep'), row('gone')])
    await refreshItemIndex()
    persistOwner('local')

    const staleOtherWindow = window.localStorage.getItem('hermes:item-index:v1:local')

    $removedSessionIds.set(new Set(['gone']))
    expect(values()).toEqual(['/work/keep.md'])

    persistOwner('local')
    window.localStorage.setItem('hermes:item-index:v1:local', staleOtherWindow!)
    persistOwner('local')
    _resetItemIndexForTests()

    expect(values()).toEqual(['/work/keep.md'])
  })

  it('keeps two machines apart', async () => {
    $connection.set(machine('homelab'))
    serveList([row('remote-chat')])
    await refreshItemIndex()

    $connection.set(machine('laptop'))
    serveList([row('laptop-chat')])
    await refreshItemIndex()
    reopen()

    expect(values()).toEqual(['/work/laptop-chat.md'])
    $connection.set(machine('homelab'))
    expect(values()).toEqual(['/work/remote-chat.md'])
  })

  it('stops reading when the machine changes mid-refresh, writing nothing into the new one', async () => {
    serveList([row('first'), row('second')])
    getSessionMessages.mockImplementationOnce(async (id: string) => {
      $connection.set(machine('elsewhere'))

      return { session_id: id, messages: [{ role: 'assistant', content: 'MEDIA:/work/first.md', timestamp: 1 }] }
    })

    await refreshItemIndex()

    expect(getSessionMessages).toHaveBeenCalledTimes(1)
    expect(values()).toEqual([])
  })

  it('ignores a stored snapshot of another version, a corrupt one, and a malformed chat', () => {
    const key = 'hermes:item-index:v1:local'

    window.localStorage.setItem(key, '{not json')
    expect(values()).toEqual([])

    _resetItemIndexForTests()
    window.localStorage.setItem(key, JSON.stringify({ v: 99, chats: [], checkedAt: 1, removed: {} }))
    expect(values()).toEqual([])

    _resetItemIndexForTests()
    window.localStorage.setItem(
      key,
      JSON.stringify({
        v: 1,
        checkedAt: 1,
        removed: {},
        chats: [
          {
            s: 'ok',
            p: 'default',
            c: 'local',
            ti: 'Ok',
            a: 1,
            u: 1,
            m: '1:1',
            i: [{ k: 'file', v: '/a.md', h: 'file:///a.md', l: 'a.md', t: 1 }]
          },
          { s: 'bad', p: 'default', c: 'local', ti: 'Bad', a: 'yesterday', u: 1, m: null, i: [] },
          { s: 'mixed', p: 'default', c: 'local', ti: 'Mixed', a: 1, u: 1, m: null, i: [{ k: 'video', v: '/x' }, null] }
        ]
      })
    )

    expect(values()).toEqual(['/a.md'])
    expect(Object.keys($itemIndex.get().local!.snapshot.chats)).toHaveLength(2)
  })

  it('keeps a chat whose profile could not be listed this time', async () => {
    serveList([row('a', 100, { profile: 'work' })])
    await refreshItemIndex()

    listAllProfileSessions.mockResolvedValue({ sessions: [], errors: [{ profile: 'work', error: 'locked' }] })
    await refreshItemIndex()

    expect(values()).toEqual(['/work/a.md'])
  })
})
