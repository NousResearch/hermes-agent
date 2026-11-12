import { type PluginRestOptions, type PluginStorage, queryClient } from '@hermes/plugin-sdk'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { KanbanBoard, KanbanBoardDelta, KanbanTask } from './types'

// The board read asks the backend for a delta once it holds a board it can diff from. A backend that
// predates deltas ignores `since` and `etag` and answers a full snapshot every time; the app must keep
// rendering the whole, current board against it.

const { bindApi, boardKey, fetchBoard } = await import('./api')

const SLUG = 'ops'
const KEY = boardKey('local', SLUG, false)

const task = (id: string, status: string): KanbanTask => ({ id, status, title: `card ${id}` })

const snapshot = (columns: Record<string, string[]>, over: Partial<KanbanBoard> = {}): KanbanBoard => ({
  assignees: [],
  columns: Object.entries(columns).map(([name, ids]) => ({ name, tasks: ids.map(id => task(id, name)) })),
  latest_event_id: 10,
  now: 1_000,
  tenants: [],
  ...over
})

const storage: PluginStorage = {
  get: <T>(key: string, fallback: T) => (key === 'boardSlug' ? (SLUG as T) : fallback),
  remove: vi.fn(),
  set: vi.fn()
}

const shown = (b: KanbanBoard) => b.columns.map(column => [column.name, column.tasks.map(t => t.id)])

let paths: string[] = []
let answers: Array<() => unknown> = []
let dispose: () => void = () => undefined

// Binds the api to a scripted backend: each `/board` read takes the next answer, and the path it was
// asked on is recorded. The first read is bindApi's own snapshot (it opens the events socket at the
// board's tail); the board it leaves in the query cache is the one the next read diffs from, and is
// returned.
async function bindBackend(first: KanbanBoard, ...next: Array<() => unknown>): Promise<KanbanBoard> {
  answers = [() => first, ...next]

  const rest = async <T>(path: string, _opts?: PluginRestOptions): Promise<T> => {
    if (!path.startsWith('/board?board=ops')) {
      throw new Error(`unexpected REST path: ${path}`)
    }

    paths.push(path)

    const answer = answers.shift()

    return (answer ? answer() : first) as T
  }

  const socket = vi.fn(() => vi.fn())

  dispose = bindApi(rest, storage, socket)
  await vi.waitFor(() => expect(socket).toHaveBeenCalled())
  paths = []

  return queryClient.getQueryData<KanbanBoard>(KEY)!
}

// What react-query does with a queryFn result: hold it under the key the next read looks at.
async function read(): Promise<KanbanBoard> {
  const board = await fetchBoard(false, 'local', SLUG)
  queryClient.setQueryData(KEY, board)

  return board
}

beforeEach(() => {
  paths = []
})

afterEach(() => {
  dispose()
  queryClient.clear()
})

describe('board read against a backend that ignores since and etag', () => {
  it('asks for the full board every time and shows what it answers', async () => {
    await bindBackend(
      snapshot({ todo: ['a', 'b'] }),
      () => snapshot({ todo: ['a', 'b'] }),
      () => snapshot({ done: ['b'], todo: ['a', 'c'] }, { latest_event_id: 14, now: 1_060 })
    )

    await read()
    const latest = await read()

    // No etag ever came back, so no read had a base to diff from, and none asked for one.
    expect(paths).toEqual(['/board?board=ops', '/board?board=ops'])
    expect(shown(latest)).toEqual([
      ['done', ['b']],
      ['todo', ['a', 'c']]
    ])
    expect(latest.latest_event_id).toBe(14)
  })

  it('replaces the held board when a backend that hands out an etag still answers a delta request in full', async () => {
    await bindBackend(snapshot({ todo: ['a'] }, { etag: 'e1' }), () =>
      snapshot({ todo: ['a', 'b'] }, { etag: 'e2', latest_event_id: 12 })
    )

    const latest = await read()

    expect(paths).toEqual(['/board?board=ops&since=10&etag=e1'])
    expect(shown(latest)).toEqual([['todo', ['a', 'b']]])
    expect(latest.etag).toBe('e2')
  })
})

describe('board read against a backend that answers deltas', () => {
  it('merges an unchanged delta into the held board without rebuilding it', async () => {
    const unchanged: KanbanBoardDelta = { delta: true, etag: 'e1', latest_event_id: 10, now: 1_030, unchanged: true }
    const held = await bindBackend(snapshot({ todo: ['a', 'b'] }, { etag: 'e1' }), () => unchanged)

    const next = await read()

    expect(paths).toEqual(['/board?board=ops&since=10&etag=e1'])
    expect(next.columns[0]).toBe(held.columns[0])
    expect(next.now).toBe(1_030)
  })

  it('asks again in full when the delta names a card the client does not hold', async () => {
    const stray: KanbanBoardDelta = {
      column_order: [{ ids: ['a', 'ghost'], name: 'todo' }],
      delta: true,
      etag: 'e2',
      latest_event_id: 12,
      now: 1_030
    }

    await bindBackend(
      snapshot({ todo: ['a'] }, { etag: 'e1' }),
      () => stray,
      () => snapshot({ todo: ['a', 'ghost'] }, { etag: 'e2', latest_event_id: 12 })
    )

    const latest = await read()

    expect(paths).toEqual(['/board?board=ops&since=10&etag=e1', '/board?board=ops'])
    expect(shown(latest)).toEqual([['todo', ['a', 'ghost']]])
  })

  it('keeps the held board when the gateway answers with no body', async () => {
    const held = await bindBackend(snapshot({ todo: ['a'] }, { etag: 'e1' }), () => undefined)

    expect(await read()).toBe(held)
  })
})
