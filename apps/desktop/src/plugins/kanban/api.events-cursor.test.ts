import { type PluginRestOptions, type PluginStorage, queryClient } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Opening /events with no since replayed the board's whole task_events history.
// The socket must start at the snapshot's tail, and a late snapshot must not
// reopen a board the user already left. The cursor is per connection.

const { $boardSlug, bindApi, boardKey } = await import('./api')
const { setConnection } = await import('@/store/session')

const BOARD = (latest_event_id: number) => ({
  assignees: [],
  columns: [],
  latest_event_id,
  now: 0,
  tenants: []
})

const storage = (slug = ''): PluginStorage => ({
  get: <T>(key: string, fallback: T) => (key === 'boardSlug' ? (slug as T) : fallback),
  remove: vi.fn(),
  set: vi.fn()
})

afterEach(() => {
  setConnection(null)
  $boardSlug.set('')
  queryClient.clear()
})

describe('kanban event cursor', () => {
  it('does not open the socket until the board snapshot resolves, then starts at its tail', async () => {
    let resolveBoard: (board: ReturnType<typeof BOARD>) => void = () => undefined

    const board = new Promise<ReturnType<typeof BOARD>>(resolve => {
      resolveBoard = resolve
    })

    const rest = async <T>(path: string, _opts?: PluginRestOptions): Promise<T> => {
      if (path === '/board?board=ops') {
        return board as Promise<T>
      }

      throw new Error(`unexpected REST path: ${path}`)
    }

    const socket = vi.fn(() => vi.fn())
    const dispose = bindApi(rest, storage('ops'), socket)

    expect(socket).not.toHaveBeenCalled()
    resolveBoard(BOARD(14_386))

    await vi.waitFor(() => {
      expect(socket).toHaveBeenCalledWith('/events?board=ops&since=14386', expect.any(Function))
    })

    dispose()
  })

  it('seeds since from the cached snapshot instead of replaying history', () => {
    queryClient.setQueryData(boardKey('local', 'ops', false), BOARD(14_386))

    const socket = vi.fn(() => vi.fn())

    const dispose = bindApi(
      async () => {
        throw new Error('cached snapshot must not refetch')
      },
      storage('ops'),
      socket
    )

    expect(socket).toHaveBeenCalledWith('/events?board=ops&since=14386', expect.any(Function))
    dispose()
  })

  it('drops a snapshot that resolves after the selected board changed', async () => {
    let resolveOps: (board: ReturnType<typeof BOARD>) => void = () => undefined

    const ops = new Promise<ReturnType<typeof BOARD>>(resolve => {
      resolveOps = resolve
    })

    const rest = async <T>(path: string, _opts?: PluginRestOptions): Promise<T> => {
      if (path === '/board?board=ops') {
        return ops as Promise<T>
      }

      if (path === '/board?board=ship') {
        return BOARD(7) as T
      }

      throw new Error(`unexpected REST path: ${path}`)
    }

    const socket = vi.fn((_path: string) => vi.fn())
    const dispose = bindApi(rest, storage('ops'), socket)

    expect(socket).not.toHaveBeenCalled()
    $boardSlug.set('ship')

    await vi.waitFor(() => {
      expect(socket).toHaveBeenCalledWith('/events?board=ship&since=7', expect.any(Function))
    })

    resolveOps(BOARD(14_386))
    await Promise.resolve()
    await Promise.resolve()

    expect(socket.mock.calls.map(([path]) => path)).toEqual(['/events?board=ship&since=7'])
    dispose()
  })

  it('resumes this connection from the last frame, and does not lend that cursor to another gateway', async () => {
    queryClient.setQueryData(boardKey('local', 'ship', false), BOARD(10))
    queryClient.setQueryData(boardKey('spark', 'ship', false), BOARD(4))

    const frames: Array<(data: unknown) => void> = []

    const socket = vi.fn((_path: string, onMessage: (data: unknown) => void) => {
      frames.push(onMessage)

      return vi.fn()
    })

    const dispose = bindApi(
      async () => {
        throw new Error('cached snapshots must not refetch')
      },
      {
        get: <T>(key: string, fallback: T) => (key.startsWith('boardSlug') ? ('ship' as T) : fallback),
        remove: vi.fn(),
        set: vi.fn()
      },
      socket
    )

    expect(socket).toHaveBeenCalledWith('/events?board=ship&since=10', expect.any(Function))
    frames.at(-1)!({ cursor: 25, events: [{ id: 25, kind: 'spawned', task_id: 't_1' }] })

    $boardSlug.set('ops')
    $boardSlug.set('ship')
    expect(socket.mock.calls.at(-1)?.[0]).toBe('/events?board=ship&since=25')

    setConnection({ connectionId: 'spark', mode: 'remote' } as never)
    await vi.waitFor(() => {
      expect(socket.mock.calls.at(-1)?.[0]).toBe('/events?board=ship&since=4')
    })

    dispose()
  })
})
