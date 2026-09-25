import { QueryObserver } from '@tanstack/react-query'
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// A board lives on ONE gateway; the kanban data layer follows the active
// connection. See the scope comments in ./api.ts.

const routed = vi.hoisted(() => ({ id: null as null | string }))

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activeGatewayConnectionId: () => routed.id
}))

const { $boardSlug, bindApi, boardsKey, useKanbanScope } = await import('./api')
const { setConnection } = await import('@/store/session')
const { queryClient } = await import('@/lib/query-client')
const { host } = await import('@hermes/plugin-sdk')

// A socket with no known cursor dials after the board snapshot resolves.
const settle = () => new Promise(resolve => setTimeout(resolve, 0))

const noopStorage = { get: <T>(_key: string, fallback: T) => fallback, remove: vi.fn(), set: vi.fn() }

afterEach(() => {
  setConnection(null)
  routed.id = null
  queryClient.clear()
})

describe('kanban connection scope', () => {
  it('render-time keys follow the active connection', () => {
    const { result } = renderHook(() => useKanbanScope())

    expect(boardsKey(result.current)).toEqual(['kanban', 'boards', 'local'])

    // No rerender(): the subscription itself must re-render the component.
    act(() => setConnection({ connectionId: 'spark', mode: 'remote' } as never))

    expect(boardsKey(result.current)).toEqual(['kanban', 'boards', 'spark'])
  })

  it('remembers the slug per connection and dials the socket once per switch', async () => {
    const stored = new Map<string, unknown>([
      ['boardSlug', 'ops'],
      ['boardSlug.spark', 'research']
    ])

    const storage = {
      get: <T>(key: string, fallback: T) => (stored.has(key) ? (stored.get(key) as T) : fallback),
      remove: vi.fn(),
      set: (key: string, value: unknown) => void stored.set(key, value)
    }

    const dials: string[] = []

    const socket = vi.fn((path: string) => {
      dials.push(path)

      return vi.fn()
    })

    const dispose = bindApi(async () => ({}) as never, storage, socket)

    expect($boardSlug.get()).toBe('ops')
    await settle()
    expect(dials).toEqual(['/events?board=ops'])

    // Boot publishes the local descriptor after plugins bound: same scope, no dial.
    setConnection({ mode: 'local' } as never)
    await settle()
    expect(dials).toEqual(['/events?board=ops'])

    // Different slug on the next gateway: exactly one dial, not one per listener.
    setConnection({ connectionId: 'spark', mode: 'remote' } as never)
    expect($boardSlug.get()).toBe('research')
    await settle()
    expect(dials).toEqual(['/events?board=ops', '/events?board=research'])

    // Same slug on the way back to a gateway with an equal selection still
    // dials once — the backend behind the slug changed.
    stored.set('boardSlug', 'research')
    setConnection({ mode: 'local' } as never)
    expect($boardSlug.get()).toBe('research')
    await settle()
    expect(dials).toEqual(['/events?board=ops', '/events?board=research', '/events?board=research'])

    // Writes land under the scope current at write time.
    $boardSlug.set('triage')
    expect(stored.get('boardSlug')).toBe('triage')
    expect(stored.get('boardSlug.spark')).toBe('research')

    dispose()
  })

  it('an event cursor never crosses to another connection', async () => {
    // Both gateways have a board named `ship`, with unrelated event ids.
    const storage = {
      ...noopStorage,
      get: <T>(key: string, fallback: T) => (key.startsWith('boardSlug') ? 'ship' : fallback) as T
    }

    const dials: Array<{ onMessage: (data: unknown) => void; path: string }> = []

    const socket = vi.fn((path: string, onMessage: (data: unknown) => void) => {
      dials.push({ onMessage, path })

      return vi.fn()
    })

    const dispose = bindApi(async () => ({}) as never, storage, socket)

    await settle()
    dials.at(-1)!.onMessage({ cursor: 14_386, events: [{ id: 14_386, kind: 'created', task_id: 't_1' }] })

    routed.id = 'spark'
    setConnection({ connectionId: 'spark', mode: 'remote' } as never)
    await settle()
    expect(dials.at(-1)!.path).toBe('/events?board=ship')

    // Back on local, local's own cursor is still there.
    routed.id = null
    setConnection({ mode: 'local' } as never)
    expect(dials.at(-1)!.path).toBe('/events?board=ship&since=14386')

    dispose()
  })

  it('an observer still keyed to the outgoing scope is not refetched onto the incoming backend', async () => {
    const dispose = bindApi(
      async () => ({}) as never,
      noopStorage,
      vi.fn(() => vi.fn())
    )

    const fetches: Array<null | string> = []

    const observer = new QueryObserver(queryClient, {
      queryFn: async () => {
        fetches.push(routed.id)

        return { boards: [] }
      },
      queryKey: boardsKey('local')
    })

    const unsubscribe = observer.subscribe(() => undefined)
    await vi.waitFor(() => expect(observer.getCurrentResult().status).toBe('success'))
    expect(fetches).toEqual([null])

    // The request tag has moved to spark but React has not re-keyed the
    // observer yet: the switch's invalidation must skip it.
    routed.id = 'spark'
    await queryClient.invalidateQueries()
    expect(fetches).toEqual([null])

    // Back on local the same observer is live again.
    routed.id = null
    await queryClient.invalidateQueries()
    expect(fetches).toEqual([null, null])

    unsubscribe()
    dispose()
  })

  it('completion notifications baseline each connection on its own, even for a shared board slug', async () => {
    // Both gateways have a board named `ship`, with unrelated event-id sequences.
    const storage = {
      ...noopStorage,
      get: <T>(key: string, fallback: T) => (key.startsWith('boardSlug') ? 'ship' : fallback) as T
    }

    const latest: Record<string, number> = { local: 100, spark: 3 }
    const rest = vi.fn(async () => ({ latest_event_id: latest[routed.id ?? 'local'] }))
    const frames: Array<(data: unknown) => void> = []

    const socket = vi.fn((_path: string, onMessage: (data: unknown) => void) => {
      frames.push(onMessage)

      return vi.fn()
    })

    const notify = vi.spyOn(host, 'notify').mockImplementation(() => '')
    const dispose = bindApi(rest as never, storage, socket)

    // Each socket dials from its board snapshot, seeding that connection's baseline.
    await vi.waitFor(() => expect(frames).toHaveLength(1))
    frames.at(-1)!({ events: [{ id: 100, kind: 'created', task_id: 't_local' }] })

    routed.id = 'spark'
    setConnection({ connectionId: 'spark', mode: 'remote' } as never)
    await vi.waitFor(() => expect(frames).toHaveLength(2))
    // spark's stream replays its history (id 2) and then a fresh completion (id 4).
    frames.at(-1)!({
      events: [
        { id: 2, kind: 'completed', task_id: 't_old' },
        { id: 4, kind: 'completed', task_id: 't_new' }
      ]
    })

    await vi.waitFor(() => expect(notify).toHaveBeenCalledTimes(1))
    expect(notify.mock.calls[0][0]).toMatchObject({ message: 't_new' })

    notify.mockRestore()
    dispose()
  })
})
