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
    // No snapshot yet: the socket waits for /board, then opens. This gateway
    // has no latest_event_id, so the URL stays unscoped by since and the
    // server starts at the tail.
    await vi.waitFor(() => expect(dials).toEqual(['/events?board=ops']))

    // Boot publishes the local descriptor after plugins bound: same scope, no dial.
    setConnection({ mode: 'local' } as never)
    expect(dials).toEqual(['/events?board=ops'])

    // Different slug on the next gateway: exactly one dial, not one per listener.
    setConnection({ connectionId: 'spark', mode: 'remote' } as never)
    expect($boardSlug.get()).toBe('research')
    await vi.waitFor(() => expect(dials).toEqual(['/events?board=ops', '/events?board=research']))

    // Same slug on the way back to a gateway with an equal selection still
    // dials once — the backend behind the slug changed.
    stored.set('boardSlug', 'research')
    setConnection({ mode: 'local' } as never)
    expect($boardSlug.get()).toBe('research')
    await vi.waitFor(() =>
      expect(dials).toEqual(['/events?board=ops', '/events?board=research', '/events?board=research'])
    )

    // Writes land under the scope current at write time.
    $boardSlug.set('triage')
    expect(stored.get('boardSlug')).toBe('triage')
    expect(stored.get('boardSlug.spark')).toBe('research')

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
})
