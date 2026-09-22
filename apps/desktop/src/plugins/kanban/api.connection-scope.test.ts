import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// A board lives on ONE gateway. The kanban data layer follows the active
// connection: keys built during render change with it (so an observer is a
// clean cache miss on a switch), the selected slug is remembered per
// connection (the local pool keeps the bare key — the lib/connection-scoped.ts
// contract, and the slug picked before per-connection keys existed survives),
// and the events socket dials the new backend exactly once per switch.

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const { $boardSlug, bindApi, boardsKey, useKanbanScope } = await import('./api')
const { setConnection } = await import('@/store/session')

afterEach(() => {
  setConnection(null)
})

describe('kanban connection scope', () => {
  it('render-time keys follow the active connection', () => {
    const { result } = renderHook(() => useKanbanScope())

    expect(boardsKey(result.current)).toEqual(['kanban', 'boards', 'local'])

    // No rerender(): the subscription itself must re-render the component.
    act(() => setConnection({ connectionId: 'spark', mode: 'remote' } as never))

    expect(boardsKey(result.current)).toEqual(['kanban', 'boards', 'spark'])
  })

  it('remembers the slug per connection and dials the socket once per switch', () => {
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
    expect(dials).toEqual(['/events?board=ops'])

    // Boot publishes the local descriptor after plugins bound: same scope, no dial.
    setConnection({ mode: 'local' } as never)
    expect(dials).toEqual(['/events?board=ops'])

    // Different slug on the next gateway: exactly one dial, not one per listener.
    setConnection({ connectionId: 'spark', mode: 'remote' } as never)
    expect($boardSlug.get()).toBe('research')
    expect(dials).toEqual(['/events?board=ops', '/events?board=research'])

    // Same slug on the way back to a gateway with an equal selection still
    // dials once — the backend behind the slug changed.
    stored.set('boardSlug', 'research')
    setConnection({ mode: 'local' } as never)
    expect($boardSlug.get()).toBe('research')
    expect(dials).toEqual(['/events?board=ops', '/events?board=research', '/events?board=research'])

    // Writes land under the scope current at write time.
    $boardSlug.set('triage')
    expect(stored.get('boardSlug')).toBe('triage')
    expect(stored.get('boardSlug.spark')).toBe('research')

    dispose()
  })
})
