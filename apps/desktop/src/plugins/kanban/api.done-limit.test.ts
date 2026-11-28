import { afterEach, describe, expect, it, vi } from 'vitest'

// The done column is completion history: the board asks for its newest page
// only, and the status-bar pill reads counts instead of the board.

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const { $boardSlug, $doneLimit, bindApi, DONE_PAGE, fetchBoard, fetchStats, statsKey, boardKeyPrefix } =
  await import('./api')

const { queryClient } = await import('@/lib/query-client')

const noopStorage = { get: <T>(_key: string, fallback: T) => fallback, remove: vi.fn(), set: vi.fn() }

let dispose: (() => void) | undefined

afterEach(() => {
  dispose?.()
  dispose = undefined
  $doneLimit.set(DONE_PAGE)
  queryClient.clear()
})

function bind() {
  const rest = vi.fn(async (path: string) =>
    path === '/boards' ? { current: '', boards: [] } : { latest_event_id: 1 }
  )

  dispose = bindApi(
    rest as never,
    noopStorage,
    vi.fn(() => vi.fn())
  )

  return rest
}

describe('kanban done paging', () => {
  it('asks the board for one page of done cards and follows the limit', async () => {
    const rest = bind()
    $boardSlug.set('ops')

    await fetchBoard(false)
    expect(rest).toHaveBeenCalledWith('/board?done_limit=100&board=ops', undefined)

    $doneLimit.set(300)
    await fetchBoard(true)
    expect(rest).toHaveBeenCalledWith('/board?include_archived=true&done_limit=300&board=ops', undefined)
  })

  it('reads the status-bar counts from /stats under the board key prefix', async () => {
    const rest = bind()
    $boardSlug.set('ops')

    await fetchStats()
    expect(rest).toHaveBeenCalledWith('/stats?board=ops', undefined)
    // Board invalidations (prefix) must refresh the pill as well.
    expect(statsKey('local', 'ops').slice(0, 3)).toEqual([...boardKeyPrefix('local')])
  })
})
