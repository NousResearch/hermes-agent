import { afterEach, describe, expect, it, vi } from 'vitest'

const doors = vi.hoisted(() => ({
  notify: vi.fn(),
  navigate: vi.fn(),
  setConnection: vi.fn<(id: string | null) => void>()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')
  const { QueryClient } = await import('@tanstack/react-query')
  const connectionId = atom<string | null>(null)
  doors.setConnection.mockImplementation(id => connectionId.set(id))

  return {
    atom,
    queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
    host: { ...doors, state: { connectionId }, activeConnectionId: () => connectionId.get() },
    useValue: vi.fn(),
    usePluginI18n: vi.fn(),
    captureGatewayFileDownload: vi.fn()
  }
})

const { queryClient } = await import('@hermes/plugin-sdk')
const { bindApi, $boardSlug } = await import('./api')
let dispose: (() => void) | undefined

afterEach(() => {
  dispose?.()
  dispose = undefined
  doors.setConnection(null)
  queryClient.clear()
  vi.clearAllMocks()
})

describe('default board event notifications', () => {
  it('refreshes an open alias-keyed task drawer on a live event', async () => {
    const { QueryObserver } = await import('@tanstack/react-query')
    const { taskKey } = await import('./api')
    let frame!: (data: unknown) => void
    dispose = bindApi(
      (async (path: string) => (path === '/boards' ? { current: 'drawer-board' } : { latest_event_id: 100 })) as never,
      { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
      (_path, callback) => {
        frame = callback

        return vi.fn()
      }
    )
    let revision = 1

    const observer = new QueryObserver(queryClient, {
      queryKey: taskKey('local', '', 'task'),
      queryFn: async () => ({ revision }),
      staleTime: Infinity
    })

    const unsubscribe = observer.subscribe(() => undefined)

    try {
      await vi.waitFor(() => expect(observer.getCurrentResult().data).toEqual({ revision: 1 }))
      await vi.waitFor(() => expect(frame).toBeTypeOf('function'))
      revision = 2
      frame({ cursor: 101, events: [{ id: 101, kind: 'updated', task_id: 'task' }] })
      await vi.waitFor(() => expect(observer.getCurrentResult().data).toEqual({ revision: 2 }))
    } finally {
      unsubscribe()
    }
  })

  it.each(['empty', 'rejected'])('preserves the live alias socket when board resolution is %s', async mode => {
    const rest = vi.fn(async (path: string) => {
      if (path === '/boards') {
        if (mode === 'rejected') {
          throw new Error('offline')
        }

        return { current: '' }
      }

      return { latest_event_id: 20 }
    })

    const socket = vi.fn(() => vi.fn())
    dispose = bindApi(rest as never, { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() }, socket)
    await vi.waitFor(() => expect(socket).toHaveBeenCalledWith('/events?since=20', expect.any(Function)))
    expect(doors.notify).not.toHaveBeenCalled()
  })

  it.each(['selection', 'dispose'])('ignores alias resolution after %s', async change => {
    let resolveBoards!: (value: unknown) => void

    const boards = new Promise(resolve => {
      resolveBoards = resolve
    })

    const rest = vi.fn(async (path: string) => (path === '/boards' ? boards : { latest_event_id: 30 }))
    const socket = vi.fn(() => vi.fn())
    dispose = bindApi(rest as never, { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() }, socket)

    if (change === 'selection') {
      $boardSlug.set('chosen')
      await vi.waitFor(() => expect(socket).toHaveBeenCalledWith('/events?board=chosen&since=30', expect.any(Function)))
    } else {
      dispose()
      dispose = undefined
    }

    resolveBoards({ current: 'late' })
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(socket).toHaveBeenCalledTimes(change === 'selection' ? 1 : 0)
    expect(rest).not.toHaveBeenCalledWith('/board?board=late')
  })

  it('does not resolve or override a caller-selected board', async () => {
    const rest = vi.fn(async () => ({ latest_event_id: 30 }))
    const socket = vi.fn(() => vi.fn())
    dispose = bindApi(
      rest as never,
      {
        get: <T>(key: string, fallback: T) => (key === 'boardSlug' ? ('chosen' as T) : fallback),
        set: vi.fn(),
        remove: vi.fn()
      },
      socket
    )
    await vi.waitFor(() => expect(socket).toHaveBeenCalledWith('/events?board=chosen&since=30', expect.any(Function)))
    expect(rest).not.toHaveBeenCalledWith('/boards')
  })

  it('pins the socket to the resolved board and notifies a post-baseline blocked event once', async () => {
    const callbacks: Array<(data: unknown) => void> = []

    const socket = vi.fn((_path: string, cb: (data: unknown) => void) => {
      callbacks.push(cb)

      return vi.fn()
    })

    let latest = 100

    const rest = vi.fn(async (path: string) => {
      if (path === '/boards') {
        return { current: 'default', boards: [{ slug: 'default' }] }
      }

      if (path === '/board?board=default' || path === '/board') {
        return { latest_event_id: latest }
      }
      throw new Error(`Unexpected REST call: ${path}`)
    })

    const storage = { get: <T>(_key: string, fallback: T) => fallback, set: vi.fn(), remove: vi.fn() }
    dispose = bindApi(rest as never, storage, socket)
    await vi.waitFor(() => expect(callbacks).toHaveLength(1))
    expect(socket).toHaveBeenCalledWith('/events?board=default&since=100', expect.any(Function))
    expect($boardSlug.get()).toBe('')
    // Establish the existing notifier's baseline from a nonterminal frame;
    // first-live-frame baseline seeding is separately owned by PR #116236.
    latest = 101
    callbacks[0]({ cursor: 101, events: [{ id: 101, kind: 'created' }] })
    await vi.waitFor(() => expect(rest.mock.calls.filter(([path]) => path === '/board?board=default')).toHaveLength(2))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(doors.notify).not.toHaveBeenCalled()
    latest = 102
    callbacks[0]({
      cursor: 102,
      events: [
        { id: 99, kind: 'completed' },
        { id: 102, kind: 'blocked', task_id: 'task', payload: { reason: 'Need approval' } }
      ]
    })
    await vi.waitFor(() => expect(doors.notify).toHaveBeenCalledTimes(1))
    expect(doors.notify).toHaveBeenCalledWith(expect.objectContaining({ message: 'Need approval' }))
    callbacks[0]({ cursor: 102, events: [{ id: 102, kind: 'blocked' }] })
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(doors.notify).toHaveBeenCalledTimes(1)
    dispose()
    dispose = undefined
    callbacks[0]({ cursor: 103, events: [{ id: 103, kind: 'blocked' }] })
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(doors.notify).toHaveBeenCalledTimes(1)
  })
})
