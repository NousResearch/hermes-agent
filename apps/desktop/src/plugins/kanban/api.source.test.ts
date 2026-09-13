import { afterEach, expect, it, vi } from 'vitest'
import { atom, type WritableAtom } from 'nanostores'
const state = vi.hoisted(() => ({ notify: vi.fn(), invalidate: vi.fn(), update: vi.fn() }))
vi.mock('@hermes/plugin-sdk', () => ({
  atom,
  host: { state: { connectionId: atom('local'), profile: atom('default') } },
  queryClient: { setQueriesData: state.update, invalidateQueries: state.invalidate }
}))
vi.mock('./completion-notify', () => ({
  bindCompletionNotify: vi.fn(),
  onKanbanEventsFrame: state.notify.mockResolvedValue(undefined)
}))
const { host } = await import('@hermes/plugin-sdk')
const { bindApi } = await import('./api')
let dispose: (() => void) | undefined
afterEach(() => {
  dispose?.()
  vi.clearAllMocks()
})
it.each(['connectionId', 'profile'] as const)(
  'rebinds on %s changes and rejects late frames even after returning',
  key => {
    const frames: Array<(data: unknown) => void> = []
    const closes: Array<ReturnType<typeof vi.fn>> = []
    dispose = bindApi(
      async <T>() => ({}) as T,
      {
        get: <T>(_key: string, fallback: T) => fallback,
        set: vi.fn(),
        remove: vi.fn()
      },
      (_path, callback) => {
        frames.push(callback)
        const close = vi.fn()
        closes.push(close)
        return close
      }
    )
    const source = host.state[key] as WritableAtom<string | null>
    const original = source.get()
    source.set('other')
    expect(closes[0]).toHaveBeenCalledOnce()
    const frame = { events: [{ id: 999, task_id: 'task', kind: 'completed' }] }
    frames[0](frame)
    expect(state.notify).not.toHaveBeenCalled()
    expect(state.update).not.toHaveBeenCalled()
    frames.at(-1)!(frame)
    expect(state.notify).toHaveBeenCalledWith(
      '',
      frame.events,
      `${host.state.connectionId.get()}::${host.state.profile.get()}`
    )
    state.notify.mockClear()
    source.set(original)
    frames[0](frame)
    expect(state.notify).not.toHaveBeenCalled()
  }
)
