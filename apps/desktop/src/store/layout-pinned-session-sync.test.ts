import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getPins: vi.fn(),
  savePins: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getDesktopPinnedSessions: (...args: unknown[]) => api.getPins(...args),
  saveDesktopPinnedSessions: (...args: unknown[]) => api.savePins(...args)
}))

const PIN_STORAGE_KEY = 'hermes.desktop.pinnedSessions'
const PIN_DIRTY_STORAGE_KEY = 'hermes.desktop.pinnedSessionsDirty'

interface Deferred<T> {
  promise: Promise<T>
  reject: (reason?: unknown) => void
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((accept, fail) => {
    resolve = accept
    reject = fail
  })

  return { promise, reject, resolve }
}

async function loadLayout(localPins: string[] = []) {
  if (localPins.length > 0) {
    window.localStorage.setItem(PIN_STORAGE_KEY, JSON.stringify(localPins))
  }

  return import('./layout')
}

describe('durable pinned-session synchronization', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/')
    window.localStorage.clear()
    api.getPins.mockReset()
    api.savePins.mockReset()
    api.savePins.mockResolvedValue({ ok: true, pinned_session_ids: [] })
    vi.resetModules()
  })

  afterEach(() => {
    window.history.replaceState({}, '', '/')
    window.localStorage.clear()
    vi.useRealTimers()
  })

  it('merges a pin made during recovery with the recovered list', async () => {
    const recovery = deferred<{ exists: boolean; pinned_session_ids: string[] }>()
    api.getPins.mockReturnValue(recovery.promise)
    const layout = await loadLayout()

    const hydration = layout.hydratePinnedSessionIds()
    layout.pinSession('new-pin')
    recovery.resolve({ exists: true, pinned_session_ids: ['saved-a', 'saved-b'] })
    await hydration

    await vi.waitFor(() => {
      expect(layout.$pinnedSessionIds.get()).toEqual(['saved-a', 'saved-b', 'new-pin'])
      expect(api.savePins).toHaveBeenLastCalledWith(['saved-a', 'saved-b', 'new-pin'])
    })
  })

  it('persists a mutation made while the legacy migration write is pending', async () => {
    api.getPins.mockResolvedValue({ exists: false, pinned_session_ids: [] })
    const migrationWrite = deferred<{ ok: boolean; pinned_session_ids: string[] }>()
    api.savePins.mockImplementationOnce(() => migrationWrite.promise).mockResolvedValue({
      ok: true,
      pinned_session_ids: ['legacy-pin', 'new-pin']
    })
    const layout = await loadLayout(['legacy-pin'])

    await layout.hydratePinnedSessionIds()
    await vi.waitFor(() => expect(api.savePins).toHaveBeenCalledWith(['legacy-pin']))
    layout.pinSession('new-pin')
    migrationWrite.resolve({ ok: true, pinned_session_ids: ['legacy-pin'] })

    await vi.waitFor(() => {
      expect(api.savePins).toHaveBeenLastCalledWith(['legacy-pin', 'new-pin'])
      expect(window.localStorage.getItem(PIN_DIRTY_STORAGE_KEY)).toBe('false')
    })
  })

  it('keeps a locally dirty list authoritative after a failed final write and restart', async () => {
    window.localStorage.setItem(PIN_DIRTY_STORAGE_KEY, 'true')
    api.getPins.mockResolvedValue({ exists: true, pinned_session_ids: ['stale-remote'] })
    const layout = await loadLayout(['newer-local'])

    await layout.hydratePinnedSessionIds()

    await vi.waitFor(() => {
      expect(layout.$pinnedSessionIds.get()).toEqual(['newer-local'])
      expect(api.savePins).toHaveBeenLastCalledWith(['newer-local'])
      expect(window.localStorage.getItem(PIN_DIRTY_STORAGE_KEY)).toBe('false')
    })
  })

  it('leaves the local dirty marker set when the final write exhausts its retries', async () => {
    vi.useFakeTimers()
    api.getPins.mockResolvedValue({ exists: true, pinned_session_ids: [] })
    api.savePins.mockRejectedValue(new Error('offline'))
    const layout = await loadLayout()

    await layout.hydratePinnedSessionIds()
    layout.pinSession('newer-local')
    expect(window.localStorage.getItem(PIN_DIRTY_STORAGE_KEY)).toBe('true')

    await vi.runAllTimersAsync()
    expect(api.savePins).toHaveBeenCalledTimes(3)
    expect(window.localStorage.getItem(PIN_DIRTY_STORAGE_KEY)).toBe('true')
  })

  it('coalesces duplicate StrictMode hydration calls into one backend read', async () => {
    const recovery = deferred<{ exists: boolean; pinned_session_ids: string[] }>()
    api.getPins.mockReturnValue(recovery.promise)
    const layout = await loadLayout()

    const first = layout.hydratePinnedSessionIds()
    const second = layout.hydratePinnedSessionIds()
    recovery.resolve({ exists: true, pinned_session_ids: ['saved-a'] })
    await Promise.all([first, second])

    expect(api.getPins).toHaveBeenCalledTimes(1)
    expect(layout.$pinnedSessionIds.get()).toEqual(['saved-a'])
  })

  it('never hydrates or persists pins from a secondary session window', async () => {
    window.history.replaceState({}, '', '/?win=secondary#/session-a')
    api.getPins.mockResolvedValue({ exists: true, pinned_session_ids: ['saved-a'] })
    const layout = await loadLayout(['stale-secondary'])

    await layout.hydratePinnedSessionIds()
    layout.pinSession('secondary-only')
    await Promise.resolve()

    expect(api.getPins).not.toHaveBeenCalled()
    expect(api.savePins).not.toHaveBeenCalled()
    expect(window.localStorage.getItem(PIN_DIRTY_STORAGE_KEY)).toBeNull()
  })
})
