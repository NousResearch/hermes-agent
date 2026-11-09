import { afterEach, describe, expect, it, vi } from 'vitest'

// t_1af76b41: each peer "File > New Window" Kanban tab must remember its OWN
// selected board across quit/relaunch, not the last window to write the
// shared localStorage key (the same root cause as t_7690aaea's tiles/layout
// bug — Electron partitions localStorage by origin, not by BrowserWindow).

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const { $boardSlug, bindApi } = await import('./api')

afterEach(() => {
  window.history.replaceState({}, '', '/')
})

// A fake localStorage-backed PluginStorage: a single Map shared across every
// "window" in the test, exactly like the real shared origin-scoped storage.
function sharedStorage() {
  const backing = new Map<string, unknown>()

  return {
    backing,
    storage: {
      get: <T>(key: string, fallback: T) => (backing.has(key) ? (backing.get(key) as T) : fallback),
      remove: (key: string) => void backing.delete(key),
      set: (key: string, value: unknown) => void backing.set(key, value)
    }
  }
}

describe('kanban board slug is scoped per peer window', () => {
  it('two peer windows writing the shared storage never clobber each other', () => {
    const { backing, storage } = sharedStorage()
    const socket = vi.fn(() => vi.fn())

    // Window A boots first and picks board "alpha".
    window.history.replaceState({}, '', '/?peer=1&windowId=win-a')
    const disposeA = bindApi(async () => ({}) as never, storage, socket)
    $boardSlug.set('alpha')
    expect(backing.get('boardSlug::win:win-a')).toBe('alpha')
    disposeA()

    // Window B boots second and picks board "beta" — must land under its
    // OWN key, not overwrite window A's entry (the pre-fix bug).
    window.history.replaceState({}, '', '/?peer=1&windowId=win-b')
    const disposeB = bindApi(async () => ({}) as never, storage, socket)
    $boardSlug.set('beta')
    expect(backing.get('boardSlug::win:win-b')).toBe('beta')
    expect(backing.get('boardSlug::win:win-a')).toBe('alpha')
    disposeB()

    // "Relaunch": window A reopens with the SAME windowId (restored by
    // electron/window-session-state.ts's BUILD-STABLE-WINDOW-ID) and must
    // come back on its own board, unaffected by window B's write.
    window.history.replaceState({}, '', '/?peer=1&windowId=win-a')
    const disposeA2 = bindApi(async () => ({}) as never, storage, socket)
    expect($boardSlug.get()).toBe('alpha')
    disposeA2()

    window.history.replaceState({}, '', '/?peer=1&windowId=win-b')
    const disposeB2 = bindApi(async () => ({}) as never, storage, socket)
    expect($boardSlug.get()).toBe('beta')
    disposeB2()
  })

  it('the main window (no windowId) keeps the legacy unscoped key', () => {
    const { backing, storage } = sharedStorage()
    const socket = vi.fn(() => vi.fn())

    window.history.replaceState({}, '', '/')
    const dispose = bindApi(async () => ({}) as never, storage, socket)
    $boardSlug.set('ops')
    expect(backing.get('boardSlug')).toBe('ops')
    expect(backing.has('boardSlug::win:win-a')).toBe(false)
    dispose()
  })

  it('a malformed/missing peer windowId degrades to the legacy key, not a crash', () => {
    const { backing, storage } = sharedStorage()
    const socket = vi.fn(() => vi.fn())

    window.history.replaceState({}, '', '/?peer=1&windowId=bad/id')
    const dispose = bindApi(async () => ({}) as never, storage, socket)
    $boardSlug.set('triage')
    expect(backing.get('boardSlug')).toBe('triage')
    dispose()
  })
})
