import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $sessionStates } from '@/store/session-states'
import { $todosBySession, setSessionTodos } from '@/store/todos'

/**
 * Regression guard for "does the task overview survive a relaunch?" — the
 * user explicitly asked for this after learning the previous (in-memory-only)
 * implementation reset on every app restart. A real restart cannot be
 * exercised in vitest (it's a fresh Electron process), so this simulates it
 * the way `persistentAtom` itself is proven safe elsewhere in this codebase:
 * reset all module-level state, then RE-IMPORT the store module fresh via
 * `vi.resetModules()` so it re-reads from `localStorage` exactly as a cold
 * boot would (module-level `readJson()` calls only run once, at import time).
 */
describe('session-todos-overview persistence across a simulated relaunch', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $todosBySession.set({})
    $sessionStates.set({})
  })

  afterEach(() => {
    window.localStorage.clear()
    $todosBySession.set({})
  })

  it('reloads a previously captured todo list after the module is re-imported', async () => {
    const mod1 = await import('./session-todos-overview')

    mod1.resetSessionTodoOverview()
    setSessionTodos('sess-1', [{ content: '재시작 후에도 남아야 하는 항목', id: '1', status: 'pending' }])

    expect(mod1.sessionTodoOverviewRows()).toHaveLength(1)
    expect(window.localStorage.getItem('hermes.desktop.taskOverview.snapshots')).toBeTruthy()

    // Simulate an app relaunch: fresh module graph, fresh module-level reads
    // (the real failure mode: an in-memory-only Map/atom would come back
    // empty here because nothing wrote it to a durable store).
    const { vi } = await import('vitest')

    vi.resetModules()

    const mod2 = await import('./session-todos-overview')
    const rows = mod2.sessionTodoOverviewRows()

    expect(rows).toHaveLength(1)
    expect(rows[0]?.todos[0]?.content).toBe('재시작 후에도 남아야 하는 항목')
  })

  it('keeps a dismissed row dismissed after a simulated relaunch', async () => {
    const mod1 = await import('./session-todos-overview')

    mod1.resetSessionTodoOverview()
    setSessionTodos('sess-2', [{ content: '숨겨질 항목', id: '1', status: 'pending' }])
    mod1.dismissSessionTodoOverviewRow('sess-2')

    expect(mod1.sessionTodoOverviewRows()).toHaveLength(0)

    const { vi } = await import('vitest')

    vi.resetModules()

    const mod2 = await import('./session-todos-overview')

    expect(mod2.sessionTodoOverviewRows()).toHaveLength(0)
  })
})
