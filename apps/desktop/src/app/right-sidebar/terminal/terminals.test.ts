import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.terminals.v1'

async function loadTerminalStore() {
  const $currentCwd = atom('/workspace')

  vi.doMock('@/store/session', () => ({
    $currentCwd
  }))

  return { ...(await import('./terminals')), $currentCwd }
}

describe('terminal store persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('restores user tabs, active tab, and history on module load', async () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        activeTerminalId: 'term-two',
        terminals: [
          { auto: false, cwd: '/repo/one', id: 'term-one', reviveBuffer: 'last output', title: 'zsh' },
          { auto: true, cwd: '/repo/two', id: 'term-two', title: 'Terminal' }
        ]
      })
    )

    const { $activeTerminalId, $terminals } = await loadTerminalStore()

    expect($activeTerminalId.get()).toBe('term-two')
    expect($terminals.get()).toEqual([
      { auto: false, cwd: '/repo/one', id: 'term-one', kind: 'user', reviveBuffer: 'last output', title: 'zsh' },
      { auto: true, cwd: '/repo/two', id: 'term-two', kind: 'user', title: 'Terminal' }
    ])
  })

  it('persists user tabs and history synchronously, skipping agent mirrors', async () => {
    const { createTerminal, ensureAgentTerminal, renameTerminal, selectTerminal, updateTerminalReviveBuffer } =
      await loadTerminalStore()

    const userId = createTerminal('/repo')
    renameTerminal(userId, 'server')
    updateTerminalReviveBuffer(userId, 'recent scrollback')
    ensureAgentTerminal('proc-1', 'background task')
    selectTerminal(userId)

    // No flush/tick: persistence is synchronous, so the snapshot is already on
    // disk (this is what makes app-quit restore reliable).
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}')).toEqual({
      activeTerminalId: userId,
      terminals: [{ auto: false, cwd: '/repo', id: userId, reviveBuffer: 'recent scrollback', title: 'server' }]
    })
  })

  it('never attaches a revive buffer to an agent tab', async () => {
    const { $terminals, ensureAgentTerminal, updateTerminalReviveBuffer } = await loadTerminalStore()

    const agentId = ensureAgentTerminal('proc-1', 'background task')!
    updateTerminalReviveBuffer(agentId, 'should be ignored')

    expect($terminals.get().find(term => term.id === agentId)?.reviveBuffer).toBeUndefined()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('tail-trims an oversized revive buffer to stay under the storage budget', async () => {
    const { $terminals, createTerminal, updateTerminalReviveBuffer } = await loadTerminalStore()

    const userId = createTerminal('/repo')
    const huge = 'x'.repeat(60_000)
    updateTerminalReviveBuffer(userId, huge)

    const stored = $terminals.get().find(term => term.id === userId)?.reviveBuffer ?? ''
    expect(stored.length).toBe(48_000)
    expect(stored).toBe(huge.slice(-48_000))
  })

  it('clears remembered tabs when all terminals close', async () => {
    const { closeAllTerminals, createTerminal } = await loadTerminalStore()

    createTerminal('/repo')
    expect(window.localStorage.getItem(STORAGE_KEY)).not.toBeNull()

    closeAllTerminals()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('restores and persists the last observed cwd so a reopened tab lands where the user cd-d', async () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        activeTerminalId: 'term-one',
        terminals: [{ auto: false, cwd: '/repo', id: 'term-one', restoreCwd: '/repo/packages/api', title: 'zsh' }]
      })
    )

    const { $terminals, updateTerminalRestoreCwd } = await loadTerminalStore()

    expect($terminals.get()[0]?.restoreCwd).toBe('/repo/packages/api')

    updateTerminalRestoreCwd('term-one', '/repo/packages/web')
    expect($terminals.get()[0]?.restoreCwd).toBe('/repo/packages/web')
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}').terminals[0].restoreCwd).toBe(
      '/repo/packages/web'
    )
  })

  it('never attaches a restore cwd to an agent tab and ignores empty values', async () => {
    const { $terminals, createTerminal, ensureAgentTerminal, updateTerminalRestoreCwd } = await loadTerminalStore()

    const userId = createTerminal('/repo')
    const agentId = ensureAgentTerminal('proc-1', 'background task')!

    updateTerminalRestoreCwd(agentId, '/somewhere')
    updateTerminalRestoreCwd(userId, '   ')

    expect($terminals.get().find(term => term.id === agentId)?.restoreCwd).toBeUndefined()
    expect($terminals.get().find(term => term.id === userId)?.restoreCwd).toBeUndefined()
  })
})

describe('session cwd → terminal tab linking', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('re-selects the tab already pointed at the new session cwd (trailing slash tolerated)', async () => {
    const { $activeTerminalId, $currentCwd, createTerminal } = await loadTerminalStore()

    const repoTab = createTerminal('/repo')
    const otherTab = createTerminal('/elsewhere')
    expect($activeTerminalId.get()).toBe(otherTab)

    $currentCwd.set('/repo/')
    expect($activeTerminalId.get()).toBe(repoTab)
  })

  it('matches the live shell cwd (restoreCwd) over the launch dir', async () => {
    const { $activeTerminalId, $currentCwd, createTerminal, updateTerminalRestoreCwd } = await loadTerminalStore()

    const movedTab = createTerminal('/repo')
    updateTerminalRestoreCwd(movedTab, '/repo/packages/api')
    const otherTab = createTerminal('/elsewhere')
    expect($activeTerminalId.get()).toBe(otherTab)

    $currentCwd.set('/repo/packages/api')
    expect($activeTerminalId.get()).toBe(movedTab)

    // The launch dir no longer describes where that shell lives.
    $currentCwd.set('/repo')
    expect($activeTerminalId.get()).toBe(movedTab)
  })

  it('leaves the active tab alone when no tab lives in the session cwd or the cwd is empty', async () => {
    const { $activeTerminalId, $currentCwd, createTerminal } = await loadTerminalStore()

    createTerminal('/repo')
    const activeTab = createTerminal('/elsewhere')

    $currentCwd.set('/unrelated')
    expect($activeTerminalId.get()).toBe(activeTab)

    $currentCwd.set('')
    expect($activeTerminalId.get()).toBe(activeTab)
  })

  it('stays put when the active tab already lives in the target cwd, and never matches agent tabs', async () => {
    const { $activeTerminalId, $currentCwd, createTerminal, ensureAgentTerminal, selectTerminal } =
      await loadTerminalStore()

    const first = createTerminal('/repo')
    const second = createTerminal('/repo')
    ensureAgentTerminal('proc-1', 'background task')
    selectTerminal(second)

    // Both tabs match; the one already active keeps focus (no first-match steal).
    $currentCwd.set('/repo')
    expect($activeTerminalId.get()).toBe(second)

    selectTerminal(first)
    $currentCwd.set('/repo')
    expect($activeTerminalId.get()).toBe(first)
  })
})

describe('shared WebGL atlas refresh fan-out', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('coalesces multiple requests into one frame and drops unregistered terminals', async () => {
    const { redrawAllTerminals, registerWebglRefresh } = await loadTerminalStore()

    const termA = { refresh: vi.fn(), rows: 24 }
    const termB = { refresh: vi.fn(), rows: 24 }
    const getWebgl = () => ({ clearTextureAtlas: vi.fn() }) as never
    const unregister = registerWebglRefresh(termA as never, getWebgl)
    registerWebglRefresh(termB as never, getWebgl)

    // A theme switch fires one redraw request per mounted terminal; they must
    // coalesce into a single atlas clear/refresh pass.
    redrawAllTerminals()
    redrawAllTerminals()
    expect(termA.refresh).not.toHaveBeenCalled()
    expect(termB.refresh).not.toHaveBeenCalled()

    await new Promise<void>(resolve => {
      requestAnimationFrame(() => resolve())
    })

    expect(termA.refresh).toHaveBeenCalledTimes(1)
    expect(termB.refresh).toHaveBeenCalledTimes(1)

    // A terminal that disposes (tab close) must stop being refreshed.
    unregister()
    redrawAllTerminals()
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => resolve())
    })

    expect(termA.refresh).toHaveBeenCalledTimes(1)
    expect(termB.refresh).toHaveBeenCalledTimes(2)
  })

  it('drops a terminal unregistered while a refresh frame is pending', async () => {
    const { redrawAllTerminals, registerWebglRefresh } = await loadTerminalStore()

    const termA = { refresh: vi.fn(), rows: 24 }
    const termB = { refresh: vi.fn(), rows: 24 }
    const getWebgl = () => ({ clearTextureAtlas: vi.fn() }) as never
    registerWebglRefresh(termA as never, getWebgl)
    const unregister = registerWebglRefresh(termB as never, getWebgl)

    // Both are pending in the same frame; second disposes before it flushes.
    redrawAllTerminals()
    unregister()
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => resolve())
    })

    expect(termA.refresh).toHaveBeenCalledTimes(1)
    expect(termB.refresh).not.toHaveBeenCalled()
  })

  it('skips the caller terminal so it is not force-cleared through the fan-out', async () => {
    const { redrawAllTerminals, registerWebglRefresh } = await loadTerminalStore()

    const caller = { refresh: vi.fn(), rows: 24 }
    const sibling = { refresh: vi.fn(), rows: 24 }
    const getWebgl = () => ({ clearTextureAtlas: vi.fn() }) as never
    registerWebglRefresh(caller as never, getWebgl)
    registerWebglRefresh(sibling as never, getWebgl)

    // A font change clears the caller inline (applyTerminalFontFamily), so the
    // fan-out must not re-clear it — only the siblings sharing the atlas.
    redrawAllTerminals(caller as never)
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => resolve())
    })

    expect(caller.refresh).not.toHaveBeenCalled()
    expect(sibling.refresh).toHaveBeenCalledTimes(1)
  })

  it('clears all atlases before refreshing any terminal (two-phase ordering)', async () => {
    const { redrawAllTerminals, registerWebglRefresh } = await loadTerminalStore()

    // Track operation order across terminals sharing one atlas.
    const ops: string[] = []
    const sharedAtlas = { clearTextureAtlas: () => ops.push('clear') }
    const getWebgl = () => sharedAtlas as never

    const termA = {
      refresh: () => ops.push('refresh-A'),
      rows: 24
    }
    const termB = {
      refresh: () => ops.push('refresh-B'),
      rows: 24
    }

    registerWebglRefresh(termA as never, getWebgl)
    registerWebglRefresh(termB as never, getWebgl)

    redrawAllTerminals()
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => resolve())
    })

    // Both clears must precede both refreshes. If a clear ran after a refresh,
    // that terminal's rebuilt model would reference freed atlas rows.
    const clearIndices = ops.map((op, i) => op === 'clear' ? i : -1).filter(i => i >= 0)
    const refreshIndices = ops.map((op, i) => op.startsWith('refresh-') ? i : -1).filter(i => i >= 0)

    expect(clearIndices).toHaveLength(2)
    expect(refreshIndices).toHaveLength(2)
    expect(Math.max(...clearIndices)).toBeLessThan(Math.min(...refreshIndices))
  })
})

