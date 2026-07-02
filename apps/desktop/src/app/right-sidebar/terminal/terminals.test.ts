import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.terminals.v1'

async function loadTerminalStore() {
  vi.doMock('@/store/session', () => ({
    $currentCwd: atom('/workspace')
  }))

  return import('./terminals')
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
})
