import { act, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const pane = vi.hoisted(() => ({ visible: true }))

vi.mock(import('@/components/pane-shell/tree/store'), async importOriginal => ({
  ...(await importOriginal()),
  isPaneVisible: (id: string) => id === 'terminal' && pane.visible
}))

// The workspace test only needs the tab list, not live xterms.
vi.mock('./instance', () => ({
  AgentTerminalInstance: () => null,
  TerminalInstance: () => null
}))

vi.mock('./agent-terminal-stream', () => ({
  seedAgentTerminalCommand: vi.fn(),
  syncAgentTerminalSnapshot: vi.fn()
}))

const SUCCESS = 60_000
const FAILURE = 5 * 60_000

// Fresh module state (lastSeen/timers, surfacedProcs, tab list) per test.
async function load() {
  const autoclose = await import('./agent-terminal-autoclose')
  const terminals = await import('./terminals')

  return { ...autoclose, ...terminals }
}

type Store = Awaited<ReturnType<typeof load>>

const agentTab = (store: Store, procId: string) =>
  store.$terminals.get().find(term => term.kind === 'agent' && term.procId === procId)

describe('agent terminal auto-close', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useFakeTimers()
    window.localStorage.clear()
    pane.visible = true
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('exports the grace periods it promises', async () => {
    const store = await load()

    expect(store.AGENT_TAB_SUCCESS_GRACE_MS).toBe(SUCCESS)
    expect(store.AGENT_TAB_FAILURE_GRACE_MS).toBe(FAILURE)
  })

  it('closes a finished (exit 0) tab 60 s after the finish is observed', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-ok', 'npm run build')
    store.observeAgentProcessState('proc-ok', 'running')
    store.observeAgentProcessState('proc-ok', 'done')

    vi.advanceTimersByTime(SUCCESS - 1)
    expect(agentTab(store, 'proc-ok')).toBeDefined()

    vi.advanceTimersByTime(1)
    expect(agentTab(store, 'proc-ok')).toBeUndefined()
  })

  it('keeps a failed tab readable for 5 min, then closes it', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-bad', 'pytest')
    store.observeAgentProcessState('proc-bad', 'running')
    store.observeAgentProcessState('proc-bad', 'failed')

    vi.advanceTimersByTime(FAILURE - 1)
    expect(agentTab(store, 'proc-bad')).toBeDefined()

    vi.advanceTimersByTime(1)
    expect(agentTab(store, 'proc-bad')).toBeUndefined()
  })

  it('never closes a tab whose process is still running', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-live', 'npm run dev')

    for (let i = 0; i < 30; i++) {
      store.observeAgentProcessState('proc-live', 'running')
      vi.advanceTimersByTime(FAILURE)
    }

    expect(agentTab(store, 'proc-live')).toBeDefined()
  })

  it('a process seen running again cancels a pending close', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-flip', 'watch')
    store.observeAgentProcessState('proc-flip', 'done')
    vi.advanceTimersByTime(SUCCESS / 2)
    store.observeAgentProcessState('proc-flip', 'running')

    vi.advanceTimersByTime(FAILURE * 2)
    expect(agentTab(store, 'proc-flip')).toBeDefined()
  })

  it('never closes a tab whose process vanished while last seen running', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-gone', 'long job')
    store.observeAgentProcessState('proc-gone', 'running')
    // No further observations: its session stopped polling, the row is gone.

    vi.advanceTimersByTime(FAILURE * 10)
    expect(agentTab(store, 'proc-gone')).toBeDefined()
  })

  it('still closes a finished tab after its status row self-clears from the snapshot', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-cleared', 'make')
    store.observeAgentProcessState('proc-cleared', 'running')
    store.observeAgentProcessState('proc-cleared', 'done')
    // The status stack drops the row after 4 s; no more observations arrive.

    vi.advanceTimersByTime(SUCCESS)
    expect(agentTab(store, 'proc-cleared')).toBeUndefined()
  })

  it('defers the active tab of a visible pane until it is no longer on screen', async () => {
    const store = await load()
    const id = store.ensureAgentTerminal('proc-watched', 'deploy')!
    const other = store.ensureAgentTerminal('proc-other', 'lint')!
    store.selectTerminal(id)
    store.observeAgentProcessState('proc-watched', 'done')

    vi.advanceTimersByTime(SUCCESS * 5)
    expect(agentTab(store, 'proc-watched')).toBeDefined()

    store.selectTerminal(other)
    vi.advanceTimersByTime(store.AGENT_TAB_ON_SCREEN_RECHECK_MS)
    expect(agentTab(store, 'proc-watched')).toBeUndefined()
  })

  it('closes the active tab once the pane is hidden', async () => {
    const store = await load()
    const id = store.ensureAgentTerminal('proc-hidden', 'deploy')!
    store.selectTerminal(id)
    store.observeAgentProcessState('proc-hidden', 'done')

    vi.advanceTimersByTime(SUCCESS)
    expect(agentTab(store, 'proc-hidden')).toBeDefined()

    pane.visible = false
    vi.advanceTimersByTime(store.AGENT_TAB_ON_SCREEN_RECHECK_MS)
    expect(agentTab(store, 'proc-hidden')).toBeUndefined()
  })

  it('does not touch user tabs', async () => {
    const store = await load()
    const user = store.createTerminal('/repo')
    store.ensureAgentTerminal('proc-x', 'echo hi')
    store.selectTerminal(user)
    store.observeAgentProcessState('proc-x', 'done')

    vi.advanceTimersByTime(FAILURE * 2)
    expect(store.$terminals.get().map(term => term.id)).toEqual([user])
    expect(store.$activeTerminalId.get()).toBe(user)
  })

  it('a tab reopened from the status stack after auto-close stays until the process finishes again', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-re', 'build')
    store.observeAgentProcessState('proc-re', 'running')
    store.observeAgentProcessState('proc-re', 'done')
    vi.advanceTimersByTime(SUCCESS)
    expect(agentTab(store, 'proc-re')).toBeUndefined()

    // Reopen, then look elsewhere; repeated "done" snapshots must not re-close it.
    store.openAgentTerminal('proc-re', 'build')
    const user = store.createTerminal('/repo')
    store.selectTerminal(user)
    store.observeAgentProcessState('proc-re', 'done')
    vi.advanceTimersByTime(FAILURE * 2)
    expect(agentTab(store, 'proc-re')).toBeDefined()

    // A fresh run that finishes again does arm a new close.
    store.observeAgentProcessState('proc-re', 'running')
    store.observeAgentProcessState('proc-re', 'done')
    vi.advanceTimersByTime(SUCCESS)
    expect(agentTab(store, 'proc-re')).toBeUndefined()
  })

  it('a tab dismissed and reopened before the grace period ends is not closed by the old timer', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-swap', 'build')
    store.observeAgentProcessState('proc-swap', 'running')
    store.observeAgentProcessState('proc-swap', 'done')
    const first = agentTab(store, 'proc-swap')?.id

    // Dismiss it, then reopen it from the status row: a new terminal id.
    vi.advanceTimersByTime(SUCCESS / 2)
    store.closeAgentTerminalByProc('proc-swap')
    store.openAgentTerminal('proc-swap', 'build')
    const user = store.createTerminal('/repo')
    store.selectTerminal(user)
    const reopened = agentTab(store, 'proc-swap')?.id
    expect(reopened).toBeDefined()
    expect(reopened).not.toBe(first)

    vi.advanceTimersByTime(FAILURE * 2)
    expect(agentTab(store, 'proc-swap')?.id).toBe(reopened)
  })

  it('an auto-closed tab is not resurfaced by the next snapshot', async () => {
    const store = await load()
    store.ensureAgentTerminal('proc-once', 'build')
    store.observeAgentProcessState('proc-once', 'done')
    vi.advanceTimersByTime(SUCCESS)

    expect(store.ensureAgentTerminal('proc-once', 'build')).toBeNull()
    expect(agentTab(store, 'proc-once')).toBeUndefined()
  })
})

describe('TerminalWorkspace wiring', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useFakeTimers()
    window.localStorage.clear()
    pane.visible = true
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('auto-closes the tab of a background process the workspace saw finish', async () => {
    const { $backgroundStatusBySession } = await import('@/store/composer-status')
    const { TerminalWorkspace } = await import('./workspace')
    const { $terminals } = await import('./terminals')

    const item = (state: 'done' | 'running') => ({
      id: 'proc-ws',
      state,
      title: 'npm test',
      type: 'background' as const
    })

    render(<TerminalWorkspace onAddSelectionToChat={() => {}} />)
    act(() => $backgroundStatusBySession.set({ s1: [item('running')] }))
    expect($terminals.get().some(term => term.procId === 'proc-ws')).toBe(true)

    act(() => $backgroundStatusBySession.set({ s1: [item('done')] }))
    act(() => vi.advanceTimersByTime(SUCCESS))

    expect($terminals.get().some(term => term.procId === 'proc-ws')).toBe(false)
  })
})
