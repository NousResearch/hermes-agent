import { beforeEach, describe, expect, it } from 'vitest'

import { $terminalInjection, $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { $terminals } from '@/app/right-sidebar/terminal/terminals'
import { setConnectionsRegistry } from '@/store/connections'
import { $connection } from '@/store/session'

import { host } from './index'

// Plugins get ONE narrow terminal door: resume a Hermes session in the in-app
// TUI and expand/collapse the pane. They never get a "type this into the
// user's shell" verb (ctx.runAction refuses the terminal actions for the same
// reason), so a chat ⇄ TUI toggle is possible without widening the trust model.

describe('host.terminal', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $terminals.set([])
    $terminalInjection.set(null)
    setTerminalTakeover(false)
    setConnectionsRegistry({
      version: 1,
      primary: 'local',
      connections: [{ id: 'local', kind: 'local', label: 'This device' }]
    } as never)
    $connection.set({ connectionId: 'local', mode: 'local' } as never)
  })

  it('exposes only session-resume, takeover and its live state', () => {
    expect(Object.keys(host.terminal).sort()).toEqual(['canOpenSession', 'closeSession', 'openSession', 'setTakeover', 'takeover'])
  })

  it('openSession queues the TUI resume and takes the pane over; setTakeover collapses it again', () => {
    expect(host.terminal.canOpenSession()).toBe(true)

    const result = host.terminal.openSession('20260101_120000_abc', { profile: 'default' })

    expect(result.ok).toBe(true)
    expect($terminalInjection.get()?.command).toBe("'hermes' '--profile' 'default' '--tui' '--resume' '20260101_120000_abc'")
    expect(host.terminal.takeover.get()).toBe(true)
    expect($terminalTakeover.get()).toBe(true)

    host.terminal.setTakeover(false)
    expect(host.terminal.takeover.get()).toBe(false)
    // Collapsing never kills the tab: the TUI keeps running behind the rail.
    expect($terminals.get()).toHaveLength(1)
  })

  it('closeSession closes a tab openSession returned and refuses any other', () => {
    const result = host.terminal.openSession('sess-1')

    expect(result.ok).toBe(true)
    expect(host.terminal.closeSession('not-ours')).toBe(false)

    if (result.ok) {
      expect(host.terminal.closeSession(result.terminalId)).toBe(true)
      expect($terminals.get()).toHaveLength(0)
    }
  })

  it('a restart pins the resume to the NEW tab even while the old one is still mounted', () => {
    const first = host.terminal.openSession('sess-1')

    expect(first.ok).toBe(true)
    $terminalInjection.set(null) // the first tab consumed its command

    const second = host.terminal.openSession('sess-1')

    expect(second.ok).toBe(true)

    if (first.ok && second.ok) {
      expect($terminalInjection.get()).toEqual({ command: second.command, terminalId: second.terminalId })
      expect(host.terminal.closeSession(first.terminalId)).toBe(true)
    }
  })
})
