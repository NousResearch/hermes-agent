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
    expect(Object.keys(host.terminal).sort()).toEqual(['canOpenSession', 'openSession', 'setTakeover', 'takeover'])
  })

  it('openSession queues the TUI resume and takes the pane over; setTakeover collapses it again', () => {
    expect(host.terminal.canOpenSession()).toBe(true)

    const result = host.terminal.openSession('20260101_120000_abc', { profile: 'default' })

    expect(result.ok).toBe(true)
    expect($terminalInjection.get()).toBe("'hermes' '--profile' 'default' '--tui' '--resume' '20260101_120000_abc'")
    expect(host.terminal.takeover.get()).toBe(true)
    expect($terminalTakeover.get()).toBe(true)

    host.terminal.setTakeover(false)
    expect(host.terminal.takeover.get()).toBe(false)
    // Collapsing never kills the tab: the TUI keeps running behind the rail.
    expect($terminals.get()).toHaveLength(1)
  })
})
