import { beforeEach, describe, expect, it } from 'vitest'

import { $terminalInjection, $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { setConnectionsRegistry } from '@/store/connections'
import { $connection } from '@/store/session'

import { closeSessionTerminal, openSessionInTerminalPane, tuiResumeCommand } from './open-session-tui'
import { $activeTerminalId, $terminals, createTerminal } from './terminals'

// The in-app twin of "Open in terminal": a plugin's "chat ⇄ TUI" toggle needs
// a fresh tab on the gateway the session lives on, the terminal taken over,
// and exactly one `hermes --tui --resume <id>` queued for that tab. Anything
// that would resume a session on the WRONG host (a URL/token gateway's session
// from this machine's shell) must refuse, not silently run a `hermes` here.

const registry = {
  version: 1,
  primary: 'local',
  connections: [
    { id: 'local', kind: 'local', label: 'This device' },
    { id: 'homelab', kind: 'ssh', label: 'Homelab', host: 'homelab' },
    { id: 'vps', kind: 'remote', label: 'VPS', url: 'https://vps.example', authMode: 'token' }
  ]
} as never

describe('tuiResumeCommand', () => {
  it('pins a slug profile and quotes every argv word', () => {
    expect(tuiResumeCommand('20260101_120000_abc', 'work')).toBe(
      "'hermes' '--profile' 'work' '--tui' '--resume' '20260101_120000_abc'"
    )
  })

  it('drops a profile that must not cross into argv (#88842)', () => {
    expect(tuiResumeCommand('sess', 'Display Name')).toBe("'hermes' '--tui' '--resume' 'sess'")
    expect(tuiResumeCommand('sess', '')).toBe("'hermes' '--tui' '--resume' 'sess'")
    expect(tuiResumeCommand('sess', null)).toBe("'hermes' '--tui' '--resume' 'sess'")
  })
})

describe('openSessionInTerminalPane', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $terminals.set([])
    $activeTerminalId.set(null)
    $terminalInjection.set(null)
    setTerminalTakeover(false)
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' } as never)
  })

  it('opens a fresh tab, takes the terminal over, and queues the resume once', () => {
    const result = openSessionInTerminalPane('20260101_120000_abc', { profile: 'work', cwd: '/repo' })

    expect(result.ok).toBe(true)

    if (!result.ok) {
      return
    }

    const tab = $terminals.get().find(term => term.id === result.terminalId)

    expect(tab).toMatchObject({ cwd: '/repo', kind: 'user', title: 'TUI', auto: false })
    expect($activeTerminalId.get()).toBe(result.terminalId)
    expect($terminalTakeover.get()).toBe(true)
    expect($terminalInjection.get()).toBe(result.command)
    expect(result.command).toBe("'hermes' '--profile' 'work' '--tui' '--resume' '20260101_120000_abc'")
  })

  it('runs on an SSH gateway when that is the active source', () => {
    $connection.set({ connectionId: 'homelab', mode: 'remote' } as never)

    const result = openSessionInTerminalPane('sess-1')

    expect(result.ok).toBe(true)
    expect($terminals.get().at(-1)).toMatchObject({ connectionId: 'homelab' })
  })

  it('refuses a URL/token gateway instead of resuming a remote session locally', () => {
    $connection.set({ connectionId: 'vps', mode: 'remote' } as never)

    const result = openSessionInTerminalPane('sess-1')

    expect(result).toMatchObject({ ok: false, reason: 'no-shell' })
    expect($terminals.get()).toEqual([])
    expect($terminalInjection.get()).toBeNull()
    expect($terminalTakeover.get()).toBe(false)
  })

  it('closes only the tabs it opened', () => {
    const result = openSessionInTerminalPane('sess-1')
    const foreign = createTerminal('/elsewhere')

    expect(result.ok).toBe(true)
    expect(closeSessionTerminal(foreign)).toBe(false)
    expect($terminals.get().some(term => term.id === foreign)).toBe(true)

    if (result.ok) {
      expect(closeSessionTerminal(result.terminalId)).toBe(true)
      expect($terminals.get().some(term => term.id === result.terminalId)).toBe(false)
      expect(closeSessionTerminal(result.terminalId)).toBe(false)
    }
  })

  it('refuses an unusable session id without touching the pane', () => {
    for (const bad of ['', '   ', 'a b', "x'; rm -rf /", 'é']) {
      expect(openSessionInTerminalPane(bad)).toMatchObject({ ok: false, reason: 'invalid-session' })
    }

    expect($terminals.get()).toEqual([])
    expect($terminalTakeover.get()).toBe(false)
  })
})
