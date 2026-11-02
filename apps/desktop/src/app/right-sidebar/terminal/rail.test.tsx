import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { $connectionsRegistry } from '@/store/connections'
import { $bindings } from '@/store/keybinds'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { TerminalRail } from './rail'
import { $activeTerminalId, $terminals } from './terminals'

describe('TerminalRail', () => {
  beforeEach(() => {
    $terminals.set([{ auto: true, cwd: 'C:\\repo', id: 'term-1', kind: 'user', title: 'PowerShell' }])
    $activeTerminalId.set('term-1')
    $bindings.set({ ...$bindings.get(), 'view.showTerminal': ['ctrl+`'] })
  })

  afterEach(() => {
    cleanup()
    $terminals.set([])
    $activeTerminalId.set(null)
  })

  it('⌘-click closes the tab; a plain click selects it', () => {
    $terminals.set([...$terminals.get(), { auto: true, cwd: 'C:\\repo', id: 'term-2', kind: 'user', title: 'zsh' }])

    render(<TerminalRail />)

    fireEvent.click(screen.getByRole('tab', { name: '2. zsh' }), { metaKey: true })
    expect($terminals.get().map(term => term.id)).toEqual(['term-1'])

    fireEvent.click(screen.getByRole('tab', { name: '1. PowerShell' }))
    expect($activeTerminalId.get()).toBe('term-1')
    expect($terminals.get()).toHaveLength(1)
  })

  it('right-click on + opens the gateway menu and opens a tab on the picked gateway', async () => {
    $connectionsRegistry.set({
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        { id: 'nas', kind: 'ssh', label: 'NAS' }
      ],
      primary: 'local',
      version: 2
    } as unknown as DesktopConnectionsRegistry)

    try {
      render(<TerminalRail />)
      const plus = screen.getByRole('button', { name: 'New terminal' })

      // The app-wide menu leaves a marked trigger alone; an unmarked one gets the generic menu.
      expect(plus.hasAttribute('data-hermes-context-menu-trigger')).toBe(true)
      fireEvent.contextMenu(plus)

      fireEvent.click(await screen.findByRole('menuitem', { name: 'NAS' }))
      expect($terminals.get().at(-1)).toMatchObject({ connectionId: 'nas', kind: 'user' })
    } finally {
      $connectionsRegistry.set(null)
    }
  })

  it('a menu-opened SSH tab records the active profile and no path from another host', async () => {
    $connectionsRegistry.set({
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        { id: 'nas', kind: 'ssh', label: 'NAS' }
      ],
      primary: 'local',
      version: 2
    } as unknown as DesktopConnectionsRegistry)
    $connection.set({ connectionId: 'local' } as never)
    const profile = $activeGatewayProfile.get()
    $activeGatewayProfile.set('work')

    try {
      render(<TerminalRail />)
      fireEvent.contextMenu(screen.getByRole('button', { name: 'New terminal' }))
      fireEvent.click(await screen.findByRole('menuitem', { name: 'NAS' }))

      // The tunnel is keyed by (gateway, profile); This device's cwd means nothing on the NAS.
      expect($terminals.get().at(-1)).toMatchObject({ connectionId: 'nas', cwd: '', profile: 'work' })
    } finally {
      $activeGatewayProfile.set(profile)
      $connection.set(null)
      $connectionsRegistry.set(null)
    }
  })
})
