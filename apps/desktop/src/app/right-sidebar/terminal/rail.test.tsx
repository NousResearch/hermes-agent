import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $bindings } from '@/store/keybinds'

import { TERMINAL_COLOR_VARS, TerminalRail } from './rail'
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

  it('keeps a hotkey label in inline flow inside the portaled tooltip decoration', async () => {
    const view = render(<TerminalRail />)

    fireEvent.pointerMove(screen.getByRole('tab', { name: '1. PowerShell' }), { pointerType: 'mouse' })
    await screen.findByRole('tooltip')

    const content = document.querySelector<HTMLElement>('[data-slot="tooltip-content"]')
    const decoration = content?.firstElementChild

    expect(content).not.toBeNull()
    expect(view.container.contains(content)).toBe(false)
    // No flex box under the decoration: its per-line background only wraps
    // inline flow, so a flex label would hang its overflow dark-on-dark.
    expect(decoration?.querySelector('.flex, .inline-flex')).toBeNull()
    expect(decoration?.textContent).toContain('PowerShell')
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

  it('renders the tab icon and tint the entry carries, falling back to the kind default', () => {
    $terminals.set([
      { auto: false, color: 'green', cwd: 'C:\\repo', icon: 'server', id: 'term-1', kind: 'user', title: 'api' },
      { auto: true, cwd: 'C:\\repo', id: 'term-2', kind: 'user', title: 'plain' }
    ])

    render(<TerminalRail />)

    const styled = screen.getByRole('tab', { name: '1. api' }).querySelector('[data-terminal-color]')
    expect(styled?.classList.contains('codicon-server')).toBe(true)
    expect(styled?.getAttribute('data-terminal-color')).toBe('green')
    expect((styled as HTMLElement | null)?.style.color).toBe(TERMINAL_COLOR_VARS.green)

    const plain = screen.getByRole('tab', { name: '2. plain' }).querySelector('[data-terminal-color]')
    expect(plain).toBeNull()
  })
})
