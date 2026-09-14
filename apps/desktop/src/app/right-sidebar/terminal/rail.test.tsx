import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $bindings } from '@/store/keybinds'

import { TerminalRail } from './rail'
import { $activeTerminalId, $terminals } from './terminals'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

function openTabContextMenu(name: string) {
  const tab = screen.getByRole('tab', { name })

  // Radix ContextMenuTrigger opens on the secondary-button press + contextmenu pair.
  fireEvent.pointerDown(tab, { button: 2, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.contextMenu(tab, { button: 2 })
}

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

  it('renames a tab from its context menu, seeding the dialog with the current label', async () => {
    render(<TerminalRail />)

    openTabContextMenu('1. PowerShell')
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))

    const dialog = await screen.findByRole('dialog')
    const input = within(dialog).getByRole<HTMLInputElement>('textbox')
    expect(input.value).toBe('PowerShell')

    fireEvent.change(input, { target: { value: 'server' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }))

    // The dialog closes and the rail relabels immediately.
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByRole('tab', { name: '1. server' })).toBeTruthy()
    // A custom label pins the tab: the resolved shell name can no longer adopt over it.
    expect($terminals.get()[0]).toMatchObject({ auto: false, title: 'server' })
  })

  it('Enter saves the dialog, and an emptied name falls back to the previous label instead of blanking the tab', async () => {
    render(<TerminalRail />)

    openTabContextMenu('1. PowerShell')
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))

    const input = await screen.findByRole('textbox')
    fireEvent.change(input, { target: { value: '' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(screen.queryByRole('dialog')).toBeNull()
    expect($terminals.get()[0]?.title).toBe('PowerShell')
  })

  it('ignores the Enter that commits an IME composition, then saves on the next plain Enter', async () => {
    render(<TerminalRail />)

    openTabContextMenu('1. PowerShell')
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))

    const input = await screen.findByRole('textbox')
    fireEvent.change(input, { target: { value: 'サーバー' } })

    // With a ja/zh IME active, the Enter that ends the composition arrives
    // flagged as such — saving then would take a half-composed name.
    const composingEnter = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter' })
    Object.defineProperty(composingEnter, 'isComposing', { value: true })
    fireEvent(input, composingEnter)
    expect(screen.queryByRole('dialog')).not.toBeNull()
    expect($terminals.get()[0]?.title).toBe('PowerShell')

    // The following plain Enter (composition over) commits normally.
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(screen.queryByRole('dialog')).toBeNull()
    expect($terminals.get()[0]).toMatchObject({ auto: false, title: 'サーバー' })
  })

  it('treats a whitespace-only name like an emptied one, keeping the previous label', async () => {
    render(<TerminalRail />)

    openTabContextMenu('1. PowerShell')
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))

    const input = await screen.findByRole('textbox')
    fireEvent.change(input, { target: { value: '   ' } })
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Save' }))

    expect(screen.queryByRole('dialog')).toBeNull()
    expect($terminals.get()[0]?.title).toBe('PowerShell')
  })

  it('focuses the rename input on open and releases it on cancel', async () => {
    render(<TerminalRail />)

    openTabContextMenu('1. PowerShell')
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))

    const dialog = await screen.findByRole('dialog')
    const input = within(dialog).getByRole<HTMLInputElement>('textbox')

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(input))

    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
      // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
      expect(document.activeElement).not.toBe(input)
    })
  })
})
