import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import {
  $statusbarHiddenIds,
  $statusbarItemOrder,
  $statusbarVisible,
  STATUSBAR_HIDDEN_BY_DEFAULT,
  toggleStatusbarVisible
} from '@/store/statusbar-prefs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
  $statusbarItemOrder.set({ left: [], right: [] })
  $statusbarVisible.set(true)
})

const item = (id: string, label: string, extra: Partial<StatusbarItem> = {}): StatusbarItem => ({
  id,
  label,
  toggleLabel: label,
  variant: 'action',
  ...extra
})

function bar(items: StatusbarItem[]) {
  render(
    <MemoryRouter>
      <StatusbarControls items={items} />
    </MemoryRouter>
  )

  return screen.getByRole('contentinfo')
}

/** Radix opens a ContextMenu on contextmenu after a pointerdown positions it. */
function openContextMenu(target: HTMLElement) {
  fireEvent.pointerDown(target, { button: 2, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.contextMenu(target, { button: 2 })
}

describe('statusbar item visibility', () => {
  it('uses the saved right-cluster order while keeping an unknown item in its shipped position', () => {
    $statusbarHiddenIds.set([])
    $statusbarItemOrder.set({ left: [], right: ['gateway', 'cron'] })
    const statusbar = bar([item('workspace', 'Workspace'), item('cron', 'Cron'), item('gateway', 'Gateway')])

    const labels = Array.from(statusbar.querySelectorAll('button')).map(button => button.textContent)
    expect(labels).toEqual(['Workspace', 'Gateway', 'Cron'])
  })

  it('shows an item once the user enables it from the bar context menu', async () => {
    const statusbar = bar([item('cron', 'Cron'), item('gateway-health', 'Gateway')])

    expect(screen.queryByText('Cron')).toBeNull()

    openContextMenu(statusbar)

    const row = await screen.findByRole('menuitemcheckbox', { name: 'Cron' })
    fireEvent.click(row)

    expect($statusbarHiddenIds.get()).not.toContain('cron')
    expect(within(statusbar).getByText('Cron')).toBeTruthy()
  })

  it('never lets the user hide a locked item (system icon / update pill)', async () => {
    const statusbar = bar([item('command-center', 'Command Center', { lockedVisible: true })])

    openContextMenu(statusbar)

    const row = await screen.findByRole('menuitemcheckbox', { name: 'Command Center' })
    expect(row.getAttribute('data-disabled')).not.toBeNull()
    expect(row.getAttribute('aria-checked')).toBe('true')
  })

  it('leaves items that never opted into the menu alone', () => {
    $statusbarHiddenIds.set(['plugin-thing'])
    bar([{ id: 'plugin-thing', label: 'Plugin thing', variant: 'action' }])

    expect(screen.getByText('Plugin thing')).toBeTruthy()
  })
})

describe('reset to defaults', () => {
  it('puts a customized bar back to the shipped show/hide set', async () => {
    $statusbarHiddenIds.set(['gateway-health'])

    const statusbar = bar([item('cron', 'Cron'), item('gateway-health', 'Gateway')])

    expect(screen.queryByText('Gateway')).toBeNull()
    expect(within(statusbar).getByText('Cron')).toBeTruthy()

    openContextMenu(statusbar)
    fireEvent.click(await screen.findByRole('menuitem', { name: /reset to defaults/i }))

    expect($statusbarHiddenIds.get()).toEqual([...STATUSBAR_HIDDEN_BY_DEFAULT])
    expect($statusbarItemOrder.get()).toEqual({ left: [], right: [] })
    expect(within(statusbar).getByText('Gateway')).toBeTruthy()
    // Scoped to the bar: the menu stays open after a reset, so an unscoped query
    // matches its still-listed 'Cron' checkbox row rather than a bar item.
    expect(within(statusbar).queryByText('Cron')).toBeNull()
  })

  it('disables the row when the layout is already default', async () => {
    const statusbar = bar([item('cron', 'Cron'), item('gateway-health', 'Gateway')])

    openContextMenu(statusbar)

    const row = await screen.findByRole('menuitem', { name: /reset to defaults/i })
    expect(row.getAttribute('data-disabled')).not.toBeNull()
  })

  it('enables the row as soon as one item differs, in either direction', async () => {
    const statusbar = bar([item('cron', 'Cron'), item('gateway-health', 'Gateway')])

    // Showing a default-hidden item counts…
    $statusbarHiddenIds.set(STATUSBAR_HIDDEN_BY_DEFAULT.filter(id => id !== 'cron'))
    openContextMenu(statusbar)
    expect(
      (await screen.findByRole('menuitem', { name: /reset to defaults/i })).getAttribute('data-disabled')
    ).toBeNull()

    // …and so does hiding a default-shown one.
    $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT, 'gateway-health'])
    expect(
      (await screen.findByRole('menuitem', { name: /reset to defaults/i })).getAttribute('data-disabled')
    ).toBeNull()
  })

  it('leaves whole-bar visibility alone — reset is about items, not the bar', async () => {
    // Set to the NON-default so a reset that wrongly restored bar visibility too
    // would flip this back to true and fail. StatusbarControls doesn't read the
    // atom (the controller gates the mount), so the menu is still reachable here.
    $statusbarVisible.set(false)
    $statusbarHiddenIds.set([])

    const statusbar = bar([item('gateway-health', 'Gateway')])

    openContextMenu(statusbar)
    fireEvent.click(await screen.findByRole('menuitem', { name: /reset to defaults/i }))

    expect($statusbarHiddenIds.get()).toEqual([...STATUSBAR_HIDDEN_BY_DEFAULT])
    expect($statusbarVisible.get()).toBe(false)
  })
})

describe('whole-bar visibility', () => {
  it('hides the bar from the context menu, leaving the keybind as the way back', async () => {
    const statusbar = bar([item('gateway-health', 'Gateway')])

    openContextMenu(statusbar)
    fireEvent.click(await screen.findByRole('menuitem', { name: /hide status bar/i }))

    expect($statusbarVisible.get()).toBe(false)

    toggleStatusbarVisible()
    expect($statusbarVisible.get()).toBe(true)
  })

  it('offers the hide row even when no item opted into the show/hide list', async () => {
    const statusbar = bar([{ id: 'plugin-thing', label: 'Plugin thing', variant: 'action' }])

    openContextMenu(statusbar)

    expect(await screen.findByRole('menuitem', { name: /hide status bar/i })).toBeTruthy()
    expect(screen.queryByRole('menuitemcheckbox')).toBeNull()
  })
})

describe('statusbar menu reordering', () => {
  it('persists an actual pointer drop without toggling visibility', async () => {
    const statusbar = bar([item('alpha', 'Alpha'), item('beta', 'Beta')])
    openContextMenu(statusbar)
    const handle = await screen.findByRole('button', { name: 'Reorder Alpha' })

    const measure = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      const rows = Array.from(document.querySelectorAll('[role="menuitemcheckbox"]'))
      const index = rows.indexOf(this)

      return new DOMRect(0, Math.max(0, index) * 24, 200, 24)
    })

    try {
      await act(async () => {
        fireEvent.pointerDown(handle, { button: 0, isPrimary: true, pointerId: 1, clientX: 10, clientY: 12 })
        await Promise.resolve()
      })
      await act(async () => {
        fireEvent.pointerMove(document, { pointerId: 1, clientX: 10, clientY: 36 })
        await Promise.resolve()
      })
      await act(async () => {
        fireEvent.pointerUp(document, { pointerId: 1, clientX: 10, clientY: 36 })
        await Promise.resolve()
      })
      await waitFor(() => expect($statusbarItemOrder.get().right).toEqual(['beta', 'alpha']))
      expect($statusbarItemOrder.get().left).toEqual([])
      expect(Array.from(statusbar.querySelectorAll('button')).map(button => button.textContent)).toEqual([
        'Beta',
        'Alpha'
      ])
      expect($statusbarHiddenIds.get()).toEqual([...STATUSBAR_HIDDEN_BY_DEFAULT])
    } finally {
      measure.mockRestore()
    }
  })
})
