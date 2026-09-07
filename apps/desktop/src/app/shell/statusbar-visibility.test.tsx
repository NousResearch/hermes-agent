import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import {
  $statusbarHiddenIds,
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
  it('hides the route/toggle items out of the box and keeps status items', () => {
    bar([
      item('cron', 'Cron'),
      item('webhooks', 'Webhooks'),
      item('agents', 'Agents'),
      item('terminal', 'Terminal'),
      item('approval-mode', 'Approvals'),
      item('gateway-health', 'Gateway')
    ])

    for (const label of ['Cron', 'Webhooks', 'Agents', 'Terminal', 'Approvals']) {
      expect(screen.queryByText(label)).toBeNull()
    }

    expect(screen.getByText('Gateway')).toBeTruthy()
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

  it('starts the per-turn session readouts hidden and restores them from the menu', async () => {
    const statusbar = bar([
      item('running-timer', 'Turn timer', { variant: 'text' }),
      item('context-usage', 'Context meter', { variant: 'menu' }),
      item('cache-hit-rate', 'Cache hit rate', { variant: 'text' }),
      item('tokens-per-second', 'Tokens per second', { variant: 'text' }),
      item('session-timer', 'Session timer', { variant: 'text' }),
      item('gateway-health', 'Gateway')
    ])

    for (const label of ['Turn timer', 'Context meter', 'Cache hit rate', 'Tokens per second', 'Session timer']) {
      expect(screen.queryByText(label)).toBeNull()
    }

    openContextMenu(statusbar)

    for (const [id, label] of [
      ['session-timer', 'Session timer'],
      ['cache-hit-rate', 'Cache hit rate']
    ]) {
      fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: label }))

      expect($statusbarHiddenIds.get()).not.toContain(id)
      expect(within(statusbar).getByText(label)).toBeTruthy()
    }
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
