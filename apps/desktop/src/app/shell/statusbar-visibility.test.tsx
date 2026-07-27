import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import { $statusbarHiddenIds, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'

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

afterEach(() => {
  cleanup()
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
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
      item('session-timer', 'Session timer', { variant: 'text' }),
      item('gateway-health', 'Gateway')
    ])

    for (const label of ['Turn timer', 'Context meter', 'Session timer']) {
      expect(screen.queryByText(label)).toBeNull()
    }

    openContextMenu(statusbar)

    const row = await screen.findByRole('menuitemcheckbox', { name: 'Session timer' })
    fireEvent.click(row)

    expect($statusbarHiddenIds.get()).not.toContain('session-timer')
    expect(within(statusbar).getByText('Session timer')).toBeTruthy()
  })
})
