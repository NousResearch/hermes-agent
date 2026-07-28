import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import { TooltipDetails } from '@/components/ui/tooltip'
import {
  $statusbarHiddenIds,
  $statusbarVisible,
  STATUSBAR_HIDDEN_BY_DEFAULT,
  toggleStatusbarVisible
} from '@/store/statusbar-prefs'

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
  it('groups version-tooltip details below its primary label in one panel', async () => {
    bar([
      item('version-client', 'v1.2.3', {
        title: (
          <>
            <span>Hermes Desktop v1.2.3</span>
            <TooltipDetails className="flex flex-col gap-0.5">
              <span>Commit abc1234</span>
              <span>Branch ethie/hermes-version</span>
            </TooltipDetails>
          </>
        )
      })
    ])

    fireEvent.pointerMove(screen.getByRole('button', { name: 'v1.2.3' }), { pointerType: 'mouse' })

    const tooltip = await screen.findByRole('tooltip')
    const label = tooltip.firstElementChild

    expect(tooltip.textContent).toBe('Hermes Desktop v1.2.3Commit abc1234Branch ethie/hermes-version')
    expect(label?.classList.contains('bg-foreground')).toBe(true)
    expect(label?.classList.contains('flex-col')).toBe(true)
    expect(screen.getByText('Commit abc1234').parentElement?.classList.contains('text-background/65')).toBe(true)
  })

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
