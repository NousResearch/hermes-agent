import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { AppContextMenu } from '@/app/context-menu/app-context-menu'
import { $contextMenu } from '@/app/context-menu/store'
import { StatusbarControls } from '@/app/shell/statusbar-controls'
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
  $contextMenu.set(null)
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
})

// End-to-end companion to the ContextMenuTrigger unit test: that one proves
// the marker survives the Slot merge, this one proves the real status bar and
// the real coordinator agree, so the default-hidden items stay reachable.
describe('statusbar context menu coordination', () => {
  it('lets the statusbar Radix menu handle right-clicks instead of the app fallback (#89953)', async () => {
    render(
      <MemoryRouter>
        <AppContextMenu />
        <StatusbarControls
          items={[
            {
              id: 'context-usage',
              label: 'Context meter',
              toggleLabel: 'Context meter',
              variant: 'menu'
            }
          ]}
        />
      </MemoryRouter>
    )

    const statusbar = screen.getByRole('contentinfo')

    fireEvent.pointerDown(statusbar, { button: 2, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.contextMenu(statusbar, { button: 2 })

    expect(await screen.findByRole('menuitemcheckbox', { name: 'Context meter' })).toBeTruthy()
    expect(screen.queryByText('Settings')).toBeNull()
    expect($contextMenu.get()).toBeNull()
  })
})
