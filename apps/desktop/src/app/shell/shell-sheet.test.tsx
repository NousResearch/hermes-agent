import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $widthClass } from '@/app/redesign/breakpoints'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { $narrowOverlayPaneIds } from '@/components/pane-shell/narrow-overlay-state'
import { ESCAPE_PRIORITY, pushEscapeLayer } from '@/lib/escape-layers'
import { CHAT_SIDEBAR_PANE_ID } from '@/store/layout'
import { $shellChoice } from '@/store/shell-variant'

import {
  $openSheet,
  $shellSheet,
  closeOpenSheet,
  installShellSheetAttribute,
  openShellSheet,
  toggleShellSheet
} from './shell-sheet'
import { ShellSheetController, ShellSheetScrim } from './shell-sheet-view'

beforeEach(() => {
  $widthClass.set('compact')
})

afterEach(() => {
  cleanup()
  $shellSheet.set(null)
  $narrowOverlayPaneIds.set(new Set())
  $widthClass.set('wide')
  $shellChoice.set(null)
})

describe('the open sheet', () => {
  it('names the band sheet, else the chat sidebar or a pane from the tree overlay', () => {
    expect($openSheet.get()).toBeNull()
    $narrowOverlayPaneIds.set(new Set(['files']))
    expect($openSheet.get()).toBe('pane')
    $narrowOverlayPaneIds.set(new Set(['sessions', CHAT_SIDEBAR_PANE_ID]))
    expect($openSheet.get()).toBe('sessions')
    openShellSheet('page')
    expect($openSheet.get()).toBe('page')
  })

  it('never shows on a wide window or in classic, and widening drops the band sheet', () => {
    openShellSheet('nav')
    $shellChoice.set(false)
    expect($openSheet.get()).toBeNull()
    expect($shellSheet.get()).toBeNull()

    $shellChoice.set(null)
    openShellSheet('page')
    $widthClass.set('wide')
    expect($shellSheet.get()).toBeNull()
  })

  it('toggles, and closes the tree overlay through its explicit close', () => {
    toggleShellSheet('nav')
    expect($shellSheet.get()).toBe('nav')
    toggleShellSheet('nav')
    expect($shellSheet.get()).toBeNull()

    const intents: unknown[] = []
    const listener = (event: Event) => intents.push((event as CustomEvent).detail)
    window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)
    $narrowOverlayPaneIds.set(new Set(['sessions', CHAT_SIDEBAR_PANE_ID]))
    closeOpenSheet()
    window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)
    expect(intents).toEqual([{ id: 'sessions', mode: 'close' }])
  })

  it('mirrors itself on the root for CSS', () => {
    const root = globalThis.globalThis.document.createElement('html')
    const stop = installShellSheetAttribute(root)
    expect(root.dataset.portalSheet).toBeUndefined()
    openShellSheet('page')
    expect(root.dataset.portalSheet).toBe('page')
    $shellSheet.set(null)
    expect(root.dataset.portalSheet).toBeUndefined()
    stop()
  })
})

function Go({ to }: { to: string }) {
  const navigate = useNavigate()

  return (
    <button onClick={() => navigate(to)} type="button">
      go
    </button>
  )
}

function Shell() {
  // A page panel renders only while its sheet is open, as usePageSidebarOpen() does below wide.
  const open = useStore($shellSheet) === 'page'

  return (
    <MemoryRouter>
      <button type="button">menu</button>
      <aside data-page-surface="">
        {open && (
          <div className="page-sidebar-panel">
            <button type="button">Agents</button>
            <button aria-current="page" type="button">
              default
            </button>
          </div>
        )}
      </aside>
      <Go to="/agents/ops" />
      <ShellSheetScrim />
      <ShellSheetController />
    </MemoryRouter>
  )
}

const flushFrame = () => act(() => new Promise(resolve => requestAnimationFrame(() => resolve(undefined))))

describe('sheet behaviour', () => {
  it('moves focus to the selected row when it opens and back to the menu when Esc closes it', async () => {
    render(<Shell />)
    const menu = screen.getByRole('button', { name: 'menu' })
    menu.focus()
    act(() => openShellSheet('page'))
    await flushFrame()
    expect(globalThis.document.activeElement).toBe(screen.getByRole('button', { name: 'default' }))

    fireEvent.keyDown(window, { key: 'Escape' })
    expect($shellSheet.get()).toBeNull()
    expect(globalThis.document.activeElement).toBe(menu)
  })

  it('takes Esc before the page under it', () => {
    let pageClosed = false
    const releasePage = pushEscapeLayer(ESCAPE_PRIORITY.overlay)

    const onPage = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented) {
        pageClosed = true
      }
    }

    render(<Shell />)
    act(() => openShellSheet('page'))
    window.addEventListener('keydown', onPage)
    fireEvent.keyDown(window, { key: 'Escape' })
    window.removeEventListener('keydown', onPage)
    releasePage()
    expect($shellSheet.get()).toBeNull()
    expect(pageClosed).toBe(false)
  })

  it('closes when a tap lands on its scrim', () => {
    const { container } = render(<Shell />)
    act(() => openShellSheet('nav'))
    fireEvent.pointerDown(container.querySelector('[data-slot="sheet-scrim"]')!)
    expect($shellSheet.get()).toBeNull()
    expect(container.querySelector('[data-slot="sheet-scrim"]')).toBeNull()
  })

  it('closes when a destination is picked (any navigation)', () => {
    render(<Shell />)
    act(() => openShellSheet('page'))
    fireEvent.click(screen.getByRole('button', { name: 'go' }))
    expect($shellSheet.get()).toBeNull()
  })
})
