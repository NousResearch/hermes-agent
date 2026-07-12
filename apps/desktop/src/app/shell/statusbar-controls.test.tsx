import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { StatusbarControls } from './statusbar-controls'

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

afterEach(cleanup)

function renderStatusbar(mode: 'auto-hide' | 'off' | 'on') {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <MemoryRouter>
        <StatusbarControls items={[{ id: 'session', label: 'Session 4:00', variant: 'text' }]} mode={mode} />
      </MemoryRouter>
    </I18nProvider>
  )
}

describe('Desktop status bar visibility', () => {
  it('renders normally in on mode', () => {
    renderStatusbar('on')

    expect(screen.getByRole('contentinfo')).not.toBeNull()
    expect(screen.queryByLabelText('Reveal the desktop status bar')).toBeNull()
  })

  it('does not render the status bar in off mode', () => {
    renderStatusbar('off')

    expect(screen.queryByRole('contentinfo')).toBeNull()
  })

  it('keeps a keyboard-focusable bottom-edge reveal target in auto-hide mode', () => {
    renderStatusbar('auto-hide')

    const revealZone = screen.getByLabelText('Reveal the desktop status bar')
    const statusbar = screen.getByRole('contentinfo')

    expect(revealZone.getAttribute('tabindex')).toBe('0')
    expect(statusbar.classList.contains('translate-y-full')).toBe(true)
    expect(statusbar.classList.contains('opacity-0')).toBe(true)
  })

  it('stays revealed while focus is inside a portaled status-bar menu', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <MemoryRouter>
          <StatusbarControls
            items={[
              {
                id: 'session',
                label: 'Session',
                menuItems: [{ id: 'settings', label: 'Settings' }],
                variant: 'menu'
              }
            ]}
            mode="auto-hide"
          />
        </MemoryRouter>
      </I18nProvider>
    )

    const revealZone = screen.getByLabelText('Reveal the desktop status bar')
    const statusbar = screen.getByRole('contentinfo')
    const trigger = screen.getByRole('button', { name: 'Session' })

    fireEvent.pointerDown(trigger, { button: 0 })

    const menuItem = await screen.findByRole('menuitem', { name: 'Settings' })
    menuItem.focus()

    expect(revealZone.contains(menuItem)).toBe(false)
    expect(menuItem.ownerDocument.activeElement).toBe(menuItem)
    expect(trigger.getAttribute('data-state')).toBe('open')
    expect(statusbar.classList.contains('has-data-[state=open]:translate-y-0')).toBe(true)
    expect(statusbar.classList.contains('has-data-[state=open]:opacity-100')).toBe(true)
  })
})
