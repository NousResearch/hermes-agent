import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { OverlayView } from '@/app/overlays/overlay-view'
import { I18nProvider } from '@/i18n'

import { Sheet, SheetContent } from './sheet'
import { Sidebar, SidebarProvider } from './sidebar'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('navigation motion boundaries', () => {
  it('animates full-page overlay entry and exit', () => {
    vi.useFakeTimers()
    const onClose = vi.fn()

    render(
      <OverlayView closeLabel="Close preview" onClose={onClose}>
        <p>Overlay preview</p>
      </OverlayView>
    )

    const surface = screen.getByText('Overlay preview').closest('[data-overlay-surface]')
    const card = surface?.firstElementChild

    expect(card).not.toBeNull()
    expect(surface?.getAttribute('data-motion')).toBe('open')
    expect(card?.getAttribute('data-slot')).toBe('overlay-card')

    fireEvent.click(screen.getByRole('button', { name: 'Close preview' }))

    expect(surface?.getAttribute('data-motion')).toBe('closing')
    expect(onClose).not.toHaveBeenCalled()

    act(() => vi.advanceTimersByTime(180))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('animates sidebar sheets on mobile', () => {
    render(
      <I18nProvider configClient={null}>
        <Sheet open>
          <SheetContent showCloseButton={false}>Sheet preview</SheetContent>
        </Sheet>
      </I18nProvider>
    )

    const content = screen.getByText('Sheet preview').closest('[data-slot="sheet-content"]')
    const overlay = content?.parentElement?.querySelector('[data-slot="sheet-overlay"]')

    expect(content).not.toBeNull()
    expect(content?.className).toContain('data-[state=open]:animate-in')
    expect(content?.className).toContain('data-[state=open]:duration-[220ms]')
    expect(overlay?.className).toContain('data-[state=open]:animate-in')
  })

  it('animates desktop sidebar geometry as its state changes', () => {
    const { container } = render(
      <I18nProvider configClient={null}>
        <SidebarProvider onOpenChange={vi.fn()} open>
          <Sidebar collapsible="offcanvas">Sidebar preview</Sidebar>
        </SidebarProvider>
      </I18nProvider>
    )

    expect(container.querySelector('[data-slot="sidebar-gap"]')?.className).toContain('transition-[width]')
    expect(container.querySelector('[data-slot="sidebar-container"]')?.className).toContain(
      'transition-[left,right,width]'
    )
  })
})
