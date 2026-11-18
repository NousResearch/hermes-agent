import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $widthClass } from '@/app/redesign/breakpoints'
import { $historyPosition } from '@/app/shell/history-position'
import { $shellSheet } from '@/app/shell/shell-sheet'
import { $interfaceMode } from '@/store/interface-mode'
import { $sidebarOpen, setSidebarOpen } from '@/store/layout'

import { usePageSidebarOpen } from './page-sidebar'
import { PageToolbar } from './page-toolbar'

afterEach(() => {
  cleanup()
  act(() => setSidebarOpen(true))
})

function PanelPage() {
  usePageSidebarOpen()

  return null
}

it('offers "show sidebar" beside Back only while a page panel is folded away', () => {
  const onBack = vi.fn()
  const { rerender } = render(<PageToolbar onBack={onBack} />)

  act(() => setSidebarOpen(false))
  expect(screen.queryByRole('button', { name: 'Show sidebar' })).toBeNull()

  rerender(
    <>
      <PageToolbar onBack={onBack} />
      <PanelPage />
    </>
  )
  fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
  expect($sidebarOpen.get()).toBe(true)
  expect(screen.queryByRole('button', { name: 'Show sidebar' })).toBeNull()

  fireEvent.click(screen.getByRole('button', { name: 'Back' }))
  expect(onBack).toHaveBeenCalledOnce()
})

it('brings a folded page panel back from a floating button at the stage in Simple mode, not the band', () => {
  const previousMode = $interfaceMode.get()
  $interfaceMode.set('simple')
  render(
    <>
      <PageToolbar onBack={vi.fn()} />
      <PanelPage />
    </>
  )

  act(() => setSidebarOpen(false))
  expect(screen.queryByRole('button', { name: 'Show sidebar' })).toBeNull()
  const reveal = screen.getByRole('button', { name: 'Show panel' })
  expect(reveal.getAttribute('data-slot')).toBe('panel-reveal')
  fireEvent.click(reveal)
  expect($sidebarOpen.get()).toBe(true)
  expect(screen.queryByRole('button', { name: 'Show panel' })).toBeNull()
  $interfaceMode.set(previousMode)
})

it('goes back through the window history when there is an entry behind the page, else closes it', () => {
  const onBack = vi.fn()
  const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)

  act(() => $historyPosition.set({ index: 2, last: 2 }))
  render(<PageToolbar onBack={onBack} />)
  fireEvent.click(screen.getByRole('button', { name: 'Back' }))
  expect(back).toHaveBeenCalledOnce()
  expect(onBack).not.toHaveBeenCalled()
  expect(screen.getByRole('button', { name: 'Forward' }).hasAttribute('disabled')).toBe(true)

  act(() => $historyPosition.set({ index: 0, last: 1 }))
  fireEvent.click(screen.getByRole('button', { name: 'Back' }))
  expect(onBack).toHaveBeenCalledOnce()
  expect(screen.getByRole('button', { name: 'Forward' }).hasAttribute('disabled')).toBe(false)

  back.mockRestore()
  act(() => $historyPosition.set({ index: 0, last: 0 }))
})

describe('below wide', () => {
  afterEach(() => {
    act(() => {
      $shellSheet.set(null)
      $widthClass.set('wide')
    })
  })

  it('compact: drops Back and Forward, and the menu opens the rail alone on a page with no panel', () => {
    act(() => $widthClass.set('compact'))
    render(<PageToolbar onBack={vi.fn()} />)

    expect(screen.queryByRole('button', { name: 'Back' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Forward' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    expect($shellSheet.get()).toBe('nav')
    expect(screen.getByRole('button', { name: 'Show sidebar' }).getAttribute('aria-expanded')).toBe('true')
  })

  it("medium: keeps history, and the menu opens the page's panel as a sheet, whatever the docked state", () => {
    act(() => {
      $widthClass.set('medium')
      setSidebarOpen(true)
    })
    render(
      <>
        <PageToolbar onBack={vi.fn()} />
        <PanelPage />
      </>
    )

    expect(screen.getByRole('button', { name: 'Back' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    expect($shellSheet.get()).toBe('page')
    // The docked preference waits for the window to widen again.
    expect($sidebarOpen.get()).toBe(true)
  })

  it('a panel renders only while its sheet is open below wide', () => {
    act(() => $widthClass.set('compact'))
    const seen: boolean[] = []

    function Probe() {
      seen.push(usePageSidebarOpen())

      return null
    }

    render(<Probe />)
    act(() => $shellSheet.set('page'))
    expect(seen.at(0)).toBe(false)
    expect(seen.at(-1)).toBe(true)
  })
})

it('leaves Back and Forward to the browser when the app is hosted in one', () => {
  document.documentElement.dataset.hermesDesktopHost = 'browser'

  try {
    render(<PageToolbar onBack={vi.fn()} />)

    expect(screen.queryByRole('button', { name: 'Back' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Forward' })).toBeNull()
  } finally {
    delete document.documentElement.dataset.hermesDesktopHost
  }
})
