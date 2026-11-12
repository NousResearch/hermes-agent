import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $historyPosition } from '@/app/shell/history-position'
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
