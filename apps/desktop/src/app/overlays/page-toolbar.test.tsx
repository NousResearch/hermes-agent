import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

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
