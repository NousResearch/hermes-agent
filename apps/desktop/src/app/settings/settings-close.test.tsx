import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { RootTooltipProvider } from '@/components/ui/tooltip'
import { $shellChoice } from '@/store/shell-variant'

import { $widthClass } from '../redesign/breakpoints'

import { SettingsView } from './index'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  $widthClass.set('wide')
  delete document.documentElement.dataset.hermesDesktopHost
})

function renderSettings(onClose: () => void) {
  return render(
    <MemoryRouter initialEntries={['/settings?tab=about']}>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <RootTooltipProvider>
          <SettingsView onClose={onClose} />
        </RootTooltipProvider>
      </QueryClientProvider>
    </MemoryRouter>
  )
}

const closeButton = () => screen.queryByRole('button', { name: 'Close settings' })

it('closes the redesigned Settings from its band in a browser tab, which has no Back there', () => {
  $shellChoice.set(true)
  document.documentElement.dataset.hermesDesktopHost = 'browser'
  const close = vi.fn()
  renderSettings(close)

  expect(screen.queryByRole('button', { name: 'Back' })).toBeNull()
  fireEvent.click(closeButton()!)
  expect(close).toHaveBeenCalledOnce()
})

it('leaves closing to the band’s Back in the app, until the window is too narrow for one', () => {
  $shellChoice.set(true)
  const close = vi.fn()
  renderSettings(close)

  expect(screen.getByRole('button', { name: 'Back' })).toBeTruthy()
  expect(closeButton()).toBeNull()

  act(() => $widthClass.set('compact'))
  expect(screen.queryByRole('button', { name: 'Back' })).toBeNull()
  fireEvent.click(closeButton()!)
  expect(close).toHaveBeenCalledOnce()
})

it('keeps classic’s one close button', () => {
  $shellChoice.set(false)
  document.documentElement.dataset.hermesDesktopHost = 'browser'
  renderSettings(vi.fn())

  expect(screen.getAllByRole('button', { name: 'Close settings' })).toHaveLength(1)
})
