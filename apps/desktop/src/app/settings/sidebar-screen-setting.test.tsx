import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { AppearanceSettings } from './appearance-settings'

const key = 'hermes.desktop.sidebar.screenPortal.v1'

afterEach(() => {
  cleanup()
  localStorage.removeItem(key)
})

it('offers a default-on Screen sidebar switch that persists hiding and restoring it', () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <AppearanceSettings subpage="window-layout" />
    </QueryClientProvider>
  )

  const toggle = screen.getByRole('switch', { name: 'Screen in Sessions sidebar' })
  expect(toggle.getAttribute('aria-checked')).toBe('true')

  fireEvent.click(toggle)
  expect(toggle.getAttribute('aria-checked')).toBe('false')
  expect(localStorage.getItem(key)).toBe('false')

  fireEvent.click(toggle)
  expect(toggle.getAttribute('aria-checked')).toBe('true')
  expect(localStorage.getItem(key)).toBe('true')
})
