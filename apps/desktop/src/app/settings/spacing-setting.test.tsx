// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { $portalDensity } from '@/store/portal-density'
import { $shellChoice } from '@/store/shell-variant'

import { AppearanceSettings } from './appearance-settings'

afterEach(() => {
  cleanup()
  $portalDensity.set('comfortable')
  $shellChoice.set(null)
})

const windowLayout = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <AppearanceSettings subpage="window-layout" />
    </QueryClientProvider>
  )

// Window & layout is where a search for the setting lands, in either shell.
it('offers Spacing on the redesign’s Window & layout page, and Compact sets it', () => {
  $shellChoice.set(true)
  windowLayout()

  expect(screen.getByText('Spacing')).toBeTruthy()
  expect(screen.queryByText('Detailed')).toBeNull()

  fireEvent.click(screen.getByText('Compact'))
  expect($portalDensity.get()).toBe('compact')
})

it('keeps classic’s three-way session list density there', () => {
  $shellChoice.set(false)
  windowLayout()

  expect(screen.getByText('Session List Density')).toBeTruthy()
  expect(screen.getByText('Detailed')).toBeTruthy()
})
