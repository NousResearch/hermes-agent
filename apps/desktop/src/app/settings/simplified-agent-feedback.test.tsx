import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { bindDesktopMetrics, resetDesktopMetricsForTests, setDesktopMetricsGate } from '@/store/desktop-metrics'
import { $portalPreferred, $shellChoice } from '@/store/shell-variant'
import { ThemeProvider } from '@/themes/context'

import { AppearanceSettings } from './appearance-settings'

const request = vi.fn(async () => ({ ok: true }) as never)
const initialScope = $apiRequestScope.get()

function renderSettings(subpage = 'general') {
  const view = render(
    <MemoryRouter initialEntries={['/settings?tab=appearance']}>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ThemeProvider>
          <AppearanceSettings subpage={subpage} />
        </ThemeProvider>
      </QueryClientProvider>
    </MemoryRouter>
  )

  if (subpage === 'general') {
    fireEvent.click(view.container.querySelector('details > summary')!)
  }

  return view
}

beforeEach(() => {
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  resetDesktopMetricsForTests()
  localStorage.clear()
  request.mockClear()
  $shellChoice.set(true)
  $apiRequestScope.set({ connectionId: 'sample', profile: 'default' })
  bindDesktopMetrics(request, 'sample|default')
})

afterEach(() => {
  cleanup()
  resetDesktopMetricsForTests()
  $shellChoice.set(false)
  $apiRequestScope.set(initialScope)
})

it('restores the original sidebar immediately and accepts one optional reason inline', async () => {
  setDesktopMetricsGate('on')
  renderSettings()
  fireEvent.click(screen.getByRole('switch', { name: 'Redesigned interface' }))
  expect($portalPreferred.get()).toBe(false)
  expect(screen.queryByRole('dialog')).toBeNull()
  const feedback = screen.getByRole('combobox', { name: 'Why switch back? (optional)' })
  fireEvent.keyDown(feedback, { key: 'ArrowDown' })
  fireEvent.click(await screen.findByRole('option', { name: 'I prefer the classic interface' }))
  expect(screen.getByRole('status').textContent).toContain('Thanks')
  expect(screen.queryByRole('combobox', { name: 'Why switch back? (optional)' })).toBeNull()
  await vi.waitFor(() =>
    expect(request).toHaveBeenCalledWith('shared_metrics.desktop_dislike', {
      signal: 'sidebar_disabled_reason',
      target: 'prefer_classic'
    })
  )
})

it('does not ask for feedback or report a disable when usage sharing is off', async () => {
  setDesktopMetricsGate('off')
  renderSettings()
  fireEvent.click(screen.getByRole('switch', { name: 'Redesigned interface' }))
  expect($portalPreferred.get()).toBe(false)
  expect(screen.queryByRole('combobox', { name: 'Why switch back? (optional)' })).toBeNull()
  await act(async () => {})
  expect(request).not.toHaveBeenCalled()
})

it('lets users ignore feedback and removes it when they re-enable or revoke consent', () => {
  setDesktopMetricsGate('on')
  renderSettings('window-layout')
  const toggle = screen.getByRole('switch', { name: 'Redesigned interface' })
  fireEvent.click(toggle)
  expect(screen.getByRole('combobox', { name: 'Why switch back? (optional)' })).toBeTruthy()
  fireEvent.click(screen.getByRole('switch', { name: 'Redesigned interface' }))
  expect(screen.queryByRole('combobox', { name: 'Why switch back? (optional)' })).toBeNull()
  fireEvent.click(screen.getByRole('switch', { name: 'Redesigned interface' }))
  act(() => setDesktopMetricsGate('off'))
  expect(screen.queryByRole('combobox', { name: 'Why switch back? (optional)' })).toBeNull()
})
