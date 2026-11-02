import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { bindDesktopMetrics, resetDesktopMetricsForTests, setDesktopMetricsGate } from '@/store/desktop-metrics'
import { $agentSidebarChoice, $simplifiedAgentMode } from '@/store/simplified-agent-mode'

import { InterfaceModeSettings } from './interface-mode-settings'

const request = vi.fn(async () => ({ ok: true }) as never)

beforeEach(() => {
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  resetDesktopMetricsForTests()
  localStorage.clear()
  request.mockClear()
  $agentSidebarChoice.set(true)
  bindDesktopMetrics(request, 'sample|default')
})

afterEach(() => {
  cleanup()
  resetDesktopMetricsForTests()
  $agentSidebarChoice.set(false)
})

it('restores the original sidebar immediately and accepts one optional reason inline', async () => {
  setDesktopMetricsGate('on')
  render(<InterfaceModeSettings />)
  fireEvent.click(screen.getByRole('switch', { name: 'Simplified agent mode' }))
  expect($simplifiedAgentMode.get()).toBe(false)
  expect(screen.queryByRole('dialog')).toBeNull()
  const feedback = screen.getByRole('combobox', { name: 'Why switch back? (optional)' })
  fireEvent.keyDown(feedback, { key: 'ArrowDown' })
  fireEvent.click(await screen.findByRole('option', { name: 'I prefer the original sidebar' }))
  expect(screen.getByRole('status').textContent).toContain('Thanks')
  expect(screen.queryByRole('combobox')).toBeNull()
  await vi.waitFor(() =>
    expect(request).toHaveBeenCalledWith('shared_metrics.desktop_dislike', {
      signal: 'sidebar_disabled_reason',
      target: 'prefer_classic'
    })
  )
})

it('does not ask for feedback or report a disable when usage sharing is off', async () => {
  setDesktopMetricsGate('off')
  render(<InterfaceModeSettings />)
  fireEvent.click(screen.getByRole('switch', { name: 'Simplified agent mode' }))
  expect($simplifiedAgentMode.get()).toBe(false)
  expect(screen.queryByRole('combobox')).toBeNull()
  await act(async () => {})
  expect(request).not.toHaveBeenCalled()
})

it('lets users ignore feedback and removes it when they re-enable or revoke consent', () => {
  setDesktopMetricsGate('on')
  render(<InterfaceModeSettings />)
  const toggle = screen.getByRole('switch', { name: 'Simplified agent mode' })
  fireEvent.click(toggle)
  expect(screen.getByRole('combobox')).toBeTruthy()
  fireEvent.click(toggle)
  expect(screen.queryByRole('combobox')).toBeNull()
  fireEvent.click(toggle)
  act(() => setDesktopMetricsGate('off'))
  expect(screen.queryByRole('combobox')).toBeNull()
})
