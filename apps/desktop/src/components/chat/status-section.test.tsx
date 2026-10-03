import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { StatusRow } from './status-row'
import { StatusSection } from './status-section'

vi.stubGlobal(
  'ResizeObserver',
  class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
)

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const section = (onDismiss: () => void) => (
  <StatusSection label="Background tasks">
    <StatusRow dismiss={{ label: 'Stop task', onDismiss }}>
      <span>Task title</span>
    </StatusRow>
  </StatusSection>
)

const expand = (x: number, y: number) =>
  fireEvent.click(screen.getByRole('button', { name: 'Background tasks' }), { clientX: x, clientY: y })

it('reads the collapse click that lands on the control revealed under it as a collapse', () => {
  const dismiss = vi.fn()
  render(section(dismiss))

  expand(48, 300)
  expect(screen.getByText('Task title')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Stop task' }), { clientX: 48, clientY: 300 })

  expect(dismiss).not.toHaveBeenCalled()
  expect(screen.queryByText('Task title')).toBeNull()
})

it('fires the control when the click is somewhere the expanding click was not', () => {
  const dismiss = vi.fn()
  render(section(dismiss))

  expand(48, 300)
  fireEvent.click(screen.getByRole('button', { name: 'Stop task' }), { clientX: 260, clientY: 340 })

  expect(dismiss).toHaveBeenCalledOnce()
})

it('drops the anchor once the reveal read window has passed', () => {
  const dismiss = vi.fn()
  const now = vi.spyOn(performance, 'now').mockReturnValue(1_000)
  render(section(dismiss))

  expand(48, 300)
  now.mockReturnValue(1_000 + 1_600)
  fireEvent.click(screen.getByRole('button', { name: 'Stop task' }), { clientX: 48, clientY: 300 })

  expect(dismiss).toHaveBeenCalledOnce()
})
