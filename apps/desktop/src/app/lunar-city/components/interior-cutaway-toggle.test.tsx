// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { InteriorCutawayToggle } from './interior-cutaway-toggle'

it('requires a selected building and exposes a reversible local prototype toggle', () => {
  const onChange = vi.fn()
  const { rerender } = render(<InteriorCutawayToggle enabled={false} onChange={onChange} />)
  const button = screen.getByRole('button', { name: 'See inside' }) as HTMLButtonElement
  expect(button.disabled).toBe(true)
  fireEvent.click(button)
  expect(onChange).not.toHaveBeenCalled()
  rerender(<InteriorCutawayToggle buildingId="library" enabled={false} onChange={onChange} title="Library" />)
  fireEvent.click(screen.getByRole('button', { name: 'See inside · Library' }))
  expect(onChange).toHaveBeenLastCalledWith(true)
  rerender(<InteriorCutawayToggle buildingId="library" enabled onChange={onChange} title="Library" />)
  expect(screen.getByRole('button', { pressed: true }).textContent).toContain('Show exterior')
  expect(screen.getByText(/Prototype interior/)).toBeTruthy()
  fireEvent.click(screen.getByRole('button'))
  expect(onChange).toHaveBeenLastCalledWith(false)
})
