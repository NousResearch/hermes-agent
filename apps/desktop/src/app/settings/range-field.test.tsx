import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { formatTimeframe, RangeField } from './range-field'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('formatTimeframe', () => {
  it('formats minutes under an hour plainly', () => {
    expect(formatTimeframe(10, 'minutes')).toBe('10 minutes')
    expect(formatTimeframe(1, 'minutes')).toBe('1 minute')
  })

  it('promotes to hours with a minute breakdown', () => {
    expect(formatTimeframe(60, 'minutes')).toBe('1 hour (60 minutes)')
    expect(formatTimeframe(90, 'minutes')).toBe('1.5 hours (90 minutes)')
  })

  it('handles seconds and hours units', () => {
    expect(formatTimeframe(45, 'seconds')).toBe('45 seconds')
    expect(formatTimeframe(90, 'seconds')).toBe('1.5 minutes (90 seconds)')
    expect(formatTimeframe(48, 'hours')).toBe('2 days (48 hours)')
  })

  it('falls back to unit-less numbers for unknown units', () => {
    expect(formatTimeframe(7, undefined)).toBe('7')
  })
})

describe('RangeField', () => {
  it('renders a slider and a text box seeded with the value', () => {
    render(<RangeField max={120} min={1} onChange={vi.fn()} step={1} unit="minutes" value={10} />)

    const slider = screen.getByRole('slider') as HTMLInputElement
    expect(slider.min).toBe('1')
    expect(slider.max).toBe('120')
    expect(slider.value).toBe('10')

    const input = screen.getByRole('textbox', { name: 'value' }) as HTMLInputElement
    expect(input.value).toBe('10')
  })

  it('shows a live timeframe preview for the configured value', () => {
    render(<RangeField max={120} min={1} onChange={vi.fn()} step={1} unit="minutes" value={10} />)
    expect(screen.getByText('10 minutes')).toBeTruthy()
  })

  it('writes clamped values through onChange when the slider moves', () => {
    const onChange = vi.fn()
    render(<RangeField max={120} min={1} onChange={onChange} step={1} unit="minutes" value={10} />)

    fireEvent.change(screen.getByRole('slider'), { target: { value: '45' } })
    expect(onChange).toHaveBeenCalledWith(45)
  })

  it('clamps typed values into [min, max]', () => {
    const onChange = vi.fn()
    render(<RangeField max={120} min={1} onChange={onChange} step={1} unit="minutes" value={10} />)

    fireEvent.change(screen.getByRole('textbox', { name: 'value' }), { target: { value: '500' } })
    expect(onChange).toHaveBeenCalledWith(120)

    onChange.mockClear()
    fireEvent.change(screen.getByRole('textbox', { name: 'value' }), { target: { value: '-3' } })
    expect(onChange).toHaveBeenCalledWith(1)
  })

  it('ignores non-numeric text-box input', () => {
    const onChange = vi.fn()
    render(<RangeField max={120} min={1} onChange={onChange} step={1} unit="minutes" value={10} />)

    fireEvent.change(screen.getByRole('textbox', { name: 'value' }), { target: { value: 'abc' } })
    expect(onChange).not.toHaveBeenCalled()
  })

  it('resets to the shipped default when the reset button is clicked', () => {
    const onChange = vi.fn()
    render(
      <RangeField
        defaultValue={10}
        max={120}
        min={1}
        onChange={onChange}
        step={1}
        unit="minutes"
        value={25}
      />
    )

    const reset = screen.getByRole('button', { name: 'reset to default' }) as HTMLButtonElement
    expect(reset.disabled).toBe(false)
    fireEvent.click(reset)
    expect(onChange).toHaveBeenCalledWith(10)
  })

  it('disables the reset button when the value already matches the default', () => {
    render(
      <RangeField
        defaultValue={10}
        max={120}
        min={1}
        onChange={vi.fn()}
        step={1}
        unit="minutes"
        value={10}
      />
    )

    const reset = screen.getByRole('button', { name: 'reset to default' }) as HTMLButtonElement
    expect(reset.disabled).toBe(true)
  })

  it('shows the default in the preview line when one is provided', () => {
    render(
      <RangeField
        defaultValue={10}
        max={120}
        min={1}
        onChange={vi.fn()}
        step={1}
        unit="minutes"
        value={25}
      />
    )

    expect(screen.getByText(/default 10 minutes/)).toBeTruthy()
  })
})
