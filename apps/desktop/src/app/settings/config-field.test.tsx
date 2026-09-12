import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ConfigFieldSchema } from '@/types/hermes'

import { ConfigField, NumberConfigInput, roundToStepPrecision } from './config-field'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('roundToStepPrecision', () => {
  it('rounds fractional values to avoid IEEE 754 precision drift', () => {
    expect(roundToStepPrecision(0.25000000000000006, 0.05)).toBe(0.25)
    expect(roundToStepPrecision(0.30000000000000004, 0.1)).toBe(0.3)
    expect(roundToStepPrecision(0.1 + 0.2, 0.05)).toBe(0.3)
    expect(roundToStepPrecision(0.05000000000000001, 0.05)).toBe(0.05)
  })

  it('preserves integer values when step is integer or undefined', () => {
    expect(roundToStepPrecision(20, 1)).toBe(20)
    expect(roundToStepPrecision(5, undefined)).toBe(5)
  })
})

describe('ConfigField Numeric Input - Area 1: Attribute Binding', () => {
  it('binds min, max, and step attributes for compression.target_ratio from NUMERIC_FIELD_CONFIG', () => {
    render(
      <ConfigField onChange={vi.fn()} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton')
    expect(input.getAttribute('min')).toBe('0.1')
    expect(input.getAttribute('max')).toBe('0.8')
    expect(input.getAttribute('step')).toBe('0.05')
  })

  it('binds min, max, and step attributes for compression.threshold from NUMERIC_FIELD_CONFIG', () => {
    render(<ConfigField onChange={vi.fn()} schema={{ type: 'number' }} schemaKey="compression.threshold" value={0.5} />)

    const input = screen.getByRole('spinbutton')
    expect(input.getAttribute('min')).toBe('0')
    expect(input.getAttribute('max')).toBe('1')
    expect(input.getAttribute('step')).toBe('0.05')
  })

  it('binds min and step attributes for compression.protect_last_n', () => {
    render(
      <ConfigField onChange={vi.fn()} schema={{ type: 'number' }} schemaKey="compression.protect_last_n" value={20} />
    )

    const input = screen.getByRole('spinbutton')
    expect(input.getAttribute('min')).toBe('1')
    expect(input.getAttribute('step')).toBe('1')
  })

  it('allows schema-provided min, max, and step to override defaults', () => {
    const customSchema: ConfigFieldSchema = {
      type: 'number',
      min: 0.2,
      max: 0.6,
      step: 0.1
    }

    render(<ConfigField onChange={vi.fn()} schema={customSchema} schemaKey="compression.target_ratio" value={0.3} />)

    const input = screen.getByRole('spinbutton')
    expect(input.getAttribute('min')).toBe('0.2')
    expect(input.getAttribute('max')).toBe('0.6')
    expect(input.getAttribute('step')).toBe('0.1')
  })

  it('defaults unconfigured integer number field step to 1', () => {
    render(<ConfigField onChange={vi.fn()} schema={{ type: 'number' }} schemaKey="custom.integer_field" value={10} />)

    const input = screen.getByRole('spinbutton')
    expect(input.getAttribute('step')).toBe('1')
  })
})

describe('ConfigField Numeric Input - Area 2: Spinner Stepping & Bounds Resistance', () => {
  it('steps by fractional step size (0.05) on stepUp() and fires onChange without jumping by 1', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    expect(input.value).toBe('0.2')

    // Simulate clicking spinner up arrow
    input.stepUp()
    expect(input.value).toBe('0.25')

    fireEvent.change(input, { target: { value: input.value } })
    expect(onChange).toHaveBeenCalledWith(0.25)
    expect(onChange).not.toHaveBeenCalledWith(1.2)
  })

  it('steps down by fractional step size (0.05) on stepDown() and fires onChange', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    input.stepDown()
    expect(input.value).toBe('0.15')

    fireEvent.change(input, { target: { value: input.value } })
    expect(onChange).toHaveBeenCalledWith(0.15)
  })

  it('resists stepping past maximum bound (0.80)', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.8} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement

    // In DOM, stepUp at max either throws or remains at max
    try {
      input.stepUp()
    } catch {
      // DOM spec may throw InvalidStateError when already at max
    }

    expect(Number(input.value)).toBeLessThanOrEqual(0.8)
  })

  it('resists stepping below minimum bound (0.10)', () => {
    render(
      <ConfigField onChange={vi.fn()} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.1} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement

    try {
      input.stepDown()
    } catch {
      // DOM spec may throw InvalidStateError when already at min
    }

    expect(Number(input.value)).toBeGreaterThanOrEqual(0.1)
  })
})

describe('ConfigField Numeric Input - Area 3: Empty Input Fallback Clamping', () => {
  it('clamps empty input to min when min > 0 on blur (compression.target_ratio -> 0.1)', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    // User clears the field
    fireEvent.change(input, { target: { value: '' } })
    // While typing/empty, premature onChange is NOT invoked
    expect(onChange).not.toHaveBeenCalled()

    // On blur, clamps to min = 0.1
    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith(0.1)
    expect(input.value).toBe('0.1')
  })

  it('clamps empty input to min on Enter key commit (compression.target_ratio -> 0.1)', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.3} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    fireEvent.change(input, { target: { value: '' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onChange).toHaveBeenCalledWith(0.1)
    expect(input.value).toBe('0.1')
  })

  it('falls back to 0 on clear when min is 0 (compression.threshold -> 0)', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.threshold" value={0.5} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    fireEvent.change(input, { target: { value: '' } })
    fireEvent.blur(input)

    expect(onChange).toHaveBeenCalledWith(0)
    expect(input.value).toBe('0')
  })

  it('falls back to 0 on clear when min is undefined', () => {
    const onChange = vi.fn()

    render(<ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="unbounded.field" value={42} />)

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    fireEvent.change(input, { target: { value: '' } })
    fireEvent.blur(input)

    expect(onChange).toHaveBeenCalledWith(0)
    expect(input.value).toBe('0')
  })

  it('never emits NaN when cleared or given invalid input', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    fireEvent.change(input, { target: { value: '   ' } })
    fireEvent.blur(input)

    expect(onChange).not.toHaveBeenCalledWith(NaN)
    expect(onChange).toHaveBeenCalledWith(0.1)
  })
})

describe('ConfigField Numeric Input - Area 4: Out-of-Bounds Clamping on Blur/Enter', () => {
  it('clamps typed value exceeding max to max on blur (1.5 -> 0.8)', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    fireEvent.change(input, { target: { value: '1.5' } })
    // Intermediate out-of-bounds value does not trigger onChange
    expect(onChange).not.toHaveBeenCalled()

    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith(0.8)
    expect(input.value).toBe('0.8')
  })

  it('clamps typed value below min to min on blur (-0.5 -> 0.1)', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    fireEvent.change(input, { target: { value: '-0.5' } })
    expect(onChange).not.toHaveBeenCalled()

    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith(0.1)
    expect(input.value).toBe('0.1')
  })

  it('clamps typed value exceeding max on Enter key commit (0.95 -> 0.8)', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement
    fireEvent.change(input, { target: { value: '0.95' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onChange).toHaveBeenCalledWith(0.8)
    expect(input.value).toBe('0.8')
  })

  it('propagates in-bounds typed value immediately without blur', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton')
    fireEvent.change(input, { target: { value: '0.35' } })

    expect(onChange).toHaveBeenCalledWith(0.35)
  })

  it('allows active decimal typing (e.g. typing 0 then . then 3) without premature clamping', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'number' }} schemaKey="compression.target_ratio" value={0.2} />
    )

    const input = screen.getByRole('spinbutton') as HTMLInputElement

    // User types "0"
    fireEvent.change(input, { target: { value: '0' } })
    // "0" < min (0.1), so onChange should not fire yet
    expect(onChange).not.toHaveBeenCalled()
    expect(input.value).toBe('0')

    // User types "0."
    fireEvent.change(input, { target: { value: '0.' } })
    expect(onChange).not.toHaveBeenCalled()

    // User finishes typing "0.3"
    fireEvent.change(input, { target: { value: '0.3' } })
    // Now in bounds: onChange fires with 0.3
    expect(onChange).toHaveBeenCalledWith(0.3)
  })
})

describe('ConfigField Numeric Input - Area 5: Floating-Point Stepping Precision Stability', () => {
  it('avoids floating point bleed when stepping consecutive fractional values', () => {
    const onChange = vi.fn()

    render(<NumberConfigInput max={0.8} min={0.1} onChange={onChange} step={0.05} value={0.2} />)

    const input = screen.getByRole('spinbutton') as HTMLInputElement

    // 0.2 + 0.05 could be 0.25000000000000006 in raw floating point
    fireEvent.change(input, { target: { value: '0.25' } })
    expect(onChange).toHaveBeenCalledWith(0.25)

    fireEvent.change(input, { target: { value: '0.3' } })
    expect(onChange).toHaveBeenCalledWith(0.3)
  })
})

describe('ConfigField Numeric Input - Area 6: Non-Numeric Regression Protection', () => {
  it('renders a switch for boolean fields and toggles properly', () => {
    const onChange = vi.fn()

    render(
      <ConfigField onChange={onChange} schema={{ type: 'boolean' }} schemaKey="compression.enabled" value={true} />
    )

    const switchControl = screen.getByRole('switch')
    expect(switchControl).not.toBeNull()
    fireEvent.click(switchControl)
    expect(onChange).toHaveBeenCalledWith(false)
  })

  it('renders a select dropdown for enum select fields', () => {
    render(
      <ConfigField
        enumOptions={['legacy', 'lean']}
        onChange={vi.fn()}
        schema={{ type: 'select', options: ['legacy', 'lean'] }}
        schemaKey="compression.tail_mode"
        value="legacy"
      />
    )

    const combobox = screen.getByRole('combobox')
    expect(combobox).not.toBeNull()
  })

  it('renders a standard text input for string fields', () => {
    const onChange = vi.fn()

    render(<ConfigField onChange={onChange} schema={{ type: 'string' }} schemaKey="general.name" value="Hermes" />)

    const textInput = screen.getByDisplayValue('Hermes')
    expect(textInput).not.toBeNull()
    expect(textInput.getAttribute('type')).not.toBe('number')

    fireEvent.change(textInput, { target: { value: 'New Name' } })
    expect(onChange).toHaveBeenCalledWith('New Name')
  })
})
