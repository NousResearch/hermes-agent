import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Input } from './input'

afterEach(cleanup)

describe('Input', () => {
  it('forwards value/onChange through the adorned field', () => {
    const onChange = vi.fn()
    const { getByRole } = render(<Input aria-label="amount" onChange={onChange} prefix="$" value="" />)

    fireEvent.change(getByRole('textbox'), { target: { value: '100' } })
    expect(onChange).toHaveBeenCalledTimes(1)
  })

  it('disables the field when disabled', () => {
    const { getByRole } = render(<Input aria-label="amount" disabled prefix="$" />)
    const el = getByRole('textbox') as HTMLInputElement

    expect(el.disabled).toBe(true)
  })
})
