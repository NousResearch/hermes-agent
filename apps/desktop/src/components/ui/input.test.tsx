import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Input } from './input'

afterEach(cleanup)

describe('Input', () => {
  it('renders a bare input carrying the control chrome when unadorned', () => {
    const { getByRole } = render(<Input aria-label="plain" />)
    const el = getByRole('textbox')

    expect(el.tagName).toBe('INPUT')
    expect(el.className).toContain('desktop-input-chrome')
    // No group wrapper — the input is the top-level node.
    expect(el.parentElement?.getAttribute('data-slot')).not.toBe('input-group')
  })

  it('wraps the field in a chrome-owning group and renders the prefix when adorned', () => {
    const { getByRole, getByText } = render(<Input aria-label="amount" prefix="$" />)
    const el = getByRole('textbox')
    const group = el.closest('[data-slot="input-group"]')

    expect(group).not.toBeNull()
    // Chrome moves to the wrapper; the input itself goes transparent/borderless.
    expect(group?.className).toContain('desktop-input-chrome')
    expect(el.className).not.toContain('desktop-input-chrome')
    expect(el.className).toContain('bg-transparent')

    const prefix = getByText('$')
    expect(group?.contains(prefix)).toBe(true)
  })

  it('renders a trailing suffix inside the group', () => {
    const { getByText } = render(<Input aria-label="rate" suffix="%" />)
    const suffix = getByText('%')

    expect(suffix.closest('[data-slot="input-group"]')).not.toBeNull()
  })

  it('forwards value/onChange through the adorned field', () => {
    const onChange = vi.fn()
    const { getByRole } = render(<Input aria-label="amount" onChange={onChange} prefix="$" value="" />)

    fireEvent.change(getByRole('textbox'), { target: { value: '100' } })
    expect(onChange).toHaveBeenCalledTimes(1)
  })

  it('dims the group and disables the field when disabled', () => {
    const { getByRole } = render(<Input aria-label="amount" disabled prefix="$" />)
    const el = getByRole('textbox') as HTMLInputElement
    const group = el.closest('[data-slot="input-group"]')

    expect(el.disabled).toBe(true)
    expect(group?.className).toContain('opacity-50')
  })

  it('lets containerClassName override the wrapper width', () => {
    const { getByRole } = render(<Input aria-label="amount" containerClassName="w-20" prefix="$" />)
    const group = getByRole('textbox').closest('[data-slot="input-group"]')

    // twMerge resolves the control's default w-full down to the caller's width.
    expect(group?.className).toContain('w-20')
    expect(group?.className).not.toContain('w-full')
  })
})
