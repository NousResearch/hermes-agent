import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RowButton } from './row-button'

afterEach(cleanup)

describe('RowButton', () => {
  it('renders a real <button> with type=button and the row-button slot', () => {
    const { getByText } = render(<RowButton>Row</RowButton>)
    const el = getByText('Row')

    expect(el.tagName).toBe('BUTTON')
    expect(el.getAttribute('type')).toBe('button')
    expect(el.getAttribute('data-slot')).toBe('row-button')
  })

  it('imposes no styling of its own — only the caller class is applied', () => {
    const onClick = vi.fn()

    const { getByText } = render(
      <RowButton className="custom-row" onClick={onClick}>
        Hit
      </RowButton>
    )

    const el = getByText('Hit')

    expect(el.className).toBe('custom-row')
    el.click()
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('allows the native button type to be overridden', () => {
    const { getByText } = render(<RowButton type="submit">Go</RowButton>)

    expect(getByText('Go').getAttribute('type')).toBe('submit')
  })
})
