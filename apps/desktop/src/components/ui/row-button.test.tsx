import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { RowButton } from './row-button'

afterEach(cleanup)

describe('RowButton', () => {
  it('renders a real <button> with type=button so it never submits an enclosing form', () => {
    const { getByText } = render(<RowButton>Row</RowButton>)
    const el = getByText('Row')

    expect(el.tagName).toBe('BUTTON')
    expect(el.getAttribute('type')).toBe('button')
  })
})
