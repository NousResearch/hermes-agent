import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SearchField } from './search-field'

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { ui: { search: { clear: 'Clear search' } } } }),
}))

afterEach(cleanup)

describe('SearchField', () => {
  it('grows the input to keep the clear action at the row edge', () => {
    const { getByRole } = render(
      <SearchField onChange={vi.fn()} placeholder="Search" value="query" />
    )

    const input = getByRole('textbox')
    expect(input.className).toContain('flex-1')
    expect(input.className).toContain('min-w-0')
  })
})
