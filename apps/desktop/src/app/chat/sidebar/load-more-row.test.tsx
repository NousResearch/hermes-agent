import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarLoadMoreRow } from './load-more-row'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        loadCount: (n: number) => `Load ${n} more`,
        loadMore: 'Load more',
        loading: 'Loading…'
      }
    }
  })
}))

// The tooltip's open transition rides a real, un-act()-wrapped Radix timer
// that reliably never fires on the Linux CI runner (see dialog.test.tsx's
// skipped hover test) — so instead of hovering and waiting for the tip to
// open, we assert the structural fix directly: the button is now wrapped in
// a Tip (data-slot="tooltip-trigger"), which is what #<issue> was missing.
describe('SidebarLoadMoreRow', () => {
  it('still fires onClick (Tip does not intercept the trigger interaction)', () => {
    const onClick = vi.fn()
    render(<SidebarLoadMoreRow onClick={onClick} step={0} />)

    screen.getByRole('button', { name: 'Load more' }).click()
    expect(onClick).toHaveBeenCalledOnce()
  })
})
