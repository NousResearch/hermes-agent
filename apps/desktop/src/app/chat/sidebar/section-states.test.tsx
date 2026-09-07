import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarLoadErrorState } from './section-states'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { retry: 'Retry' },
      sidebar: {
        // The trap this component exists to avoid: the empty-state copy.
        projectEmpty: 'No sessions yet',
        projectLoadFailed: 'Could not load sessions'
      }
    }
  })
}))

describe('SidebarLoadErrorState', () => {
  it('shows the load-failure copy, never the empty-state copy', () => {
    render(<SidebarLoadErrorState onRetry={() => {}} />)

    expect(screen.getByText('Could not load sessions')).toBeTruthy()
    expect(screen.queryByText('No sessions yet')).toBeNull()
  })

  it('fires onRetry when the retry button is clicked', () => {
    const onRetry = vi.fn()
    render(<SidebarLoadErrorState onRetry={onRetry} />)

    fireEvent.click(screen.getByRole('button', { name: /retry/i }))

    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})
