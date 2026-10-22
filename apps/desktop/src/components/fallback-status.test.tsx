import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { FallbackStatus } from './fallback-status'

describe('FallbackStatus', () => {
  it('shows active route, ordered chain, reason, and retry time', () => {
    render(
      <FallbackStatus
        status={{
          active: { model: 'backup-model', provider: 'backup-provider' },
          chain: [
            { model: 'first-model', provider: 'first-provider' },
            { model: 'second-model', provider: 'second-provider' }
          ],
          cooldown_until: 1_800_000_000,
          reason: 'Primary provider rate limited'
        }}
      />
    )

    expect(screen.getByText('Using backup-provider/backup-model')).toBeTruthy()
    expect(screen.getByText('first-provider/first-model')).toBeTruthy()
    expect(screen.getByText('second-provider/second-model')).toBeTruthy()
    expect(screen.getByText('Primary provider rate limited')).toBeTruthy()
    expect(screen.getByText(/^Next retry:/)).toBeTruthy()
  })

  it('renders nothing for legacy status responses without fallback data', () => {
    const { container } = render(<FallbackStatus />)
    expect(container.firstChild).toBeNull()
  })
})
