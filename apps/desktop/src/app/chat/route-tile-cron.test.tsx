import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { CRON_ROUTE } from '../routes'

import { RouteTilePane } from './route-tile'

const openStoredSession = vi.fn()

vi.mock('../open-session', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  openStoredSession: (...args: unknown[]) => openStoredSession(...args)
}))

vi.mock('../cron', () => ({
  CronView: ({ onOpenSession }: { onOpenSession?: (id: string, row?: SessionInfo) => void }) => (
    <button data-testid="cron-page" onClick={() => onOpenSession?.('run-1', { id: 'run-1' } as SessionInfo)}>
      cron page
    </button>
  )
}))

afterEach(() => {
  cleanup()
  openStoredSession.mockReset()
})

// The sidebar's "Open in split" on Scheduled jobs calls openRouteTile('/cron');
// the tile used to know only Artifacts/Messaging/Capabilities and showed
// "no page at /cron".
describe('RouteTilePane at /cron', () => {
  it('renders the Scheduled jobs page and opens a run through the stored-session door', async () => {
    render(
      <MemoryRouter>
        <RouteTilePane path={CRON_ROUTE} />
      </MemoryRouter>
    )

    const page = await screen.findByTestId('cron-page')

    expect(screen.queryByText(/no page at/)).toBeNull()

    act(() => page.click())

    expect(openStoredSession).toHaveBeenCalledWith('run-1', expect.any(Function), { id: 'run-1' })
  })
})
