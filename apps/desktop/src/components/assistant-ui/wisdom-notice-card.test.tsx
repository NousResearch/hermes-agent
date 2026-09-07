// @vitest-environment jsdom
import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'

const getWisdomInstallations = vi.fn()
const acknowledgeWisdomNotifications = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  acknowledgeWisdomNotifications,
  getWisdomInstallations
}))

vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

const notification = {
  category: 'new_skill' as const,
  event_id: 'event-1',
  kind: 'new',
  skill_id: 'skill-1',
  skill_name: 'team-runbook',
  source_event_ids: ['event-1'],
  version: 1
}

beforeEach(() => {
  vi.resetAllMocks()
  getWisdomInstallations.mockResolvedValue({
    installations: [],
    notifications: [notification]
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('WisdomNoticeCard', () => {
  it('preserves the last confirmed organization notice through a transient poll failure', async () => {
    const setIntervalSpy = vi.spyOn(window, 'setInterval')
    const { WisdomNoticeCard } = await import('./wisdom-notice-card')
    render(<WisdomNoticeCard profile="research" />)

    expect(await screen.findByText('team-runbook v1 was shared with your collective.')).toBeTruthy()
    getWisdomInstallations.mockRejectedValueOnce(new Error('Gateway unavailable'))
    const poll = setIntervalSpy.mock.calls.find(([, delay]) => delay === 30_000)?.[0]
    expect(poll).toBeTypeOf('function')
    await act(async () => {
      await (poll as () => Promise<void>)()
    })

    expect(screen.getByText('team-runbook v1 was shared with your collective.')).toBeTruthy()
  })

  it('clears a notice only after a successful empty projection', async () => {
    const setIntervalSpy = vi.spyOn(window, 'setInterval')
    const { WisdomNoticeCard } = await import('./wisdom-notice-card')
    render(<WisdomNoticeCard profile="research" />)

    expect(await screen.findByText('team-runbook v1 was shared with your collective.')).toBeTruthy()
    getWisdomInstallations.mockResolvedValueOnce({ installations: [], notifications: [] })
    const poll = setIntervalSpy.mock.calls.find(([, delay]) => delay === 30_000)?.[0]
    await act(async () => {
      await (poll as () => Promise<void>)()
    })

    await waitFor(() => {
      expect(screen.queryByText('team-runbook v1 was shared with your collective.')).toBeNull()
    })
  })
})
