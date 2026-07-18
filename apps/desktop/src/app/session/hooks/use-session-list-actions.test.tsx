import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { listAllProfileSessions } from '@/hermes'
import { setActiveProfile } from '@/store/profile'
import { MESSAGING_SECTION_LIMIT, setMessagingSessions } from '@/store/session'

import { useSessionListActions } from './use-session-list-actions'

vi.mock('@/hermes', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    getCronJobs: vi.fn().mockResolvedValue([]),
    listAllProfileSessions: vi.fn().mockResolvedValue({
      limit: 100,
      offset: 0,
      sessions: [],
      total: 0
    })
  }
})

type SessionListActions = ReturnType<typeof useSessionListActions>

function Harness({ onReady }: { onReady: (actions: SessionListActions) => void }) {
  const actions = useSessionListActions({ profileScope: 'orchestrator' })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

describe('useSessionListActions messaging scope', () => {
  beforeEach(() => {
    setActiveProfile('orchestrator')
    setMessagingSessions([])
    vi.mocked(listAllProfileSessions).mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('loads the bounded messaging slice from the active server profile', async () => {
    let actions: SessionListActions | null = null

    render(<Harness onReady={value => (actions = value)} />)
    await waitFor(() => expect(actions).not.toBeNull())

    await act(async () => {
      await actions!.refreshMessagingSessions()
    })

    expect(listAllProfileSessions).toHaveBeenCalledWith(
      MESSAGING_SECTION_LIMIT,
      1,
      'exclude',
      'recent',
      'orchestrator',
      expect.objectContaining({ excludeSources: expect.any(Array) })
    )
  })

  it('coalesces overlapping polling and focus refreshes', async () => {
    let actions: SessionListActions | null = null
    let resolveRequest: ((value: Awaited<ReturnType<typeof listAllProfileSessions>>) => void) | null = null
    const pending = new Promise<Awaited<ReturnType<typeof listAllProfileSessions>>>(resolve => {
      resolveRequest = resolve
    })

    vi.mocked(listAllProfileSessions).mockReturnValueOnce(pending)
    render(<Harness onReady={value => (actions = value)} />)
    await waitFor(() => expect(actions).not.toBeNull())

    const first = actions!.refreshMessagingSessions()
    const second = actions!.refreshMessagingSessions()

    expect(first).toBe(second)
    expect(listAllProfileSessions).toHaveBeenCalledTimes(1)

    resolveRequest!({ limit: 100, offset: 0, sessions: [], total: 0 })
    await act(async () => {
      await Promise.all([first, second])
    })
  })
})
