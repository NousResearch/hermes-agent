import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { listAllProfileSessions, type SessionInfo } from '@/hermes'
import {
  $messagingPlatformTotals,
  $messagingSessions,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setMessagingTruncated
} from '@/store/session'

import { useSessionListActions } from './use-session-list-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(),
  listAllProfileSessions: vi.fn()
}))

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function session(id: string, profile: string): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile,
    source: 'telegram',
    started_at: 1,
    title: null,
    tool_call_count: 0
  }
}

describe('useSessionListActions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setMessagingSessions([])
    setMessagingPlatformTotals({ telegram: 3 })
    setMessagingTruncated(false)
  })

  afterEach(() => {
    cleanup()
    setMessagingSessions([])
    setMessagingPlatformTotals({})
    setMessagingTruncated(false)
  })

  it('ignores a stale scope callback before it can invalidate the current messaging request', async () => {
    const alpha = deferred<Awaited<ReturnType<typeof listAllProfileSessions>>>()
    const beta = deferred<Awaited<ReturnType<typeof listAllProfileSessions>>>()
    const alphaRow = session('alpha-row', 'alpha')
    const betaRow = session('beta-row', 'beta')

    vi.mocked(listAllProfileSessions).mockImplementation((_limit, _min, _archived, _order, profile) =>
      profile === 'alpha' ? alpha.promise : beta.promise
    )

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'alpha' }
    })

    const staleAlphaRefresh = result.current.refreshMessagingSessions

    rerender({ profileScope: 'beta' })

    const betaRefresh = result.current.refreshMessagingSessions()
    const staleAlphaRequest = staleAlphaRefresh()

    await act(async () => {
      beta.resolve({ limit: 100, offset: 0, sessions: [betaRow], total: 1 })
      await betaRefresh
      alpha.resolve({ limit: 100, offset: 0, sessions: [alphaRow], total: 1 })
      await staleAlphaRequest
    })

    expect(vi.mocked(listAllProfileSessions).mock.calls.map(call => call[4])).toEqual(['beta'])
    expect($messagingSessions.get()).toEqual([betaRow])
    expect($messagingPlatformTotals.get()).toEqual({})
  })
})
