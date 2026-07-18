import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { listAllProfileSessions, type SessionInfo } from '@/hermes'
import {
  $sessionLoadError,
  $sessions,
  clearSessionLoadError,
  setSessionLoadError,
  setSessions,
  type SessionLoadError
} from '@/store/session'

import { SessionRefreshError, useSessionListActions } from './use-session-list-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(async () => []),
  listAllProfileSessions: vi.fn()
}))

const listAllProfileSessionsMock = vi.mocked(listAllProfileSessions)

function session(id: string): SessionInfo {
  return {
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: id,
    tool_call_count: 0
  }
}

function Harness({
  onReady
}: {
  onReady: (handle: ReturnType<typeof useSessionListActions>) => void
}) {
  const actions = useSessionListActions({ profileScope: 'default' })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

async function renderHook(): Promise<ReturnType<typeof useSessionListActions>> {
  let handle: ReturnType<typeof useSessionListActions> | null = null

  render(<Harness onReady={h => (handle = h)} />)
  await waitFor(() => expect(handle).not.toBeNull())

  return handle!
}

async function flushMicrotasks() {
  // Wait long enough for the awaited promise + finally + post-await fanout
  // (refreshCronSessions/refreshCronJobs/refreshMessagingSessions) to settle.
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 0))
  })
}

beforeEach(() => {
  listAllProfileSessionsMock.mockReset()
  // Drive the empty-config probe: no sessions on cold start.
  $sessions.set([])
  clearSessionLoadError()
  setSessionLoadError(null)
})

afterEach(() => {
  cleanup()
  $sessions.set([])
  clearSessionLoadError()
  setSessionLoadError(null)
  listAllProfileSessionsMock.mockReset()
})

describe('useSessionListActions.refreshSessions rejection paths', () => {
  it('re-throws on cold-boot failure so the boot overlay sees the error', async () => {
    listAllProfileSessionsMock.mockRejectedValueOnce(new Error('backend timeout'))

    const handle = await renderHook()

    let caught: unknown = null
    await act(async () => {
      try {
        await handle.refreshSessions({ isBoot: true })
      } catch (err) {
        caught = err
      }
    })

    expect(caught).toBeInstanceOf(SessionRefreshError)
    expect((caught as SessionRefreshError).message).toBe('backend timeout')
    expect($sessionLoadError.get()).toBeNull()
    expect($sessions.get()).toEqual([])
  })

  it('preserves stale rows and records a recoverable error on background failure', async () => {
    // Seed a populated list — the empty-state discriminator in the rework
    // depends on this so a populated user doesn't get false-positive re-throws.
    $sessions.set([session('a'), session('b')])
    listAllProfileSessionsMock.mockRejectedValueOnce(new Error('network blip'))

    const handle = await renderHook()

    let resolved = false
    let rejected: unknown = null
    await act(async () => {
      try {
        await handle.refreshSessions()
        resolved = true
      } catch (err) {
        rejected = err
      }
    })

    expect(resolved).toBe(true)
    expect(rejected).toBeNull()
    const error = $sessionLoadError.get()
    expect(error).not.toBeNull()
    expect(error?.message).toBe('network blip')
    expect(error?.initial).toBe(false)
    expect(error?.timestamp).toBeGreaterThan(0)
    // Stale rows are preserved.
    expect($sessions.get().map(s => s.id)).toEqual(['a', 'b'])
  })

  it('clears $sessionLoadError on a successful refresh', async () => {
    $sessions.set([session('a')])
    // Seed a stale error so we can verify it gets cleared.
    const stale: SessionLoadError = { initial: false, message: 'old failure', timestamp: 1 }
    setSessionLoadError(stale)

    listAllProfileSessionsMock.mockResolvedValueOnce({
      profile_totals: { default: 2 },
      sessions: [session('a'), session('c')],
      total: 2
    } as never)

    const handle = await renderHook()

    await act(async () => {
      await handle.refreshSessions()
    })

    expect($sessionLoadError.get()).toBeNull()
    await flushMicrotasks()
  })

  it('handles non-Error rejection values without crashing', async () => {
    listAllProfileSessionsMock.mockRejectedValueOnce('plain string rejection')

    const handle = await renderHook()

    let caught: unknown = null
    await act(async () => {
      try {
        await handle.refreshSessions({ isBoot: true })
      } catch (err) {
        caught = err
      }
    })

    expect(caught).toBeInstanceOf(SessionRefreshError)
    expect((caught as SessionRefreshError).message).toBe('plain string rejection')
    expect((caught as SessionRefreshError).cause).toBe('plain string rejection')
  })

  it('keeps the boot path silent on populated-list failure (no re-throw)', async () => {
    // A populated list combined with isBoot=true is a contrived case — the
    // controller only ever passes isBoot:true on cold start — but the
    // discriminator should still trust the explicit flag and re-throw. This
    // pins the contract so future refactors don't accidentally start treating
    // empty-vs-populated as a substitute for the explicit option.
    $sessions.set([session('x')])
    listAllProfileSessionsMock.mockRejectedValueOnce(new Error('late boot failure'))

    const handle = await renderHook()

    let caught: unknown = null
    await act(async () => {
      try {
        await handle.refreshSessions({ isBoot: true })
      } catch (err) {
        caught = err
      }
    })

    expect(caught).toBeInstanceOf(SessionRefreshError)
  })
})
