import { renderHook } from '@testing-library/react'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from './composer/scope'

const requestGatewayMock = vi.hoisted(() => vi.fn())

const { $activeSessionId } = await import('@/store/session')
const { $sessionTiles, setSessionTileDelegate } = await import('@/store/session-states')
const { useSessionTileActions } = await import('./session-tile-actions')

const RUNTIME_SESSION_ID = 'rt-tile-current'
const STORED_SESSION_ID = 'stored-tile-db'
const RECOVERED_SESSION_ID = 'rt-tile-recovered'

function renderTileActions() {
  return renderHook(() =>
    useSessionTileActions({
      requestGateway: requestGatewayMock,
      runtimeId: RUNTIME_SESSION_ID,
      scope: MAIN_COMPOSER_SCOPE,
      storedSessionId: STORED_SESSION_ID
    })
  )
}

// A tile's cancelRun/steerPrompt/reloadFromMessage each build their own
// requestGateway call directly instead of going through the shared
// submitPromptText pipeline (which already wraps its call in
// withSessionNotFoundResume) — see use-prompt-actions/index.test.tsx's
// "sleep/wake session recovery" suite for the same regression on the
// primary chat's own reloadFromMessage.
describe('useSessionTileActions sleep/wake session recovery', () => {
  beforeEach(() => {
    $activeSessionId.set('foreground-runtime')
    $sessionTiles.set([{ runtimeId: RUNTIME_SESSION_ID, storedSessionId: STORED_SESSION_ID }])
    setSessionTileDelegate({
      archiveSession: vi.fn(async () => undefined),
      branchSession: vi.fn(async () => undefined),
      deleteSession: vi.fn(async () => undefined),
      executeSlash: vi.fn(async () => undefined),
      interruptSession: vi.fn(async () => undefined),
      resumeTile: vi.fn(async () => RUNTIME_SESSION_ID),
      submitToSession: vi.fn(async () => undefined),
      updateSession: vi.fn((_runtimeId, updater) =>
        updater({
          attachedImages: [],
          busy: false,
          cwd: null,
          messages: [],
          model: null,
          streamId: null,
          storedSessionId: STORED_SESSION_ID
        } as never)
      )
    })
  })

  afterEach(() => {
    $activeSessionId.set(null)
    $sessionTiles.set([])
    requestGatewayMock.mockReset()
    vi.restoreAllMocks()
  })

  it('resumes the stored session and retries once when session.interrupt reports "session not found"', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let interruptAttempts = 0

    requestGatewayMock.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.interrupt') {
        interruptAttempts += 1

        if (interruptAttempts === 1) {
          throw new Error('session not found')
        }

        return {}
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID }
      }

      return {}
    })

    const { result } = renderTileActions()

    await act(async () => {
      await result.current.cancelRun()
    })

    // First interrupt (stale id) → session.resume (stored id) → retry interrupt (fresh id).
    expect(calls.map(c => c.method)).toEqual(['session.interrupt', 'session.resume', 'session.interrupt'])
    expect(calls[0]?.params).toEqual({ session_id: RUNTIME_SESSION_ID })
    expect(calls[1]?.params).toMatchObject({ session_id: STORED_SESSION_ID, source: 'desktop', omit_messages: true })
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID })
    expect($sessionTiles.get()[0]?.runtimeId).toBe(RECOVERED_SESSION_ID)
  })

  it('resumes the stored session and retries once when session.redirect (steer) reports "session not found"', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let redirectAttempts = 0

    requestGatewayMock.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.redirect') {
        redirectAttempts += 1

        if (redirectAttempts === 1) {
          throw new Error('session not found')
        }

        return { status: 'redirected' }
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID }
      }

      return {}
    })

    const { result } = renderTileActions()

    const ok = await act(async () => result.current.steerPrompt('actually use Postgres'))

    expect(ok).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['session.redirect', 'session.resume', 'session.redirect'])
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID, text: 'actually use Postgres' })
    expect($sessionTiles.get()[0]?.runtimeId).toBe(RECOVERED_SESSION_ID)
  })

  it('rebinds prompt.submit recovery to the tile without changing the foreground session', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    requestGatewayMock.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {}
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID }
      }

      return {}
    })

    const { result } = renderTileActions()

    await act(async () => {
      await expect(result.current.submitText('continue the bot chat')).resolves.toBe(true)
    })

    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[0]?.params).toMatchObject({ session_id: RUNTIME_SESSION_ID })
    expect(calls[1]?.params).toMatchObject({ session_id: STORED_SESSION_ID, source: 'desktop', omit_messages: true })
    expect(calls[2]?.params).toMatchObject({ session_id: RECOVERED_SESSION_ID })
    expect($sessionTiles.get()[0]?.runtimeId).toBe(RECOVERED_SESSION_ID)
    expect($activeSessionId.get()).toBe('foreground-runtime')
  })
})
