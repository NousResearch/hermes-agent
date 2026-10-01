import { renderHook } from '@testing-library/react'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { MAIN_COMPOSER_SCOPE } from './composer/scope'

const requestGatewayMock = vi.hoisted(() => vi.fn())

const { $activeSessionId, $sessions, setSessions } = await import('@/store/session')

const { $sessionStates, $sessionTiles, clearAllSessionStates, publishSessionState, setSessionTileDelegate } =
  await import('@/store/session-states')

const { listTileSessionRow, useSessionTileActions } = await import('./session-tile-actions')

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

describe('session tile optimistic owner metadata', () => {
  afterEach(() => {
    $sessions.set([])
    $sessionTiles.set([])
  })

  it('never lists a bots-workspace tile — hidden relationship chats stay off the Sessions list (#113273)', () => {
    expect(
      listTileSessionRow({
        preview: 'hello from the bot chat',
        runtimeId: 'rt-bot-chat',
        sessions: [],
        storedSessionId: 'stored-bot-chat',
        workspaceMode: 'bots'
      })
    ).toBe(false)

    expect($sessions.get()).toEqual([])
  })

  it('keeps the tile source on its first optimistic sidebar row', () => {
    const storedSessionId = 'stored-tile-owner-metadata'
    const ownerRoute = { connectionId: 'source-a', profile: 'default' }
    $sessionTiles.set([{ ownerRoute, storedSessionId }])

    expect(
      listTileSessionRow({
        cwd: '/remote/worktree',
        model: 'model-a',
        preview: 'hello from the tile',
        runtimeId: 'rt-tile-owner-metadata',
        sessions: [],
        storedSessionId
      })
    ).toBe(true)

    expect($sessions.get()[0]).toMatchObject({
      connection_id: 'source-a',
      id: storedSessionId,
      profile: 'default'
    })
  })
})

// A tile's cancelRun/steerPrompt/reloadFromMessage each build their own
// requestGateway call directly instead of going through the shared
// submitPromptText pipeline (which already wraps its call in
// withSessionNotFoundResume) — see use-prompt-actions/index.test.tsx's
// "sleep/wake session recovery" suite for the same regression on the
// primary chat's own reloadFromMessage.
describe('useSessionTileActions sleep/wake session recovery', () => {
  beforeEach(() => {
    $activeSessionId.set('foreground-runtime')
    setSessions([])
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
    setSessions([])
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

describe('useSessionTileActions reloadFromMessage failed-submit rollback (#95745)', () => {
  const seed = [
    { id: 'u1', parts: [textPart('first')], role: 'user' as const, timestamp: 0 },
    { id: 'a1', parts: [textPart('reply')], role: 'assistant' as const, timestamp: 1 },
    { id: 'u2', parts: [textPart('later')], role: 'user' as const, timestamp: 2 },
    { id: 'a2', parts: [textPart('later reply')], role: 'assistant' as const, timestamp: 3 }
  ]

  beforeEach(() => {
    $activeSessionId.set('foreground-runtime')
    setSessions([])
    $sessionTiles.set([{ runtimeId: RUNTIME_SESSION_ID, storedSessionId: STORED_SESSION_ID }])
    publishSessionState(RUNTIME_SESSION_ID, createClientSessionState(STORED_SESSION_ID, seed as never))
    setSessionTileDelegate({
      archiveSession: vi.fn(async () => undefined),
      branchSession: vi.fn(async () => undefined),
      deleteSession: vi.fn(async () => undefined),
      executeSlash: vi.fn(async () => undefined),
      interruptSession: vi.fn(async () => undefined),
      resumeTile: vi.fn(async () => RUNTIME_SESSION_ID),
      submitToSession: vi.fn(async () => undefined),
      updateSession: vi.fn((_runtimeId, updater) => {
        const current = $sessionStates.get()[RUNTIME_SESSION_ID]

        if (!current) {
          return undefined
        }

        const next = updater(current)

        publishSessionState(RUNTIME_SESSION_ID, next)

        return next
      })
    })
  })

  afterEach(() => {
    $activeSessionId.set(null)
    setSessions([])
    $sessionTiles.set([])
    clearAllSessionStates()
    requestGatewayMock.mockReset()
    vi.restoreAllMocks()
  })

  it('re-hydrates the tile transcript and retries once when restore hits a stale durable target', async () => {
    // #107593 (tile surface): the tile's cached rowIds go stale after resume
    // drift or a session.branch remap; Restore must recover like Edit does —
    // refresh the stored transcript, recompute the plan, retry once — instead
    // of rolling back and rethrowing as a 4018 dead-end.
    let submitAttempts = 0
    const submitCalls: { params?: Record<string, unknown> }[] = []

    requestGatewayMock.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submitAttempts += 1
        submitCalls.push({ params })

        if (submitAttempts === 1) {
          throw new Error('target user message is no longer in session history')
        }

        return {}
      }

      return {}
    })

    const resumeTile = vi.fn(async () => {
      // The refreshed transcript: same shapes, fresh row ids after the remap.
      publishSessionState(
        RUNTIME_SESSION_ID,
        createClientSessionState(
          STORED_SESSION_ID,
          [
            { id: 'u1-fresh', parts: [textPart('first')], role: 'user' as const, rowId: 501, timestamp: 0 },
            { id: 'a1-fresh', parts: [textPart('reply')], role: 'assistant' as const, rowId: 502, timestamp: 1 },
            { id: 'u2-fresh', parts: [textPart('later')], role: 'user' as const, rowId: 503, timestamp: 2 }
          ] as never
        )
      )

      return RUNTIME_SESSION_ID
    })

    // Re-wire the delegate so resumeTile is the transcript refresher.
    setSessionTileDelegate({
      archiveSession: vi.fn(async () => undefined),
      branchSession: vi.fn(async () => undefined),
      deleteSession: vi.fn(async () => undefined),
      executeSlash: vi.fn(async () => undefined),
      interruptSession: vi.fn(async () => undefined),
      resumeTile,
      submitToSession: vi.fn(async () => undefined),
      updateSession: vi.fn((_runtimeId, updater) => {
        const current = $sessionStates.get()[RUNTIME_SESSION_ID]

        if (!current) {
          return undefined
        }

        const next = updater(current)

        publishSessionState(RUNTIME_SESSION_ID, next)

        return next
      })
    })

    const { result } = renderTileActions()

    await act(async () => {
      await result.current.restoreToMessage('u1')
    })

    expect(resumeTile).toHaveBeenCalledWith(STORED_SESSION_ID, { refreshTranscript: true })
    expect(submitAttempts).toBe(2)
    expect(submitCalls[1]?.params).toMatchObject({
      truncate_before_row_id: 501
    })
  })

  it('restores the full tile transcript when regenerate is rejected', async () => {
    requestGatewayMock.mockImplementation(async (method: string) => {
      if (method === 'prompt.submit') {
        throw new Error('target user message is no longer in session history')
      }

      return {}
    })

    const { result } = renderTileActions()

    await act(async () => {
      await result.current.reloadFromMessage('u1')
    })

    const rolledBack = $sessionStates.get()[RUNTIME_SESSION_ID]?.messages

    expect(rolledBack?.map(m => m.id)).toEqual(['u1', 'a1', 'u2', 'a2'])
    expect(rolledBack?.some(m => m.hidden)).toBe(false)
    expect($sessionStates.get()[RUNTIME_SESSION_ID]?.busy).toBe(false)
  })
})
