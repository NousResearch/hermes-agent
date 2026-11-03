import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ContextBreakdownModule from '@/app/shell/hooks/use-context-breakdown'
import { invalidateContextBreakdown } from '@/app/shell/hooks/use-context-breakdown'
import { createClientSessionState } from '@/lib/chat-runtime'
import type * as SessionModule from '@/store/session'
import { setCurrentUsage } from '@/store/session'
import type * as SessionStatesModule from '@/store/session-states'
import { requestForOwnedSession } from '@/store/session-states'

import { compressSession, ownedSessionRequest, SESSION_COMPRESS_TIMEOUT_MS } from './compress-session'

// Partial mocks: these stores are imported all over the renderer, and a bare
// factory would strip exports that unrelated modules read at init.
vi.mock('@/app/shell/hooks/use-context-breakdown', async importOriginal => ({
  ...(await importOriginal<typeof ContextBreakdownModule>()),
  invalidateContextBreakdown: vi.fn()
}))
vi.mock('@/store/session', async importOriginal => ({
  ...(await importOriginal<typeof SessionModule>()),
  setCurrentUsage: vi.fn(),
  setSessions: vi.fn()
}))
vi.mock('@/store/session-states', async importOriginal => ({
  ...(await importOriginal<typeof SessionStatesModule>()),
  requestForOwnedSession: vi.fn()
}))

const sessionId = 'runtime-a'
const storedSessionId = 'stored-a'

function deps(overrides: Partial<Parameters<typeof compressSession>[0]> = {}) {
  return {
    isActive: () => true,
    request: vi.fn(),
    updateSessionState: vi.fn((_id: string, updater: (state: ReturnType<typeof createClientSessionState>) => unknown) =>
      updater(createClientSessionState())
    ),
    ...overrides
  }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('ownedSessionRequest', () => {
  it('dispatches to the backend that OWNS the captured session', async () => {
    const ambient = vi.fn().mockResolvedValue({ ok: true })
    const request = ownedSessionRequest('runtime-a', ambient)

    await request('session.compress', { session_id: 'runtime-a' }, 1_234)

    expect(requestForOwnedSession).toHaveBeenCalledWith(
      'runtime-a',
      ambient,
      'session.compress',
      { session_id: 'runtime-a' },
      1_234
    )
  })
})

describe('compressSession', () => {
  it('compresses with the long RPC budget and rewrites the transcript it gets back', async () => {
    const request = vi.fn().mockResolvedValue({
      messages: [{ content: 'summary', role: 'system' }],
      summary: { headline: 'compressed 12 → 2 messages', token_line: '128k → 9k tokens' },
      usage: { context_used: 9_000 }
    })

    const d = deps({ request })

    const outcome = await compressSession(d, { sessionId, storedSessionId })

    expect(request).toHaveBeenCalledWith('session.compress', { session_id: sessionId }, SESSION_COMPRESS_TIMEOUT_MS)
    expect(outcome).toMatchObject({
      aborted: false,
      kind: 'done',
      message: 'compressed 12 → 2 messages\n128k → 9k tokens'
    })

    // The rewrite lands on the session that was compressed, not on whatever is
    // in the foreground now.
    expect(d.updateSessionState).toHaveBeenCalledWith(sessionId, expect.any(Function), storedSessionId)
    expect(invalidateContextBreakdown).toHaveBeenCalledWith(sessionId)
    expect(setCurrentUsage).toHaveBeenCalled()
  })

  it('does not repaint the shared usage readouts when the chat is no longer on screen', async () => {
    const request = vi.fn().mockResolvedValue({ removed: 4, usage: { context_used: 4_000 } })
    const d = deps({ isActive: () => false, request })

    const outcome = await compressSession(d, { sessionId, storedSessionId })

    expect(outcome).toMatchObject({ kind: 'done', message: 'compressed 4 messages' })
    expect(invalidateContextBreakdown).toHaveBeenCalledWith(sessionId)
    expect(setCurrentUsage).not.toHaveBeenCalled()
  })

  it('reports a deferred compression without touching the transcript', async () => {
    const request = vi.fn().mockResolvedValue({ message: 'still compressing', status: 'pending' })
    const d = deps({ request })

    const outcome = await compressSession(d, { sessionId, storedSessionId })

    expect(outcome).toEqual({ kind: 'deferred', message: 'still compressing', sessionId })
    expect(d.updateSessionState).not.toHaveBeenCalled()
    expect(invalidateContextBreakdown).not.toHaveBeenCalled()
  })

  it('reports a compression locked elsewhere as a skip, not a success', async () => {
    const request = vi.fn().mockResolvedValue({
      compressed: false,
      lock_held: true,
      message: 'another process holds the compression lock for this session'
    })

    const d = deps({ request })

    const outcome = await compressSession(d, { sessionId, storedSessionId })

    expect(outcome).toMatchObject({
      kind: 'skipped',
      message: 'another process holds the compression lock for this session'
    })
    expect(d.updateSessionState).not.toHaveBeenCalled()
    expect(invalidateContextBreakdown).not.toHaveBeenCalled()
  })

  it('reports a canonical no-op reply as a skip, not a compression', async () => {
    const request = vi.fn().mockResolvedValue({
      status: 'noop',
      summary: { headline: 'No changes from compression', noop: true }
    })

    const d = deps({ request })

    const outcome = await compressSession(d, { sessionId, storedSessionId })

    expect(outcome).toMatchObject({ kind: 'skipped', message: 'No changes from compression' })
    expect(d.updateSessionState).not.toHaveBeenCalled()
  })

  it('never treats a read-only preview as a compression', async () => {
    const request = vi.fn().mockResolvedValue({
      host_ack: { output: 'would compress 24 → 4 messages' },
      status: 'preview'
    })

    const d = deps({ request })

    const outcome = await compressSession(d, { sessionId, storedSessionId })

    expect(outcome).toMatchObject({ kind: 'skipped', message: 'would compress 24 → 4 messages' })
    expect(d.updateSessionState).not.toHaveBeenCalled()
  })

  it('counts the canonical owner receipt as a completed compression', async () => {
    // What the canonical `session.mutate` compress actually answers: the
    // committed receipt (counts + summary), with no legacy `status` field and
    // no `removed`. Reading only the legacy shape classified this as a no-op.
    const request = vi.fn().mockResolvedValue({
      execution_generation: 12,
      message_count: 4,
      operation: 'compress',
      previous_target_session_id: 'stored-a',
      revision: 7,
      session_id: sessionId,
      summary: { headline: 'Compressed: 24 → 4 messages' },
      target_session_id: 'stored-a'
    })

    const d = deps({ request })

    const outcome = await compressSession(d, { sessionId, storedSessionId })

    expect(outcome).toMatchObject({
      aborted: false,
      kind: 'done',
      message: 'Compressed: 24 → 4 messages'
    })
    expect(invalidateContextBreakdown).toHaveBeenCalledWith(sessionId)
  })

  it('marks an aborted summary as a failure', async () => {
    const request = vi.fn().mockResolvedValue({
      status: 'aborted',
      summary: { aborted: true, headline: 'compression aborted' }
    })

    const outcome = await compressSession(deps({ request }), { sessionId, storedSessionId })

    expect(outcome).toMatchObject({ aborted: true, kind: 'done', message: 'compression aborted' })
  })

  it('passes the focus topic through to the RPC', async () => {
    const request = vi.fn().mockResolvedValue({ removed: 1 })

    await compressSession(deps({ request }), { focusTopic: 'the auth bug', sessionId, storedSessionId })

    expect(request).toHaveBeenCalledWith(
      'session.compress',
      { focus_topic: 'the auth bug', session_id: sessionId },
      SESSION_COMPRESS_TIMEOUT_MS
    )
  })
})
