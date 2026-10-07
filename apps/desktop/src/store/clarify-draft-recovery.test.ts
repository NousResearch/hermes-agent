import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { recoverClarifyDrafts } from '@/app/session/clarify-draft-recovery'
import {
  type ClarifyRequest,
  clearClarifyRequest,
  clearSettledClarifyRequest,
  setClarifyRequest,
  stageClarifyAnswer
} from '@/store/clarify'
import { clearSessionDraft, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { resetServerRequestsForTests } from '@/store/server-requests'

// `requestComposerInsert` dispatches a window CustomEvent through the focus
// bus; spy on it so assertions never depend on a mounted composer.
vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerInsert: vi.fn()
}))

import { requestComposerInsert } from '@/app/chat/composer/focus'

const request = (over: Partial<ClarifyRequest> = {}): ClarifyRequest => ({
  questions: [{ choices: ['yes', 'no'], multiSelect: false, qid: 'q1', question: 'Proceed?' }],
  requestId: 'req-1',
  sessionId: 'session-a',
  ...over
})

describe('clarify staged answers survive card remounts (#58783)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    clearClarifyRequest(undefined, undefined)
    clearSessionDraft('session-a')
    resetServerRequestsForTests()
    vi.useRealTimers()
  })

  it('keeps a staged answer across a re-park of the same request', () => {
    setClarifyRequest(request())
    stageClarifyAnswer('req-1', 'session-a', 'q1', { choices: [], draft: 'maybe tomorrow' })

    // Reconnect replay / resume re-delivers the SAME request: the card
    // remounts under a new assistant-row identity, and the parked staging —
    // not component state — is what the new card reads.
    setClarifyRequest(request())

    const parked = clearClarifyRequest('req-1', 'session-a')[0]
    expect(parked?.stagedAnswers?.q1?.draft).toBe('maybe tomorrow')
  })

  it('drops staging when a DIFFERENT request parks (new question, fresh card)', () => {
    setClarifyRequest(request())
    stageClarifyAnswer('req-1', 'session-a', 'q1', { choices: [], draft: 'stale typing' })

    setClarifyRequest(request({ requestId: 'req-2' }))

    const parked = clearClarifyRequest('req-2', 'session-a')[0]
    expect(parked?.stagedAnswers).toBeUndefined()
  })
})

describe('clarify draft recovery on turn unwind (#58783)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    clearClarifyRequest(undefined, undefined)
    clearSessionDraft('session-a')
    clearSessionDraft('session-b')
    resetServerRequestsForTests()
    vi.useRealTimers()
  })

  it('salvages a staged answer into the session draft on expiry (request.cancel), never auto-sending', () => {
    setClarifyRequest(request())
    stageClarifyAnswer('req-1', 'session-a', 'q1', { choices: [], draft: 'spent a minute on this' })

    // The server-side timeout path clears and recovers in one beat.
    const activeRef = { current: 'session-a' }
    recoverClarifyDrafts(clearClarifyRequest('req-1', 'session-a'), activeRef)

    expect(takeSessionDraft('session-a').text).toContain('spent a minute on this')
    expect(takeSessionDraft('session-a').text).toContain('Proceed?')
    // Never auto-sent: the recovery only stashes; delivery to the composer is
    // an insert offer through the focus bus, deferred past this tick.
    expect(requestComposerInsert).not.toHaveBeenCalled()
  })

  it('appends to — never replaces — text already in the session draft', () => {
    stashSessionDraft('session-a', 'half-typed follow-up', [])

    setClarifyRequest(request())
    stageClarifyAnswer('req-1', 'session-a', 'q1', { choices: [], draft: 'my answer' })
    recoverClarifyDrafts(clearClarifyRequest('req-1', 'session-a'), { current: 'session-a' })

    expect(takeSessionDraft('session-a').text).toContain('half-typed follow-up')
    expect(takeSessionDraft('session-a').text).toContain('my answer')
  })

  it('offers the recovered answer to the ACTIVE composer only for the session on screen', () => {
    vi.mocked(requestComposerInsert).mockClear()
    setClarifyRequest(request())
    stageClarifyAnswer('req-1', 'session-a', 'q1', { choices: [], draft: 'visible answer' })
    recoverClarifyDrafts(clearClarifyRequest('req-1', 'session-a'), { current: 'session-a' })

    vi.advanceTimersByTime(150)
    // The deferred insert targets the main composer for an on-screen session.
    expect(requestComposerInsert).toHaveBeenCalledWith(expect.stringContaining('visible answer'), {
      mode: 'block',
      target: 'main'
    })

    // A BACKGROUND session's recovery stashes but never touches the visible
    // composer: no second insert, no cross-session composer painting.
    vi.mocked(requestComposerInsert).mockClear()
    setClarifyRequest(request({ requestId: 'req-2', sessionId: 'session-b' }))
    stageClarifyAnswer('req-2', 'session-b', 'q1', { choices: [], draft: 'background answer' })
    recoverClarifyDrafts(clearClarifyRequest('req-2', 'session-b'), { current: 'session-a' })

    vi.advanceTimersByTime(150)
    expect(takeSessionDraft('session-b').text).toContain('background answer')
    expect(requestComposerInsert).not.toHaveBeenCalled()
  })

  it('recovers a picked choice, not just typed text', () => {
    setClarifyRequest(request())
    stageClarifyAnswer('req-1', 'session-a', 'q1', { choices: ['yes'], draft: '' })
    recoverClarifyDrafts(clearClarifyRequest('req-1', 'session-a'), { current: null })

    expect(takeSessionDraft('session-a').text).toContain('yes')
  })

  it('turn-end settle clear (message.complete path) recovers mid-answer staging too', () => {
    setClarifyRequest(request())
    stageClarifyAnswer('req-1', 'session-a', 'q1', { choices: [], draft: 'was answering when it unwound' })

    recoverClarifyDrafts(clearSettledClarifyRequest('session-a'), { current: 'session-a' })

    expect(takeSessionDraft('session-a').text).toContain('was answering when it unwound')
  })

  it('clearing with nothing staged recovers nothing (no empty-draft noise)', () => {
    setClarifyRequest(request())
    stashSessionDraft('session-a', '', [])

    recoverClarifyDrafts(clearClarifyRequest('req-1', 'session-a'), { current: 'session-a' })

    expect(takeSessionDraft('session-a').text).toBe('')
  })
})
