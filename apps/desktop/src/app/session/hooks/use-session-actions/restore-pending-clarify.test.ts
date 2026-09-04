import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as clarifyStore from '@/store/clarify'

const { setClarifyRequestMock, clearClarifyRequestMock } = vi.hoisted(() => ({
  clearClarifyRequestMock: vi.fn(),
  setClarifyRequestMock: vi.fn()
}))

vi.mock('@/store/clarify', async importOriginal => {
  const actual = await importOriginal<typeof clarifyStore>()

  return {
    ...actual,
    clearClarifyRequest: clearClarifyRequestMock,
    setClarifyRequest: setClarifyRequestMock
  }
})

import { $clarifyRequests } from '@/store/clarify'

import { pendingClarifyToolPayload, restorePendingClarifyFromSnapshot } from './restore-pending-clarify'

const resumeStartedAt = 1_700_000_000

describe('restorePendingClarifyFromSnapshot', () => {
  beforeEach(() => {
    clearClarifyRequestMock.mockClear()
    setClarifyRequestMock.mockClear()
    $clarifyRequests.set({})
  })

  it('restores a batch clarify snapshot (questions, no top-level question)', () => {
    const state = restorePendingClarifyFromSnapshot(
      {
        pending_clarify: {
          request_id: 'rid1',
          questions: [
            { choices: ['Yes', 'No'], multi_select: false, qid: 'q0', question: 'Proceed?' },
            { qid: 'q1', question: 'Which region?' }
          ]
        }
      },
      'sess-1',
      resumeStartedAt
    )

    expect(state.request).not.toBeNull()
    expect(setClarifyRequestMock).toHaveBeenCalledWith(
      expect.objectContaining({
        multiSelect: false,
        question: '',
        requestId: 'rid1',
        sessionId: 'sess-1',
        questions: [
          { choices: ['Yes', 'No'], multiSelect: false, qid: 'q0', question: 'Proceed?' },
          { multiSelect: false, qid: 'q1', question: 'Which region?', choices: null }
        ]
      })
    )
  })

  it('carries server-locked answers into the replayed batch card', () => {
    restorePendingClarifyFromSnapshot(
      {
        pending_clarify: {
          answers: { q0: 'Yes', junk: 42 },
          request_id: 'rid2',
          questions: [{ qid: 'q0', question: 'Proceed?' }]
        }
      },
      'sess-2',
      resumeStartedAt
    )

    expect(setClarifyRequestMock).toHaveBeenCalledWith(
      expect.objectContaining({ lockedAnswers: { q0: 'Yes' }, requestId: 'rid2' })
    )
  })

  it('still restores the single-question form', () => {
    const state = restorePendingClarifyFromSnapshot(
      {
        pending_clarify: {
          choices: ['A', 'B'],
          multi_select: true,
          question: 'Pick one',
          request_id: 'rid3'
        }
      },
      'sess-3',
      resumeStartedAt
    )

    expect(state.request).not.toBeNull()
    expect(setClarifyRequestMock).toHaveBeenCalledWith(
      expect.objectContaining({
        choices: ['A', 'B'],
        multiSelect: true,
        question: 'Pick one',
        requestId: 'rid3'
      })
    )
  })

  it('rejects a payload with neither form (no request restored)', () => {
    const state = restorePendingClarifyFromSnapshot(
      { pending_clarify: { request_id: 'rid4' } },
      'sess-4',
      resumeStartedAt
    )

    expect(state.request).toBeNull()
    expect(setClarifyRequestMock).not.toHaveBeenCalled()
  })

  it('rejects a payload with no request id', () => {
    const state = restorePendingClarifyFromSnapshot(
      { pending_clarify: { question: 'Orphaned prompt' } },
      'sess-5',
      resumeStartedAt
    )

    expect(state.request).toBeNull()
    expect(state.authoritativeAbsent).toBe(true)
    expect(setClarifyRequestMock).not.toHaveBeenCalled()
  })

  it('clears a stale local request when the snapshot has none, and leaves a newer in-flight one', () => {
    $clarifyRequests.set({
      'sess-6': {
        choices: null,
        multiSelect: false,
        question: 'Old',
        receivedAt: resumeStartedAt - 10,
        requestId: 'old-rid',
        sessionId: 'sess-6'
      }
    })

    const cleared = restorePendingClarifyFromSnapshot({}, 'sess-6', resumeStartedAt, 'old-rid')

    expect(cleared.authoritativeAbsent).toBe(true)
    expect(cleared.cleared?.requestId).toBe('old-rid')
    expect(clearClarifyRequestMock).toHaveBeenCalledWith('old-rid', 'sess-6')

    $clarifyRequests.set({
      'sess-6': {
        choices: null,
        multiSelect: false,
        question: 'Newer',
        receivedAt: resumeStartedAt + 1,
        requestId: 'new-rid',
        sessionId: 'sess-6'
      }
    })

    const kept = restorePendingClarifyFromSnapshot({}, 'sess-6', resumeStartedAt, 'old-rid')

    expect(kept.cleared).toBeNull()
    expect(kept.request).toBeNull()
    expect(clearClarifyRequestMock).toHaveBeenCalledTimes(1)
  })
})

describe('pendingClarifyToolPayload', () => {
  it('mirrors the batch wire shape for in-place re-arm', () => {
    expect(
      pendingClarifyToolPayload({
        choices: null,
        multiSelect: false,
        question: '',
        questions: [{ choices: ['Yes', 'No'], multiSelect: false, qid: 'q0', question: 'Proceed?' }],
        requestId: 'rid',
        sessionId: 'sess'
      })
    ).toEqual({
      args: {
        questions: [{ choices: ['Yes', 'No'], question: 'Proceed?' }]
      },
      tool_id: 'rid'
    })
  })
})
