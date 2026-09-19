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

  it('hands back the card the request handler already parked for a replayed open clarify request', () => {
    $clarifyRequests.set({
      'sess-1': {
        choices: null,
        multiSelect: false,
        question: 'Proceed?',
        receivedAt: resumeStartedAt + 5,
        requestId: 'rid1',
        sessionId: 'sess-1'
      }
    })

    const state = restorePendingClarifyFromSnapshot(
      { open_requests: [{ id: 'rid1', method: 'clarify', params: { question: 'Proceed?' } }] },
      'sess-1',
      resumeStartedAt
    )

    expect(state.authoritativeAbsent).toBe(false)
    expect(state.request?.requestId).toBe('rid1')
    expect(clearClarifyRequestMock).not.toHaveBeenCalled()
  })

  it('restores the parked card from a v0.21.2 pending_clarify snapshot (no open_requests)', () => {
    // Backwards compat: an old backend reports a still-open clarify as
    // `pending_clarify` instead of listing it in `open_requests`.
    $clarifyRequests.set({
      'sess-legacy': {
        choices: null,
        multiSelect: false,
        question: 'Proceed?',
        receivedAt: resumeStartedAt + 5,
        requestId: 'rid-legacy',
        sessionId: 'sess-legacy'
      }
    })

    const state = restorePendingClarifyFromSnapshot(
      { pending_clarify: { choices: [], question: 'Proceed?', request_id: 'rid-legacy' } },
      'sess-legacy',
      resumeStartedAt
    )

    expect(state.authoritativeAbsent).toBe(false)
    expect(state.request?.requestId).toBe('rid-legacy')
    expect(clearClarifyRequestMock).not.toHaveBeenCalled()
  })

  it('reports an open request the handler declined (no card parked) without inventing one', () => {
    const state = restorePendingClarifyFromSnapshot(
      { open_requests: [{ id: 'rid4', method: 'clarify', params: {} }] },
      'sess-4',
      resumeStartedAt
    )

    expect(state.authoritativeAbsent).toBe(false)
    expect(state.request).toBeNull()
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
