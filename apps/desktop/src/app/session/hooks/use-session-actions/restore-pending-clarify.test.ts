import { beforeEach, describe, expect, it, vi } from 'vitest'

import { restorePendingClarifyToolCall } from '@/lib/chat-messages'
import type * as clarifyStore from '@/store/clarify'

const { setClarifyRequestMock, clearClarifyRequestMock } = vi.hoisted(() => ({
  clearClarifyRequestMock: vi.fn(),
  setClarifyRequestMock: vi.fn()
}))

vi.mock('@/store/clarify', async importOriginal => {
  const actual = await importOriginal<typeof clarifyStore>()

  return {
    ...actual,
    clearClarifyRequest: clearClarifyRequestMock.mockImplementation(actual.clearClarifyRequest),
    setClarifyRequest: setClarifyRequestMock.mockImplementation(actual.setClarifyRequest)
  }
})

import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'

import { observeClarifySnapshot, pendingClarifyToolPayload, restorePendingClarifyFromSnapshot, settleSupersededClarifyProjection } from './restore-pending-clarify'

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

  it('keeps a request received after activation began when an older pending snapshot arrives', () => {
    const current = { choices: null, multiSelect: false, question: 'New?', receivedAt: resumeStartedAt + 1,
      requestId: 'new', sessionId: 'sess' }

    $clarifyRequests.set({ sess: current })

    const state = restorePendingClarifyFromSnapshot(
      { pending_clarify: { request_id: 'old', question: 'Old?' } }, 'sess', resumeStartedAt, 'old'
    )

    expect(state.request).toBe(current)
    expect(setClarifyRequestMock).not.toHaveBeenCalled()
    expect(clearClarifyRequestMock).not.toHaveBeenCalled()
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

describe('superseded clarify projection', () => {
  it.each(['old', 'provider-tool-id'])('settles the old %s projection and preserves an already rendered newer call', oldToolId => {
    const previous = { requestId: 'old', sessionId: 'sess', question: 'Old?', choices: null, multiSelect: false }
    const current = { ...previous, requestId: 'new', question: 'New?' }
    const oldPart = { type: 'tool-call' as const, toolName: 'clarify', toolCallId: oldToolId, args: { question: 'Old?' }, argsText: '' }
    const newPart = { ...oldPart, toolCallId: 'new', args: { question: 'New?' } }
    const messages = [{ id: 'row', role: 'assistant' as const, pending: true, parts: [oldPart, newPart] }]
    const settled = settleSupersededClarifyProjection(messages, previous, current, true)
    expect(settled[0].parts[0]).toHaveProperty('result')
    expect(settled[0].parts[1]).toBe(newPart)
    expect(settleSupersededClarifyProjection(messages, previous, previous, true)).toBe(messages)
    expect(settleSupersededClarifyProjection(messages, previous, undefined, true)).toBe(messages)
    const onlyNew = [{ ...messages[0], parts: [newPart] }]
    expect(settleSupersededClarifyProjection(onlyNew, previous, current, true)[0]).toBe(onlyNew[0])
  })
})

it.each([true, false])('F2 preserves a changed request identity without receivedAt (pending snapshot: %s)', pending => {
  const current = { requestId: 'new', sessionId: 'sess', question: 'New?', choices: null, multiSelect: false }
  $clarifyRequests.set({ sess: current })

  const state = restorePendingClarifyFromSnapshot(
    pending ? { pending_clarify: { request_id: 'old', question: 'Old?' } } : {}, 'sess', resumeStartedAt, 'old'
  )

  expect(state.request).toBe(pending ? current : null)
  expect($clarifyRequests.get().sess).toBe(current)
})

it('F5 full settle and restore chain reuses the sole same-question provider row', () => {
  const previous = { requestId: 'old', sessionId: 'sess', question: 'Same?', choices: null, multiSelect: false }
  const current = { ...previous, requestId: 'new' }

  const messages = [{ id: 'row', role: 'assistant' as const, pending: true, parts: [
    { type: 'tool-call' as const, toolName: 'clarify', toolCallId: 'provider-1', args: { question: 'Same?' }, argsText: '' }
  ] }]

  const settled = settleSupersededClarifyProjection(messages, previous, current, true)
  const restored = restorePendingClarifyToolCall(settled, pendingClarifyToolPayload(current)).messages

  const open = restored.flatMap(message => message.parts).filter(part =>
    part.type === 'tool-call' && part.toolName === 'clarify' && part.result === undefined)

  expect(open).toHaveLength(1)
  expect(restored).toHaveLength(1)
})

it('F1 observation is session-scoped and released after the restore attempt', () => {
  $clarifyRequests.set({})
  const observation = observeClarifySnapshot()
  const request = { requestId: 'R', sessionId: 'origin', question: 'Question?', choices: null, multiSelect: false }
  setClarifyRequest(request)
  clearClarifyRequest('R', 'origin')
  const response = { pending_clarify: { request_id: 'R', question: 'Question?' } }
  expect(restorePendingClarifyFromSnapshot(response, 'origin', resumeStartedAt, undefined, observation).request).toBeNull()
  expect($clarifyRequests.get().origin).toBeUndefined()
  expect(restorePendingClarifyFromSnapshot(response, 'other', resumeStartedAt, undefined, observation).request?.sessionId).toBe('other')
  observation.dispose()
  clearClarifyRequest('R', 'other')
  expect(observation.changedSessions.has('other')).toBe(true)
  setClarifyRequest({ ...request, sessionId: 'after-dispose' })
  expect(observation.changedSessions.has('after-dispose')).toBe(false)
  const nextAttempt = observeClarifySnapshot()

  try {
    expect(restorePendingClarifyFromSnapshot(response, 'origin', resumeStartedAt, undefined, nextAttempt).request?.requestId).toBe('R')
  } finally {
    nextAttempt.dispose()
    clearClarifyRequest()
  }
})
