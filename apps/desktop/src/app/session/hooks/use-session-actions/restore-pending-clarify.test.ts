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

import { type ChatMessage, restorePendingClarifyToolCall } from '@/lib/chat-messages'
import { $clarifyRequests, type ClarifyRequest } from '@/store/clarify'

import {
  pendingClarifyToolPayload,
  restorePendingClarifyFromSnapshot,
  transcriptUnderEarlyClarify
} from './restore-pending-clarify'

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
        questions: [{ choices: null, multiSelect: false, qid: 'q0', question: 'Proceed?' }],
        receivedAt: resumeStartedAt + 5,
        requestId: 'rid1',
        sessionId: 'sess-1'
      }
    })

    const state = restorePendingClarifyFromSnapshot(
      {
        open_requests: [{ id: 'rid1', method: 'clarify', params: { questions: [{ qid: 'q0', question: 'Proceed?' }] } }]
      },
      'sess-1',
      resumeStartedAt
    )

    expect(state.authoritativeAbsent).toBe(false)
    expect(state.request?.requestId).toBe('rid1')
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
        questions: [{ choices: null, multiSelect: false, qid: 'q0', question: 'Old' }],
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
        questions: [{ choices: null, multiSelect: false, qid: 'q0', question: 'Newer' }],
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

describe('transcriptUnderEarlyClarify', () => {
  const request: ClarifyRequest = {
    questions: [
      { choices: ['Keep it', 'Change it'], multiSelect: false, qid: 'q0', question: 'Which way should step 1 go?' }
    ],
    receivedAt: resumeStartedAt + 5,
    requestId: 'srq-1',
    sessionId: 'sess-1'
  }

  // What the request handler leaves on a runtime that had no transcript yet:
  // the open clarify re-delivered while session.resume was in flight.
  const earlyProjection = () => restorePendingClarifyToolCall([], pendingClarifyToolPayload(request)).messages

  const stored: ChatMessage[] = [
    { id: 'u-old', parts: [{ text: 'Read every file first.', type: 'text' }], role: 'user', rowId: 10 },
    { id: 'a-old', parts: [{ text: 'Done reading.', type: 'text' }], role: 'assistant', rowId: 11 },
    { id: 'u-now', parts: [{ text: 'So what next?', type: 'text' }], role: 'user', rowId: 12 },
    { id: 'a-now', parts: [{ text: 'A few decisions only you can make.', type: 'text' }], role: 'assistant', rowId: 13 }
  ]

  const clarifyParts = (messages: ChatMessage[]) =>
    messages.flatMap(message =>
      message.parts.flatMap(part =>
        part.type === 'tool-call' && part.toolName === 'clarify' ? [{ message, part }] : []
      )
    )

  it('paints the stored transcript under a projection-only runtime and re-arms the clarify on it', () => {
    const seeded = earlyProjection()

    expect(seeded).toHaveLength(1)

    const rebuilt = transcriptUnderEarlyClarify(seeded, stored, request)

    expect(rebuilt).not.toBeNull()
    expect(rebuilt!.messages.slice(0, 3).map(message => message.id)).toEqual(['u-old', 'a-old', 'u-now'])

    // One live card, tied to the request, after the prompt that asked for it.
    const cards = clarifyParts(rebuilt!.messages)

    expect(cards).toHaveLength(1)
    expect(cards[0]!.part).toMatchObject({ clarifyRequestId: 'srq-1' })
    expect(cards[0]!.part.result).toBeUndefined()
    expect(rebuilt!.messages.indexOf(cards[0]!.message)).toBeGreaterThan(2)
    expect(rebuilt!.streamId).toBe(cards[0]!.message.id)
  })

  it('leaves a runtime that already carries a transcript alone', () => {
    const seeded = restorePendingClarifyToolCall(stored, pendingClarifyToolPayload(request)).messages

    expect(transcriptUnderEarlyClarify(seeded, stored, request)).toBeNull()
  })

  it('keeps the projection when nothing is stored or no clarify is parked', () => {
    expect(transcriptUnderEarlyClarify(earlyProjection(), [], request)).toBeNull()
    expect(transcriptUnderEarlyClarify(earlyProjection(), stored, undefined)).toBeNull()
    expect(transcriptUnderEarlyClarify([], stored, request)).toBeNull()
  })
})
