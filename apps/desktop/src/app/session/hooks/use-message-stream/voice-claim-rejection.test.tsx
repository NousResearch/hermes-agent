import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { turnHasReply, withNoReplyNotice } from '@/store/session-no-reply'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

const SUBMISSION = 'voice-live:["call","delegation"]'
const REASON = 'Hermes did not run this spoken request because it could not be matched exactly to the saved call transcript.'

// The owner's frame for a refused recorded GPT-Live claim: a queued admission that
// never ran, so it names the owner epoch and admission but no execution generation.
const refusal = (sessionId: string, surface: Record<string, unknown> = {}) => ({
  type: 'message.complete',
  session_id: sessionId,
  authority_epoch: 2,
  admission_id: 'voice-head',
  payload: {
    admission_id: 'voice-head',
    content: '',
    error: REASON,
    error_surface: { code: 'voice_claim_unverified', layer: 'gateway', message: REASON, retryable: false, ...surface },
    outcome: 'rejected',
    status: 'error',
    submission_id: SUBMISSION,
    text: ''
  }
})

function awaitingVoiceTurn(sessionId: string) {
  const stream = renderMessageStream(sessionId)
  stream.states.set(sessionId, {
    ...stream.state(sessionId),
    awaitingResponse: true,
    busy: true,
    messages: [{ id: 'spoken', parts: [{ text: 'Ask Hermes to reply.', type: 'text' }], role: 'user' }],
    turnLive: false,
    turnStartedAt: Date.now(),
    voiceClaimSubmissionId: SUBMISSION
  } as never)

  return stream
}

it('shows why a refused voice claim did not run, without Retry', () => {
  const stream = awaitingVoiceTurn('voice-chat')
  act(() => stream.handleEvent(refusal('voice-chat') as never))

  const state = stream.state('voice-chat')
  const card = state.messages.at(-1)
  expect(state.busy).toBe(false)
  expect(state.awaitingResponse).toBe(false)
  expect(card?.role).toBe('assistant')
  expect(card?.error).toBe(REASON)
  expect(card?.errorSurface).toMatchObject({ code: 'voice_claim_unverified', retryable: false })
  // The silence watchdog's generic, retryable "ended without a reply" card stays away.
  expect(turnHasReply(state.messages)).toBe(true)
})

it('never lets a refused voice claim settle a turn that is running', () => {
  const stream = renderMessageStream('busy-chat')
  act(() => stream.handleEvent({ type: 'message.start', session_id: 'busy-chat', authority_epoch: 2, execution_generation: 5 } as never))
  const before = stream.state('busy-chat')
  act(() => stream.handleEvent(refusal('busy-chat') as never))
  expect(stream.state('busy-chat').busy).toBe(true)
  expect(stream.state('busy-chat').messages).toEqual(before.messages)
})

it('keeps fencing other admission-only completions', () => {
  const stream = awaitingVoiceTurn('other-chat')
  act(() => stream.handleEvent(refusal('other-chat', { code: 'something_else' }) as never))
  expect(stream.state('other-chat').busy).toBe(true)
  expect(stream.state('other-chat').messages.some(message => message.error)).toBe(false)
})

it('leaves another window\'s typed turn alone', () => {
  // Same chat open elsewhere: that window sent typed text, not this voice claim.
  const stream = awaitingVoiceTurn('shared-chat')
  stream.states.set('shared-chat', { ...stream.state('shared-chat'), voiceClaimSubmissionId: null } as never)
  act(() => stream.handleEvent(refusal('shared-chat') as never))
  expect(stream.state('shared-chat').busy).toBe(true)
  expect(stream.state('shared-chat').messages.some(message => message.error)).toBe(false)
})

it('renders a late refusal once, over a watchdog card that offered no Retry', () => {
  const stream = awaitingVoiceTurn('late-chat')
  const silent = stream.state('late-chat')
  // What the silence watchdog stamps on a pending voice claim before the frame lands.
  stream.states.set('late-chat', { ...silent, awaitingResponse: false, busy: false, messages: withNoReplyNotice(silent.messages, false) } as never)
  expect(stream.state('late-chat').messages.at(-1)?.errorSurface?.retryable).toBe(false)
  act(() => stream.handleEvent(refusal('late-chat') as never))
  const cards = stream.state('late-chat').messages.filter(message => message.error)
  expect(cards).toHaveLength(1)
  expect(cards[0].error).toBe(REASON)
  expect(cards[0].errorSurface).toMatchObject({ code: 'voice_claim_unverified', retryable: false })
})

it('retires the voice claim once any turn starts, so a later turn keeps Retry', () => {
  const stream = awaitingVoiceTurn('started-chat')
  act(() => stream.handleEvent({ type: 'message.start', session_id: 'started-chat', authority_epoch: 2, execution_generation: 1, admission_id: 'typed' } as never))
  expect(stream.state('started-chat').voiceClaimSubmissionId).toBeNull()
  act(() => stream.handleEvent({ type: 'message.complete', session_id: 'started-chat', authority_epoch: 2, execution_generation: 1, admission_id: 'typed', payload: { text: 'done' } } as never))
  const after = stream.state('started-chat')
  // A stale refusal for the earlier claim no longer lands on this chat.
  act(() => stream.handleEvent(refusal('started-chat') as never))
  expect(stream.state('started-chat').messages).toEqual(after.messages)
})
