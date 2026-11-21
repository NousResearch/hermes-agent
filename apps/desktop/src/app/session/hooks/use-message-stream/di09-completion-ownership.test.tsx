// State-level versions of the two qualified DI09 scenarios: no plugin SDK required.
import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createGatewayEventDedupe, DUPLICATE_WINDOW_MS } from '@/app/gateway/gateway-event-dedupe'
import { appendMidTurnUserMessage } from '@/app/session/hooks/use-prompt-actions/rewind'
import { textPart } from '@/lib/chat-messages'
import { toolCallOwnerMessageId } from '@/lib/chat-messages/tool-parts'
import { $sessionStates } from '@/store/session-states'

import recorded from './di09-recorded-events.json'
import { renderMessageStream } from './test-harness'

const SID = 'fixture-session-0001'
const beta = recorded[10].event as GatewayEvent<'tool.complete'>
const started = recorded[8].event as GatewayEvent<'tool.start'>
let clock: ReturnType<typeof vi.spyOn>

const parts = (h: ReturnType<typeof renderMessageStream>) =>
  h
    .state()
    .messages.flatMap(message => message.parts)
    .filter(part => part.type === 'tool-call')

beforeEach(() => {
  clock = vi.spyOn(Date, 'now').mockReturnValue(Date.parse(recorded[0].observedAt))
  $sessionStates.set({})
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  $sessionStates.set({})
})

function mount() {
  const h = renderMessageStream(SID)
  const dedupe = createGatewayEventDedupe()

  const send = (event: GatewayEvent) => {
    const admitted = dedupe.admit(event)

    if (admitted) {
      act(() => h.handleEvent(event))
    }

    return admitted
  }

  const finish = () => {
    for (const row of recorded) {
      clock.mockReturnValue(Date.parse(row.observedAt))
      expect(send(row.event as GatewayEvent)).toBe(true)
    }
  }

  return { h, send, finish }
}

it('keeps the recorded seq 11 completion on its original occurrence after 123.522 seconds', () => {
  const { h, send, finish } = mount()
  finish()
  expect(h.state().streamId).toBeNull()
  expect(parts(h)).toHaveLength(3)
  const before = structuredClone(h.state())
  clock.mockReturnValue(Date.parse('2026-11-03T19:37:07.891Z'))
  expect(Date.now() - Date.parse(recorded[10].observedAt)).toBe(123_522)
  expect(Date.now() - Date.parse(recorded[10].observedAt)).toBeGreaterThan(DUPLICATE_WINDOW_MS)
  expect(send(structuredClone(beta))).toBe(true)
  expect(parts(h)).toHaveLength(3)
  expect(h.state()).toEqual(before) // includes every result, timestamp and supplied duration
})

it('does not let the old completion resolve a pending later reuse of the same id', () => {
  const { h, send, finish } = mount()
  finish()
  const originalMessages = structuredClone(h.state().messages)
  const originalParts = structuredClone(parts(h))
  clock.mockReturnValue(Date.parse('2026-11-03T19:38:00Z'))
  // A genuine later user turn, not a repeated start in the original turn.
  h.states.set(
    SID,
    appendMidTurnUserMessage(h.state(), {
      id: 'di09-newer-user-turn',
      role: 'user',
      parts: [textPart('Repeat the beta read in a new turn.')]
    })
  )
  expect(send({ type: 'message.start', session_id: SID, seq: 20, payload: {} })).toBe(true)
  expect(send({ ...started, seq: 21 })).toBe(true)
  expect(parts(h)).toHaveLength(4)
  const pendingBefore = structuredClone(parts(h)[3])
  expect(pendingBefore.result).toBeUndefined()
  expect(pendingBefore.completedAt).toBeUndefined()
  const ownerBefore = toolCallOwnerMessageId(h.state().messages, recorded[10].event.payload, 'complete')
  expect(ownerBefore).toBeTruthy()
  const stateBefore = structuredClone(h.state())

  expect(send(structuredClone(beta))).toBe(true)
  expect(parts(h)).toHaveLength(4)
  expect(parts(h)[3]).toEqual(pendingBefore)
  expect(toolCallOwnerMessageId(h.state().messages, recorded[10].event.payload, 'complete')).toBe(ownerBefore)
  expect(h.state()).toEqual(stateBefore)

  expect(send({ ...beta, seq: 22, payload: { ...beta.payload, result: { content: 'later occurrence' } } })).toBe(true)
  expect(parts(h)).toHaveLength(4) // the genuine completion must not create a fifth occurrence
  expect(parts(h)[3].result).toEqual({ content: 'later occurrence' })
  expect(parts(h)[3].timestamp).toBe(pendingBefore.timestamp)
  expect(parts(h)[3].completedAt).toBe(Date.now() / 1000)
  const ownerMessage = h.state().messages.find(message => message.parts.includes(parts(h)[3]))
  expect(ownerMessage?.id).toBe(ownerBefore)
  expect(parts(h).slice(0, 3)).toEqual(originalParts)
  expect(h.state().messages.slice(0, originalMessages.length)).toEqual(originalMessages)
})
