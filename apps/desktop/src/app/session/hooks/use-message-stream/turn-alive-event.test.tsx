import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId } from '@/store/session'
import {
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState,
  setLiveTurnBackend
} from '@/store/session-states'

import { renderMessageStream } from './test-harness'

const SID = 'turn-alive'
// tui_gateway/turn_alive.py TURN_ALIVE_INTERVAL_S
const TURN_ALIVE_S = 15

let releaseBackend: () => void = () => undefined

beforeEach(() => {
  vi.useFakeTimers()
  clearAllSessionStates()
  $activeSessionId.set(SID)
})

afterEach(() => {
  cleanup()
  releaseBackend()
  releaseBackend = () => undefined
  vi.useRealTimers()
  clearAllSessionStates()
  $activeSessionId.set(null)
})

function quietTurn(): ClientSessionState {
  return {
    ...createClientSessionState('s-turn-alive'),
    awaitingResponse: true,
    busy: true,
    messages: [
      { id: 'a1', parts: [{ text: 'Running the test suite', type: 'text' }], pending: true, role: 'assistant' }
    ],
    sawAssistantPayload: true,
    streamId: 'a1',
    turnLive: true,
    turnStartedAt: Date.now()
  }
}

it('turn.alive keeps a quiet turn live without ever asking session.active_list', async () => {
  const live = quietTurn()
  publishSessionState(SID, live)
  const stream = renderMessageStream(SID, { states: new Map([[SID, live]]) })
  const request = vi.fn(async () => ({ sessions: [] }))
  releaseBackend = setLiveTurnBackend({ request: request as never })
  noteSessionEvent(SID)

  // A ten-minute tool call: nothing but the gateway's liveness frames.
  for (let tick = 0; tick < 40; tick += 1) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(TURN_ALIVE_S * 1000)
      stream.handleEvent({
        payload: { activity: 'executing tool: terminal', quiet_s: TURN_ALIVE_S, status: 'working' },
        session_id: SID,
        type: 'turn.alive'
      })
    })
  }

  expect(request).not.toHaveBeenCalled()
  expect($workingSessionIds.get()).toContain('s-turn-alive')
  // Liveness changes no state.
  expect($sessionStates.get()[SID]).toBe(live)
  expect(stream.state(SID)).toBe(live)

  // The frames stop (a backend without turn.alive, or one whose turn died):
  // the silence window falls back to asking.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(LIVE_TURN_EVENT_SILENCE_MS)
  })

  expect(request).toHaveBeenCalledTimes(1)
})

it('a turn.alive that lands after message.complete does not bring the finished turn back', async () => {
  // The gateway's ticker can race the terminal frame (it found the turn due a
  // moment before message.complete went out), and an older gateway keeps
  // sending frames while post-turn hooks hold `running`. The reply is settled:
  // no busy, no turn clock, and no silence check to keep the turn "live".
  const live = quietTurn()
  const states = new Map([[SID, live]])
  publishSessionState(SID, live)

  const stream = renderMessageStream(SID, {
    states,
    updateSessionState: (id, updater) => {
      const next = updater(states.get(id) ?? createClientSessionState())
      states.set(id, next)
      publishSessionState(id, next)

      return next
    }
  })

  const request = vi.fn(async () => ({ sessions: [{ id: SID, session_key: 's-turn-alive', status: 'working' }] }))
  releaseBackend = setLiveTurnBackend({ request: request as never })

  await act(async () => {
    stream.handleEvent({
      payload: { status: 'complete', text: 'Wave 4 complete.' },
      session_id: SID,
      type: 'message.complete'
    })
  })

  for (let tick = 0; tick < 4; tick += 1) {
    await act(async () => {
      stream.handleEvent({
        payload: { activity: 'executing tool: terminal', quiet_s: TURN_ALIVE_S, status: 'working' },
        session_id: SID,
        type: 'turn.alive'
      })
      await vi.advanceTimersByTimeAsync(TURN_ALIVE_S * 1000)
    })
  }

  await act(async () => {
    await vi.advanceTimersByTimeAsync(LIVE_TURN_EVENT_SILENCE_MS * 2)
  })

  const settled = $sessionStates.get()[SID]!
  expect(settled.busy).toBe(false)
  expect(settled.awaitingResponse).toBe(false)
  expect(settled.turnLive).toBe(false)
  expect(settled.turnStartedAt).toBeNull()
  expect($workingSessionIds.get()).not.toContain('s-turn-alive')
  // Nothing treated the late frames as a live turn worth checking.
  expect(request).not.toHaveBeenCalled()
})
