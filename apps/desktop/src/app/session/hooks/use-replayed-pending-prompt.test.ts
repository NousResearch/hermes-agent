import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $replayedPendingPrompt } from '@/store/session'

import { useReplayedPendingPrompt } from './use-replayed-pending-prompt'

const clarify = {
  event: 'clarify.request',
  kind: 'clarify',
  payload: { choices: ['staging', 'production'], question: 'Which environment?', request_id: 'req-1' },
  request_id: 'req-1'
}

afterEach(() => {
  $replayedPendingPrompt.set(null)
})

describe('useReplayedPendingPrompt', () => {
  it('replays the parked prompt as the event the backend originally emitted', () => {
    const handleGatewayEvent = vi.fn()
    renderHook(() => useReplayedPendingPrompt(handleGatewayEvent))

    act(() => {
      $replayedPendingPrompt.set({ pending: clarify, sessionId: 'sid-1' })
    })

    expect(handleGatewayEvent).toHaveBeenCalledTimes(1)
    expect(handleGatewayEvent).toHaveBeenCalledWith({
      payload: clarify.payload,
      session_id: 'sid-1',
      type: 'clarify.request'
    })
  })

  it('drains the atom so a re-render cannot replay the same request twice', () => {
    const handleGatewayEvent = vi.fn()
    const { rerender } = renderHook(() => useReplayedPendingPrompt(handleGatewayEvent))

    act(() => {
      $replayedPendingPrompt.set({ pending: clarify, sessionId: 'sid-1' })
    })

    rerender()
    rerender()

    expect(handleGatewayEvent).toHaveBeenCalledTimes(1)
    expect($replayedPendingPrompt.get()).toBeNull()
  })

  it('stays inert while nothing is parked', () => {
    const handleGatewayEvent = vi.fn()
    const { rerender } = renderHook(() => useReplayedPendingPrompt(handleGatewayEvent))

    rerender()

    expect(handleGatewayEvent).not.toHaveBeenCalled()
  })

  it('replays every blocking bridge, not just clarify', () => {
    const handleGatewayEvent = vi.fn()
    renderHook(() => useReplayedPendingPrompt(handleGatewayEvent))

    for (const event of ['sudo.request', 'secret.request', 'terminal.read.request']) {
      act(() => {
        $replayedPendingPrompt.set({
          pending: { event, kind: event.replace('.request', ''), payload: {}, request_id: `req-${event}` },
          sessionId: 'sid-1'
        })
      })
    }

    expect(handleGatewayEvent.mock.calls.map(call => call[0].type)).toEqual([
      'sudo.request',
      'secret.request',
      'terminal.read.request'
    ])
  })

  it('tolerates a request the backend sent without a payload', () => {
    const handleGatewayEvent = vi.fn()
    renderHook(() => useReplayedPendingPrompt(handleGatewayEvent))

    act(() => {
      $replayedPendingPrompt.set({
        pending: { event: 'sudo.request', kind: 'sudo', request_id: 'req-2' },
        sessionId: 'sid-1'
      })
    })

    expect(handleGatewayEvent).toHaveBeenCalledWith({
      payload: {},
      session_id: 'sid-1',
      type: 'sudo.request'
    })
  })
})
