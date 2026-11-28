import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $providerWaitSessions } from '@/store/provider-wait'
import { clearAllSessionStates, dropSessionState } from '@/store/session-states'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
let stream: MessageStreamHarness

function emit(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

describe('provider wait visibility', () => {
  beforeEach(async () => {
    $providerWaitSessions.set({})
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
    $providerWaitSessions.set({})
    vi.restoreAllMocks()
  })

  it('surfaces explained waits but ignores generic spinner rewrites', () => {
    emit('thinking.delta', { text: '⏳ waiting on local-model — 30s with no output yet' })
    expect($providerWaitSessions.get()).toEqual({
      [SID]: '⏳ waiting on local-model — 30s with no output yet'
    })

    emit('thinking.delta', { text: '◉_◉ cogitating...' })
    expect($providerWaitSessions.get()).toEqual({})
  })

  it.each(['message.delta', 'reasoning.delta', 'tool.start', 'message.complete', 'error'] as const)(
    'clears the wait when %s proves the turn progressed or ended',
    type => {
      emit('thinking.delta', { text: '⚠ no output from provider for 900s — reconnecting...' })
      emit(type, type === 'tool.start' ? { name: 'terminal', tool_id: 'tool-1' } : { text: 'progress' })

      expect($providerWaitSessions.get()).toEqual({})
    }
  )

  it('surfaces the queue frame and counts-bearing prefill frames', () => {
    emit('thinking.delta', { text: '⏳ waiting for a free slot — the local server is busy with another request' })
    expect($providerWaitSessions.get()).toEqual({
      [SID]: '⏳ waiting for a free slot — the local server is busy with another request'
    })

    emit('thinking.delta', { text: '⚙ processing prompt — 46,500 / 80,000 tokens (58%)' })
    expect($providerWaitSessions.get()).toEqual({
      [SID]: '⚙ processing prompt — 46,500 / 80,000 tokens (58%)'
    })
  })

  it('clears a prefill wait when a new call starts (message.start), with no stale percent', () => {
    emit('thinking.delta', { text: '⚙ processing prompt — 46,500 / 80,000 tokens (58%)' })
    expect($providerWaitSessions.get()).toEqual({
      [SID]: '⚙ processing prompt — 46,500 / 80,000 tokens (58%)'
    })

    emit('message.start', {})
    expect($providerWaitSessions.get()).toEqual({})
  })

  it('replaces the old prefill percent when a new call reports progress after a tool', () => {
    emit('thinking.delta', { text: '⚙ processing prompt — 31%' })
    emit('tool.start', { name: 'terminal', tool_id: 'tool-1' })
    expect($providerWaitSessions.get()).toEqual({})

    emit('thinking.delta', { text: '⚙ processing prompt — 12,288 / ~39,551 tokens (31%)' })
    expect($providerWaitSessions.get()).toEqual({
      [SID]: '⚙ processing prompt — 12,288 / ~39,551 tokens (31%)'
    })
  })

  it('clears the wait when its runtime session is dropped', () => {
    emit('thinking.delta', { text: '⏳ waiting on local-model — 30s with no output yet' })

    dropSessionState(SID)

    expect($providerWaitSessions.get()).toEqual({})
  })

  it('clears every wait when gateway session state is reset', () => {
    emit('thinking.delta', { text: '⏳ waiting on local-model — 30s with no output yet' })

    clearAllSessionStates()

    expect($providerWaitSessions.get()).toEqual({})
  })
})
