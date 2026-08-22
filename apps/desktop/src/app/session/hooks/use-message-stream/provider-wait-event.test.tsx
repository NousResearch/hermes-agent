import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $providerWaitSessions } from '@/store/provider-wait'
import { clearAllSessionStates, dropSessionState } from '@/store/session-states'
import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
let stream: MessageStreamHarness

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
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
