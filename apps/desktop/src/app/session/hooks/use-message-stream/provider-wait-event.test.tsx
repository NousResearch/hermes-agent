import type { GatewayEventMap, GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $providerWaitSessions } from '@/store/provider-wait'
import { clearAllSessionStates, dropSessionState } from '@/store/session-states'
import {
  errorPayload,
  messageCompletePayload,
  messageDeltaPayload,
  reasoningDeltaPayload,
  thinkingDeltaPayload,
  toolStartPayload
} from '@/test/contract'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
let stream: MessageStreamHarness

function emit<K extends GatewayEventName>(type: K, payload: GatewayEventMap[K]) {
  act(() => stream.emit(type, payload, SID))
}

const wait = (text: string) => emit('thinking.delta', thinkingDeltaPayload({ text }))

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
    wait('⏳ waiting on local-model — 30s with no output yet')
    expect($providerWaitSessions.get()).toEqual({
      [SID]: '⏳ waiting on local-model — 30s with no output yet'
    })

    wait('◉_◉ cogitating...')
    expect($providerWaitSessions.get()).toEqual({})
  })

  it.each([
    ['message.delta', messageDeltaPayload({ text: 'progress' })],
    ['reasoning.delta', reasoningDeltaPayload({ text: 'progress' })],
    ['tool.start', toolStartPayload({ name: 'terminal', tool_id: 'tool-1' })],
    ['message.complete', messageCompletePayload({ text: 'progress' })],
    ['error', errorPayload({ message: 'progress' })]
  ] as const)('clears the wait when %s proves the turn progressed or ended', (type, payload) => {
    wait('⚠ no output from provider for 900s — reconnecting...')
    emit(type, payload)

    expect($providerWaitSessions.get()).toEqual({})
  })

  it('clears the wait when its runtime session is dropped', () => {
    wait('⏳ waiting on local-model — 30s with no output yet')

    dropSessionState(SID)

    expect($providerWaitSessions.get()).toEqual({})
  })

  it('clears every wait when gateway session state is reset', () => {
    wait('⏳ waiting on local-model — 30s with no output yet')

    clearAllSessionStates()

    expect($providerWaitSessions.get()).toEqual({})
  })
})
