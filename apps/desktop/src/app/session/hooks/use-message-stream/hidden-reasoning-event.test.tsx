import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setShowReasoningFromConfig } from '@/store/reasoning-disclosure'

import { renderMessageStream } from './test-harness'

const SID = 'session-1'

afterEach(() => {
  cleanup()
  setShowReasoningFromConfig(undefined)
  vi.restoreAllMocks()
})

// The gateway's own show_reasoning flag is copied when a session starts, so
// after Reasoning Blocks is turned off mid-session it keeps streaming reasoning.
// A hidden reasoning part still counts as reply content: the loading row goes
// away and nothing visible takes its place while the model thinks.
describe('reasoning ingest follows display.show_reasoning', () => {
  it.each([true, false])('show_reasoning=%s decides whether reasoning and MoA lines reach the reply', enabled => {
    setShowReasoningFromConfig(enabled)
    const stream = renderMessageStream(SID)

    const emit = (type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) =>
      act(() => stream.handleEvent({ payload, session_id: SID, type }))

    emit('message.start')
    emit('moa.progress', { label: 'model-a', refs_done: 1, refs_total: 1 })
    emit('moa.phase', { phase: 'aggregator', refs_done: 1, refs_total: 1 })
    emit('moa.reference', { count: 1, index: 1, label: 'model-a', text: 'advice' })
    emit('reasoning.available', { text: 'considered' })
    emit('reasoning.delta', { text: 'thinking' })
    emit('message.complete', { text: 'The answer.' })

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.parts.some(part => part.type === 'reasoning')).toBe(enabled)
    expect(stream.text()).toBe('The answer.')
  })
})
