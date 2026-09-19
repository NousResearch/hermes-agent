import type { GatewayEventMap, GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $petActivity, $petState, setPetActivity } from '@/store/pet'
import { toolCompletePayload, toolStartPayload } from '@/test/contract'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const OTHER_SID = 'session-2'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit<K extends GatewayEventName>(type: K, payload: GatewayEventMap[K], sessionId = SID) {
  act(() => stream.emit(type, payload, sessionId))
}

describe('pet tool-failure reaction', () => {
  beforeEach(() => {
    setPetActivity({
      busy: false,
      awaitingInput: false,
      toolRunning: false,
      reasoning: false,
      error: false,
      justCompleted: false,
      celebrate: false
    })
  })

  afterEach(() => {
    cleanup()
    setPetActivity({ error: false, toolRunning: false })
    vi.restoreAllMocks()
  })

  it('briefly shows failed when the active session has an isolated tool error', () => {
    mountStream()

    emit('tool.start', toolStartPayload({ name: 'terminal', tool_id: 'tool-1' }))
    // A failed tool rides the wire as a non-empty `result.error`.
    emit('tool.complete', toolCompletePayload({ name: 'terminal', result: { error: 'exit code 1' }, tool_id: 'tool-1' }))

    expect($petActivity.get().error).toBe(true)
    expect($petState.get()).toBe('failed')
  })

  it('does not show failed for a successful tool or a background-session failure', () => {
    mountStream()

    emit('tool.complete', toolCompletePayload({ name: 'terminal', result: 'ok', tool_id: 'tool-1' }))
    expect($petActivity.get().error).toBe(false)

    emit(
      'tool.complete',
      toolCompletePayload({ name: 'terminal', result: { error: 'exit code 1' }, tool_id: 'tool-2' }),
      OTHER_SID
    )
    expect($petActivity.get().error).toBe(false)
  })
})
