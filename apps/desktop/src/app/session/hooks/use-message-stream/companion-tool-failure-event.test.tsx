import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $companionActivity, $companionState, setCompanionActivity } from '@/store/companion'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const OTHER_SID = 'session-2'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}, sessionId = SID) {
  act(() => stream.handleEvent({ payload, session_id: sessionId, type }))
}

describe('companion tool-failure reaction', () => {
  beforeEach(() => {
    setCompanionActivity({
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
    setCompanionActivity({ error: false, toolRunning: false })
    vi.restoreAllMocks()
  })

  it('briefly shows failed when the active session has an isolated tool error', () => {
    mountStream()

    emit('tool.start', { name: 'terminal', tool_id: 'tool-1' })
    emit('tool.complete', { name: 'terminal', tool_id: 'tool-1', error: 'exit code 1' })

    expect($companionActivity.get().error).toBe(true)
    expect($companionState.get()).toBe('failed')
  })

  it('does not show failed for a successful tool or a background-session failure', () => {
    mountStream()

    emit('tool.complete', { name: 'terminal', tool_id: 'tool-1' })
    expect($companionActivity.get().error).toBe(false)

    emit('tool.complete', { name: 'terminal', tool_id: 'tool-2', error: 'exit code 1' }, OTHER_SID)
    expect($companionActivity.get().error).toBe(false)
  })
})
