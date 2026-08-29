import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const OTHER_SID = 'session-2'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}, sessionId = SID) {
  act(() => stream.handleEvent({ payload, session_id: sessionId, type }))
}

function lastMessage(id = SID) {
  return stream.state(id).messages.at(-1)
}

describe('background.complete event', () => {
  beforeEach(() => {
    mountStream()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  // #97635: the finished background turn's response has to land in the
  // conversation that started it — the event carries the originating session
  // id, so the result is appended there instead of surfacing in an unlinked
  // bg_* session.
  it('appends the result to the originating session as a system message', () => {
    emit('background.complete', { task_id: 'bg_104613_385db3', text: 'the answer' })

    const message = lastMessage()

    expect(message?.role).toBe('system')
    expect(message?.id).toBe('background-complete-bg_104613_385db3')
    expect(stream.text()).toBe('[bg bg_104613_385db3]\nthe answer')
  })

  it('keeps another session untouched when the event targets this one', () => {
    emit('background.complete', { task_id: 'bg_1', text: 'for the other chat' }, OTHER_SID)

    expect(lastMessage()).toBeUndefined()
    expect(stream.text(OTHER_SID)).toBe('[bg bg_1]\nfor the other chat')
  })

  it('appends without a task-id header when the backend omits the id', () => {
    emit('background.complete', { text: 'the answer' })

    expect(stream.text()).toBe('the answer')
  })

  it('drops an empty completion instead of appending a blank line', () => {
    emit('background.complete', { task_id: 'bg_1', text: '   ' })

    expect(lastMessage()).toBeUndefined()
  })
})
