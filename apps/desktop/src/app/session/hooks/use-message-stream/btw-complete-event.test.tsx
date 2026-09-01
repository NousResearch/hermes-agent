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

describe('btw.complete event', () => {
  beforeEach(() => {
    mountStream()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  // #99065: prompt.btw delivers the answer here. The slash-worker route
  // printed it after the stdout capture window closed, so only the ack rendered.
  it('appends the answer to the originating session as a system message', () => {
    emit('btw.complete', { task_id: 'btw_ab12cd', question: 'which file was that error in?', text: 'src/main.ts' })

    const message = lastMessage()

    expect(message?.role).toBe('system')
    expect(message?.id).toBe('btw-complete-btw_ab12cd')
    expect(stream.text()).toBe('[btw "which file was that error in?" (btw_ab12cd)]\nsrc/main.ts')
  })

  it('keeps another session untouched when the event targets this one', () => {
    emit('btw.complete', { task_id: 'btw_1', text: 'for the other chat' }, OTHER_SID)

    expect(lastMessage()).toBeUndefined()
    expect(stream.text(OTHER_SID)).toBe('[btw (btw_1)]\nfor the other chat')
  })

  it('omits the question and task-id header bits when the backend omits them', () => {
    emit('btw.complete', { text: 'the answer' })

    expect(stream.text()).toBe('[btw]\nthe answer')
  })

  it('drops an empty completion instead of appending a blank line', () => {
    emit('btw.complete', { task_id: 'btw_1', question: 'q', text: '   ' })

    expect(lastMessage()).toBeUndefined()
  })
})
)