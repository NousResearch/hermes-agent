import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { claimSideChatTask, releaseSideChatTask } from '@/store/side-chat'

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
    expect(stream.text()).toContain('src/main.ts')
  })

  it('keeps another session untouched when the event targets this one', () => {
    emit('btw.complete', { task_id: 'btw_1', text: 'for the other chat' }, OTHER_SID)

    expect(lastMessage()).toBeUndefined()
    expect(stream.text(OTHER_SID)).toContain('for the other chat')
  })

  it('drops an empty completion instead of appending a blank line', () => {
    emit('btw.complete', { task_id: 'btw_1', question: 'q', text: '   ' })

    expect(lastMessage()).toBeUndefined()
  })

  // The floating side chat exists so an aside can stay OUT of the conversation
  // it is about. An answer it asked for goes to that window and nowhere else.
  describe('when the floating side chat asked the question', () => {
    let reply: ReturnType<typeof vi.fn>

    beforeEach(() => {
      reply = vi.fn()
      Object.defineProperty(window, 'hermesDesktop', {
        configurable: true,
        value: { sideChat: { open: vi.fn(), reply } }
      })
    })

    afterEach(() => {
      Reflect.deleteProperty(window, 'hermesDesktop')
    })

    it('routes the answer to the window and leaves the transcript alone', () => {
      claimSideChatTask('btw_ab12cd', 'ask-1')
      emit('btw.complete', { task_id: 'btw_ab12cd', question: 'which file?', text: 'src/main.ts' })

      expect(reply).toHaveBeenCalledWith({ askId: 'ask-1', error: '', text: 'src/main.ts' })
      expect(lastMessage()).toBeUndefined()
    })

    it('settles an empty answer as a failure rather than leaving it thinking', () => {
      claimSideChatTask('btw_empty', 'ask-2')
      emit('btw.complete', { task_id: 'btw_empty', text: '   ' })

      expect(reply).toHaveBeenCalledWith(expect.objectContaining({ askId: 'ask-2', text: '' }))
      expect(reply.mock.calls[0][0].error).not.toBe('')
      expect(lastMessage()).toBeUndefined()
    })

    it('still renders an inline /btw the side chat did not ask for', () => {
      claimSideChatTask('btw_mine', 'ask-3')
      emit('btw.complete', { task_id: 'btw_someone_elses', text: 'inline answer' })

      expect(reply).not.toHaveBeenCalled()
      expect(stream.text()).toContain('inline answer')
      releaseSideChatTask('btw_mine')
    })

    it('consumes a task id only once, so a duplicate event falls through', () => {
      claimSideChatTask('btw_dup', 'ask-4')
      emit('btw.complete', { task_id: 'btw_dup', text: 'first delivery' })
      emit('btw.complete', { task_id: 'btw_dup', text: 'first delivery' })

      expect(reply).toHaveBeenCalledTimes(1)
      expect(stream.text()).toContain('first delivery')
    })
  })
})
