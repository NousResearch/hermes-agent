import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { renderMessageStream } from './test-harness'

vi.mock('@/lib/completion-sound', () => ({ playCompletionSound: vi.fn() }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))
afterEach(cleanup)
const sid = 'user-projection'
const message = { role: 'user', text: 'canonical words', row_id: 41, message_uid: 'occurrence', timestamp: 10 }

it('projects external turns once and confirms only the submitter’s matching optimistic bubble', () => {
  const viewer = renderMessageStream(sid)
  const sender = renderMessageStream(sid, {
    states: new Map([
      [
        sid,
        createClientSessionState('stored', [
          { id: 'optimistic', role: 'user', parts: [textPart('[[ collapsed paste ]]')] }
        ])
      ]
    ])
  })
  const event = {
    type: 'message.user' as const,
    session_id: sid,
    payload: { message, client_message_ids: ['optimistic'] }
  }
  act(() => {
    viewer.handleEvent(event)
    sender.handleEvent(event)
  })
  expect(viewer.state().messages).toHaveLength(1)
  expect(chatMessageText(viewer.state().messages[0])).toBe('canonical words')
  expect(sender.state().messages).toHaveLength(1)
  expect(chatMessageText(sender.state().messages[0])).toBe('[[ collapsed paste ]]')
  const confirmed = sender.state()
  act(() => {
    viewer.handleEvent(event)
    sender.handleEvent(event)
  })
  expect(sender.state()).toBe(confirmed)
  expect(viewer.state().messages).toHaveLength(1)
  act(() => viewer.handleEvent({ ...event, payload: { message: { ...message, row_id: 42, message_uid: 'another' } } }))
  expect(viewer.state().messages).toHaveLength(2)
})

it('rebinds a hydrated queued row at dispatch and consolidates merged local submissions', () => {
  const early = toChatMessages([{ ...message, role: 'user', content: message.text, row_id: 5 }])
  const viewer = renderMessageStream(sid, {
    states: new Map([
      [
        sid,
        createClientSessionState('stored', [
          ...early,
          { id: 'previous-reply', role: 'assistant', parts: [textPart('previous reply')] }
        ])
      ]
    ])
  })
  act(() => viewer.handleEvent({ type: 'message.user', session_id: sid, payload: { message } }))
  expect(viewer.state().messages.map(m => m.role)).toEqual(['assistant', 'user'])
  expect(viewer.state().messages[1].rowId).toBe(41)
  const dispatched = viewer.state()
  act(() =>
    viewer.handleEvent({ type: 'message.user', session_id: sid, payload: { message: { ...message, row_id: 5 } } })
  )
  expect(viewer.state()).toBe(dispatched)
  const sender = renderMessageStream(sid, {
    states: new Map([
      [
        sid,
        createClientSessionState('stored', [
          { id: 'c1', role: 'user', parts: [textPart('one')] },
          { id: 'c2', role: 'user', parts: [textPart('two')] }
        ])
      ]
    ])
  })
  act(() =>
    sender.handleEvent({
      type: 'message.user',
      session_id: sid,
      payload: { message, client_message_ids: ['c1', 'c2'] }
    })
  )
  expect(sender.state().messages).toHaveLength(1)
  expect(chatMessageText(sender.state().messages[0])).toBe('canonical words')
})
