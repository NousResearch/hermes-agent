import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'

import { appendLiveSessionProjection, preserveLocalPendingTurnMessages } from './utils'

const user = (id: string, text: string, rowId?: number, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role: 'user',
  parts: [textPart(text)],
  ...(rowId !== undefined ? { rowId } : {}),
  ...extra
})

const assistant = (id: string, text: string, toolCallId: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role: 'assistant',
  parts: [{ type: 'tool-call', toolCallId, toolName: 'terminal', result: 'ok' }, ...(text ? [textPart(text)] : [])],
  ...extra
})

const ids = (messages: ChatMessage[]) => messages.map(message => message.id)

describe('page-omitted live-turn prefix restoration', () => {
  it('restores a rowId-bearing prompt before a page that starts inside its durable tool turn', () => {
    const prompt = user('user-paged', 'Inspect', 100)
    const live = assistant('assistant-stream-live', 'Done.', 'call-1', { pending: false })
    const page = [assistant('row-folded', 'Done.', 'call-1', { rowId: 120 })]

    const restored = preserveLocalPendingTurnMessages(page, [prompt, live])
    expect(ids(restored)).toEqual([prompt.id, page[0].id])
    expect(ids(preserveLocalPendingTurnMessages(page, restored))).toEqual(ids(restored))

    const backfilled = [user('row-user', 'Inspect', 100), ...page]
    expect(ids(preserveLocalPendingTurnMessages(backfilled, restored))).toEqual(ids(backfilled))
  })

  it('keeps the cached attachment prompt ahead of a running hydrated tool page without projecting a tail duplicate', () => {
    const prompt = user('user-paged', 'Inspect', 100, { attachmentRefs: ['@image:/fixture.png'] })
    const live = assistant('assistant-stream-live', 'Working.', 'call-1', { pending: true })
    const page = [assistant('row-tool', '', 'call-1', { rowId: 120 })]

    const restored = appendLiveSessionProjection(
      page,
      {
        session_id: 'session',
        inflight: { user: '@image:/fixture.png\nInspect', assistant: 'Working.', streaming: true }
      },
      [prompt, live]
    )

    expect(restored[0]).toBe(prompt)
    expect(restored.filter(message => message.role === 'user')).toHaveLength(1)
    expect(restored[0].attachmentRefs).toEqual(['@image:/fixture.png'])
  })

  it('restores a page-omitted mid-turn correction with its preceding sealed output in local order', () => {
    const prompt = user('user-1000', 'Original request', 100)
    const before = assistant('assistant-stream-before', 'Before correction.', 'call-before', { interim: true })
    const correction = user('user-2000', 'Change course', 150)
    const after = assistant('assistant-stream-after', 'After correction.', 'call-after', { pending: true })
    const page = [assistant('row-after', 'After correction.', 'call-after', { rowId: 160 })]

    const restored = preserveLocalPendingTurnMessages(page, [prompt, before, correction, after])

    expect(ids(restored)).toEqual([prompt.id, before.id, correction.id, page[0].id])
    expect(chatMessageText(restored[2])).toBe('Change course')
    expect(ids(preserveLocalPendingTurnMessages(page, restored))).toEqual(ids(restored))
  })

  it('does not resurrect an old acknowledged prompt when the page has no occurrence from its local turn', () => {
    const stale = user('user-stale', 'Old request', 100)
    const staleReply = assistant('assistant-stream-stale', 'Old answer', 'call-old', { pending: false })
    const page = [assistant('row-new', 'Different answer', 'call-new', { rowId: 200 })]

    expect(preserveLocalPendingTurnMessages(page, [stale, staleReply])).toEqual(page)
  })

  it('does not cross a foreign persisted user occurrence to repair an older local prefix', () => {
    const prompt = user('user-paged', 'Inspect', 100)
    const live = assistant('assistant-stream-live', 'Done.', 'call-1', { pending: false })
    const page = [user('row-other-user', 'Other turn', 110), assistant('row-other', 'Done.', 'call-1', { rowId: 120 })]

    expect(preserveLocalPendingTurnMessages(page, [prompt, live])).toEqual(page)
  })
})
