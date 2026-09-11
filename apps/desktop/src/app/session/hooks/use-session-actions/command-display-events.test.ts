import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import { parseCommandDispatch, parseCommandDisplayEvent } from '@/lib/chat-runtime'

import {
  preserveLocalPendingTurnMessages,
  reconcileResumeMessages,
  selectBranchMessages,
  toBranchMessages
} from './utils'

const event = {
  id: 'display:496f46bd-cafe-4f32-a4bb-123456789012',
  role: 'system' as const,
  content: 'slash:/report\nOriginal report',
  timestamp: 100,
  display_kind: 'command_result' as const
}

describe('command display hydration and model-history boundary', () => {
  it.each(['id', 'row_id'] as const)('keeps real numeric %s for ordinary rows beside display events', field => {
    const rows = toChatMessages([
      { [field]: 71, role: 'user', content: 'Hello', timestamp: 1 },
      event,
      { [field]: 72, role: 'assistant', content: 'Reply', timestamp: 101 }
    ])

    expect(rows.map(row => row.rowId)).toEqual([71, undefined, 72])
    expect(toBranchMessages(rows).map(row => [row.content, row.source.rowId])).toEqual([
      ['Hello', 71],
      ['Reply', 72]
    ])
  })

  it.each(['id', 'row_id'] as const)('preserves the stable %s across different page positions', field => {
    const row = { ...event, id: undefined, [field]: event.id }
    const [first] = toChatMessages([row])
    const [, second] = toChatMessages([{ role: 'user', content: 'Hello', timestamp: 1 }, row])
    expect(first.id).toBe(event.id)
    expect(second.id).toBe(event.id)
    expect(first.displayKind).toBe('command_result')
    expect(first.rowId).toBeUndefined()
    expect(first.timestamp).toBe(100)
    expect(chatMessageText(first)).toBe(event.content)
  })

  it('keeps one live row when hydrate catches up and retains it through an older response', () => {
    const live = toChatMessages([event])
    expect(preserveLocalPendingTurnMessages([], live)).toEqual(live)
    const restored = toChatMessages([{ ...event, row_id: event.id }])
    const merged = preserveLocalPendingTurnMessages(reconcileResumeMessages(restored, live), live)
    expect(merged).toHaveLength(1)
    expect(merged[0].id).toBe(event.id)
    expect(toChatMessages([event])[0].id).toBe(merged[0].id)
  })

  it('never seeds display events even if a projection accidentally labels them as user or assistant', () => {
    const ordinary: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [textPart('Hello')] },
      { id: 'assistant', role: 'assistant', parts: [textPart('Reply')] }
    ]

    const events = toChatMessages([{ ...event, role: 'user' }])
    expect(events[0].role).toBe('system')
    const corrupted = { ...events[0], role: 'assistant' as const }
    const unmarked = { ...corrupted, displayKind: undefined }
    const rows = [ordinary[0], ...events, corrupted, unmarked, ordinary[1]]
    expect(toBranchMessages(rows).map(row => row.content)).toEqual(['Hello', 'Reply'])
    expect(selectBranchMessages(rows, rows).map(row => row.content)).toEqual(['Hello', 'Reply'])
  })

  it('does not attach command persistence metadata to skill/send dispatch', () => {
    expect(parseCommandDispatch({ type: 'plugin', output: 'Original report', display_event: event })?.type).toBe(
      'plugin'
    )
    expect(parseCommandDispatch({ type: 'send', message: 'Prompt', display_event: event })).toEqual({
      type: 'send',
      message: 'Prompt',
      notice: undefined,
      display: undefined
    })
    expect(parseCommandDisplayEvent({ ...event, role: 'user' })).toBeUndefined()
    expect(parseCommandDisplayEvent({ ...event, timestamp: NaN })).toBeUndefined()
  })
})
