import { describe, expect, it } from 'vitest'

import { type ChatMessage } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

// Desktop 2026-10-02 repro: one reply rendered twice because the refreshed page's
// fold row and a settled live-tail row both survived the merge. In every shape the
// dropped row's text is contained in the surviving row, so dropping loses nothing.
const answer = 'The answer body, long enough to be a real reply rather than a fragment. '.repeat(3)
const fold = `Narration recorded before the tool round. ${answer}`

describe('preserveLocalPendingTurnMessages — duplicate render rows (#129993)', () => {
  it('drops a row the same turn already carries in full', () => {
    const previous = [msg('u-live', 'user', 'do the work', { rowId: 900 })]
    const next = [
      msg('s-u1', 'user', 'do the work', { rowId: 900 }),
      msg('s-fold', 'assistant', fold, { rowId: 901 }),
      msg('s-dup', 'assistant', answer, { rowId: 902 })
    ]

    const ids = preserveLocalPendingTurnMessages(next, previous).map(message => message.id)

    expect(ids).toContain('s-fold')
    expect(ids).not.toContain('s-dup')
  })

  it('drops a settled live-tail row that a mid-turn correction split away from its fold', () => {
    const previous = [msg('u-live', 'user', 'start', { rowId: 900 })]
    const next = [
      msg('s-u1', 'user', 'start', { rowId: 900 }),
      msg('s-a1', 'assistant', fold, { rowId: 901 }),
      msg('s-u2', 'user', 'a correction', { rowId: 902 }),
      msg('assistant-stream-1790923032111-13', 'assistant', answer, { pending: false, rowId: 903 })
    ]

    const ids = preserveLocalPendingTurnMessages(next, previous).map(message => message.id)

    expect(ids).toContain('s-a1')
    expect(ids).not.toContain('assistant-stream-1790923032111-13')
  })

  it('keeps only one of two textually identical committed rows in the same turn', () => {
    const previous = [msg('u-live', 'user', 'q', { rowId: 900 })]
    const next = [
      msg('s-u1', 'user', 'q', { rowId: 900 }),
      msg('s-a1', 'assistant', answer, { rowId: 901 }),
      msg('s-a2', 'assistant', answer, { rowId: 902 })
    ]

    const assistants = preserveLocalPendingTurnMessages(next, previous).filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(1)
  })

  it('never merges identical replies that belong to different turns', () => {
    const previous = [msg('u-live', 'user', 'q', { rowId: 900 })]
    const next = [
      msg('s-u1', 'user', 'q1', { rowId: 900 }),
      msg('s-a1', 'assistant', answer, { rowId: 901 }),
      msg('s-u2', 'user', 'q2', { rowId: 902 }),
      msg('s-a2', 'assistant', answer, { rowId: 903 })
    ]

    const ids = preserveLocalPendingTurnMessages(next, previous).map(message => message.id)

    expect(ids).toContain('s-a1')
    expect(ids).toContain('s-a2')
  })

  it('never merges a reply into a longer reply from a different turn', () => {
    const previous = [msg('u-live', 'user', 'q', { rowId: 900 })]
    const next = [
      msg('s-u1', 'user', 'q1', { rowId: 900 }),
      msg('s-a1', 'assistant', answer, { rowId: 901 }),
      msg('s-u2', 'user', 'q2', { rowId: 902 }),
      msg('s-a2', 'assistant', `Second, longer answer. ${answer}`, { rowId: 903 })
    ]

    const ids = preserveLocalPendingTurnMessages(next, previous).map(message => message.id)

    expect(ids).toContain('s-a1')
    expect(ids).toContain('s-a2')
  })

  it('leaves a still-streaming row alone', () => {
    const previous = [msg('u-live', 'user', 'do the work', { rowId: 900 })]
    const next = [
      msg('s-u1', 'user', 'do the work', { rowId: 900 }),
      msg('s-fold', 'assistant', fold, { rowId: 901 }),
      msg('assistant-stream-live-1', 'assistant', answer, { pending: true, rowId: 902 })
    ]

    const ids = preserveLocalPendingTurnMessages(next, previous).map(message => message.id)

    expect(ids).toContain('assistant-stream-live-1')
    expect(ids).toContain('s-fold')
  })
})
