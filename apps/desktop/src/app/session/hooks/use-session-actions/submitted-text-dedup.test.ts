import { describe, expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart, chatMessageText, textPart } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const message = (
  id: string,
  role: ChatMessage['role'],
  parts: ChatMessagePart[],
  extra: Partial<ChatMessage> = {}
): ChatMessage => ({ id, role, parts, ...extra }) as ChatMessage

// #131848: the optimistic user bubble paints a DISPLAY projection of the
// submitted text — chip labels for @terminal: selections, resolved @file:
// refs — while the gateway persists the TRANSPORT text. A background timeline
// refresh that returns the persisted twin before the submit receipt stamps the
// rowId finds no text match in the optimistic-user dedup gate, so the stored
// copy is appended beside the still-unmatched optimistic row: the duplicate
// exchange the reporter sees. The row carries the exact submitted text as
// `submitText`, and the gate compares it against the persisted copy.
describe('preserveLocalPendingTurnMessages — optimistic user dedup by submitted text (#131848)', () => {
  // The transport shape freezeComposerTransportPayload produces for a
  // terminal-selection chip: the label rides a fenced row dump, the display
  // shows just the chip caption.
  const transportText = 'Read the failing test\n\n```\nRow 3: expected 4, got 3\n```'
  const displayText = 'Read the failing test'

  const optimisticRow = message('user-1727-abc123', 'user', [textPart(displayText)], {
    submitText: transportText
  })

  const persistedTwin = message('9-2-user', 'user', [textPart(transportText)], { rowId: 18715 })

  const persistedReply = message('9-3-assistant', 'assistant', [textPart('The off-by-one is in the reducer.')], {
    pending: false,
    rowId: 18716
  })

  it('a persisted copy matching the submitted transport text replaces the optimistic row', () => {
    const previous = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      optimisticRow
    ]

    const next = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      persistedTwin,
      persistedReply
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    // One user row for the submitted turn, its committed id — no optimistic
    // duplicate appended beside the persisted twin.
    const userRows = merged.filter(row => row.role === 'user')
    expect(userRows).toHaveLength(2)
    expect(userRows[1]).toMatchObject({ id: '9-2-user', rowId: 18715 })
    expect(chatMessageText(userRows[1])).toBe(transportText)
  })

  it('still renders a single pair when the refresh lands between submit and the rowId receipt', () => {
    // The rowId receipt never arrived: the optimistic row carries no rowId and
    // the refreshed page holds the persisted copy. Only submitText bridges them.
    const bareOptimistic = message('user-1727-abc123', 'user', [textPart(displayText)], {
      submitText: transportText
    })

    const previous = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      bareOptimistic
    ]

    const next = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      persistedTwin,
      persistedReply
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged.filter(row => row.role === 'user')).toHaveLength(2)
    expect(merged.filter(row => row.role === 'assistant')).toHaveLength(2)
  })

  it('keeps a genuine unacknowledged repeat whose submitted text matches nothing persisted', () => {
    // A rowId-less repeat that never made it to the gateway (submit still in
    // flight) must survive a refresh that happens to carry an identical older
    // exchange — the submitText bridge only fires when the persisted candidate
    // equals the EXACT submitted text.
    const unsent = message('user-1727-def456', 'user', [textPart(displayText)], {
      submitText: 'a brand new question'
    })

    const previous = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      unsent
    ]

    const next = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      persistedTwin,
      persistedReply
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged.some(row => row.id === 'user-1727-def456')).toBe(true)
  })

  it('never dedupes a rowId-bearing optimistic row against a stored row it provably is not', () => {
    // The identity gate stays: submitText equal by coincidence but row ids
    // disagreeing must keep both rows (a genuine repeat of the same caption).
    // The optimistic row's receipt stamped rowId 18730 — NEWER than the stale
    // refreshed page's newest row — so the "receipt proved it saved" gate
    // (rowId <= lastStoredRowId) does not drop it either.
    const acknowledged = message('user-1727-abc123', 'user', [textPart(displayText)], {
      rowId: 18730,
      submitText: transportText
    })

    const previous = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      acknowledged
    ]

    // A DIFFERENT stored user row (rowId 18720) with the same content: the
    // optimistic row's rowId 18715 conflicts, so no dedupe.
    const otherStored = message('9-2-user', 'user', [textPart(transportText)], { rowId: 18720 })

    const next = [
      message('9-0-user', 'user', [textPart('run the suite')], { rowId: 18700 }),
      message('9-1-assistant', 'assistant', [textPart('suite ran')], { pending: false, rowId: 18701 }),
      otherStored,
      persistedReply
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged.some(row => row.id === 'user-1727-abc123')).toBe(true)
    expect(merged.some(row => row.id === '9-2-user')).toBe(true)
  })
})
