import { describe, expect, it } from 'vitest'

import { type ChatMessage, textPart } from '@/lib/chat-messages'

import { rebindSurvivorRowIds, survivorRowIdsFrom, truncateSubmitParams } from './rewind'

describe('truncateSubmitParams', () => {
  it('omits truncation fields when no ordinal is set', () => {
    expect(truncateSubmitParams(undefined)).toEqual({})
  })

  it('confirms intent for every ordinal, and the empty edge only for ordinal 0', () => {
    expect(truncateSubmitParams(0)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 0,
      confirm_empty_truncate: true
    })
    expect(truncateSubmitParams(1)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 1
    })
  })

  it('never sends an ordinal without the intent flag the gateway requires', () => {
    // The gateway drops history only for a submit that declares itself a
    // rewind. An ordinal built here without confirm_truncate would be refused
    // (and, on an older gateway, would truncate silently).
    for (const ordinal of [0, 1, 2, 7]) {
      const params = truncateSubmitParams(ordinal)

      expect(params.truncate_before_user_ordinal).toBe(ordinal)
      expect(params.confirm_truncate).toBe(true)
    }
  })

  it('includes truncate_before_row_id when passed', () => {
    expect(truncateSubmitParams(1, 'msg-123', 456)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 1,
      truncate_before_message_id: 'msg-123',
      truncate_before_row_id: 456
    })
    expect(truncateSubmitParams(undefined, undefined, 456)).toEqual({
      confirm_truncate: true,
      truncate_before_row_id: 456
    })
  })

  it('drops renderer-synthetic message ids but keeps durable row ids', () => {
    // chat-messages.ts: `${timestamp}-${index}-${role}`
    expect(truncateSubmitParams(1, '1723456789-0-user', 456)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 1,
      truncate_before_row_id: 456
    })
    expect(truncateSubmitParams(0, 'user-1723456789-0', undefined)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 0,
      confirm_empty_truncate: true
    })
  })
})

describe('survivorRowIdsFrom', () => {
  it('returns undefined when the field is absent or not an array (older gateway)', () => {
    expect(survivorRowIdsFrom(undefined)).toBeUndefined()
    expect(survivorRowIdsFrom({ status: 'streaming' })).toBeUndefined()
    expect(survivorRowIdsFrom({ survivor_user_row_ids: 'nope' })).toBeUndefined()
  })

  it('keeps integer ids and nulls anything else', () => {
    expect(survivorRowIdsFrom({ survivor_user_row_ids: [7, null, 9.5, '11', 12] })).toEqual([7, null, null, null, 12])
  })
})

describe('rebindSurvivorRowIds', () => {
  const user = (id: string, rowId?: number, hidden?: boolean): ChatMessage => ({
    id,
    role: 'user',
    parts: [textPart(`text ${id}`)],
    ...(rowId !== undefined ? { rowId } : {}),
    ...(hidden ? { hidden } : {})
  })

  const assistant = (id: string, rowId?: number): ChatMessage => ({
    id,
    role: 'assistant',
    parts: [textPart(`reply ${id}`)],
    ...(rowId !== undefined ? { rowId } : {})
  })

  it('rebinds surviving visible user turns positionally and clears the resubmitted turn', () => {
    // Post-rewind state: two survivors + the resubmitted turn (stale rowId 5).
    const messages = [user('u0', 1), assistant('a0', 2), user('u1', 3), assistant('a1', 4), user('u2', 5)]
    const rebound = rebindSurvivorRowIds(messages, [7, 9])

    expect(rebound[0].rowId).toBe(7)
    expect(rebound[2].rowId).toBe(9)
    // Resubmitted turn is past the survivor list — its durable id doesn't
    // exist yet, and keeping the stale one would 4018 the next rewind.
    expect(rebound[4].rowId).toBeUndefined()
    // Assistant rows are untouched (only user turns are rewind targets).
    expect(rebound[1].rowId).toBe(2)
  })

  it('clears the cached id for null entries instead of keeping a stale one', () => {
    const messages = [user('u0', 1), user('u1', 3)]
    const rebound = rebindSurvivorRowIds(messages, [null, 9])

    expect(rebound[0].rowId).toBeUndefined()
    expect(rebound[1].rowId).toBe(9)
  })

  it('skips hidden user turns — same visible-user filter as the ordinal math', () => {
    const messages = [user('u0', 1), user('hidden', 2, true), user('u1', 3)]
    const rebound = rebindSurvivorRowIds(messages, [7, 9])

    expect(rebound[0].rowId).toBe(7)
    expect(rebound[1].rowId).toBe(2) // hidden: untouched
    expect(rebound[2].rowId).toBe(9)
  })

  it('preserves object identity when nothing changes', () => {
    const messages = [user('u0', 7)]

    expect(rebindSurvivorRowIds(messages, [7])[0]).toBe(messages[0])
  })
})
