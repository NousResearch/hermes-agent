import { describe, expect, it } from 'vitest'

import { firstVisibleToolIndex, TOOL_TURN_PAGE_SIZE, type TurnToolRef } from './turn-pagination'

const tools = (count: number): TurnToolRef[] =>
  Array.from({ length: count }, (_, index) => ({ key: `tool-${index}`, messageId: 'message', partIndex: index }))

describe('firstVisibleToolIndex', () => {
  it.each([
    [0, 0],
    [1, 0],
    [TOOL_TURN_PAGE_SIZE - 1, 0],
    [TOOL_TURN_PAGE_SIZE, 0],
    [TOOL_TURN_PAGE_SIZE + 1, 1],
    [TOOL_TURN_PAGE_SIZE * 2, TOOL_TURN_PAGE_SIZE],
    [TOOL_TURN_PAGE_SIZE * 2 + 1, TOOL_TURN_PAGE_SIZE + 1]
  ])('keeps exactly one bounded tail for %i tool calls', (count, expected) => {
    expect(firstVisibleToolIndex(tools(count), null, false)).toBe(expected)
  })

  it('keeps the oldest revealed tool stable when newer calls append', () => {
    expect(firstVisibleToolIndex(tools(41), 'tool-10', true)).toBe(10)
  })

  it('falls back to the bounded tail when the saved tool no longer exists', () => {
    expect(firstVisibleToolIndex(tools(41), 'removed-tool', true)).toBe(21)
  })
})
