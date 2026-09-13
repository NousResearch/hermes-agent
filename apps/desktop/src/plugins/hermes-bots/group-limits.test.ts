import { describe, expect, it } from 'vitest'

import { GROUP_CHAT_DEFAULT_MAX_BOT_TURNS, isValidGroupMaxBotTurns, normalizeGroupMaxBotTurns } from './group-limits'

describe('group reply budget validation', () => {
  it.each([1, 10, 20, 100])('accepts bounded integer %s', value => {
    expect(isValidGroupMaxBotTurns(value)).toBe(true)
    expect(normalizeGroupMaxBotTurns(value)).toBe(value)
  })
  it.each([undefined, null, '', '20', 0, -1, 101, 1.5, NaN, Infinity, {}, true])(
    'defaults malformed or legacy value %s safely',
    value => {
      expect(isValidGroupMaxBotTurns(value)).toBe(false)
      expect(normalizeGroupMaxBotTurns(value)).toBe(GROUP_CHAT_DEFAULT_MAX_BOT_TURNS)
    }
  )
})
