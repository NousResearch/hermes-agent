import { describe, expect, it } from 'vitest'

import { GROUP_REACTION_LIMIT, GROUP_REACTION_USER, groupReactionChips, toggleGroupReaction } from './group-reactions'

const at = (n: number) => 1_800_000_000 + n

describe('toggleGroupReaction', () => {
  it('adds a first reaction', () => {
    expect(toggleGroupReaction(undefined, GROUP_REACTION_USER, '👍', at(1))).toEqual([
      { at: at(1), by: GROUP_REACTION_USER, emoji: '👍' }
    ])
  })

  it('replaces your own reaction when a different emoji lands', () => {
    const once = toggleGroupReaction(undefined, GROUP_REACTION_USER, '👍', at(1))
    expect(toggleGroupReaction(once, GROUP_REACTION_USER, '❤️', at(2))).toEqual([
      { at: at(2), by: GROUP_REACTION_USER, emoji: '❤️' }
    ])
  })

  it('takes the reaction back when the same emoji lands twice', () => {
    const once = toggleGroupReaction(undefined, GROUP_REACTION_USER, '👍', at(1))
    expect(toggleGroupReaction(once, GROUP_REACTION_USER, '👍', at(2))).toEqual([])
  })

  it('keeps one reaction per person — a member reacting does not move the human’s', () => {
    const both = toggleGroupReaction(toggleGroupReaction(undefined, 'ops', '👍', at(1)), GROUP_REACTION_USER, '🎉', at(2))
    expect(both).toHaveLength(2)
    expect(both.map(reaction => reaction.by)).toEqual(['ops', GROUP_REACTION_USER])
  })

  it('caps the list, dropping the oldest reactions', () => {
    let list = toggleGroupReaction(undefined, 'member-0', '👍', at(0))

    for (let i = 1; i <= GROUP_REACTION_LIMIT; i++) {
      list = toggleGroupReaction(list, `member-${i}`, '👍', at(i))
    }

    expect(list).toHaveLength(GROUP_REACTION_LIMIT)
    expect(list.some(reaction => reaction.by === 'member-0')).toBe(false)
    expect(list.at(-1)?.by).toBe(`member-${GROUP_REACTION_LIMIT}`)
  })

  it('does not mutate the list it was given', () => {
    const before = toggleGroupReaction(undefined, 'ops', '👍', at(1))
    const snapshot = structuredClone(before)
    toggleGroupReaction(before, GROUP_REACTION_USER, '👍', at(2))
    expect(before).toEqual(snapshot)
  })
})

describe('groupReactionChips', () => {
  it('collapses one chip per emoji and counts the reactors', () => {
    const reactions = [
      { at: at(1), by: 'ops', emoji: '👍' },
      { at: at(2), by: 'research', emoji: '👍' },
      { at: at(3), by: GROUP_REACTION_USER, emoji: '🎉' }
    ]

    expect(groupReactionChips(reactions)).toEqual([
      { count: 2, emoji: '👍', mine: false },
      { count: 1, emoji: '🎉', mine: true }
    ])
  })

  it('marks a chip as yours when you are among its reactors', () => {
    const reactions = [
      { at: at(1), by: 'ops', emoji: '👍' },
      { at: at(2), by: GROUP_REACTION_USER, emoji: '👍' }
    ]

    expect(groupReactionChips(reactions)).toEqual([{ count: 2, emoji: '👍', mine: true }])
  })

  it('is empty without reactions', () => {
    expect(groupReactionChips(undefined)).toEqual([])
  })
})
