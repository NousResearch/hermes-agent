import { describe, expect, it } from 'vitest'

import { advertisesRoomMentionAliases, roomMentionTag } from './canonical-group-mentions'

// Hosted rooms keep the immutable `handle` a member was seated with; a gateway that
// advertises `room_mention_aliases_v1` also accepts the frozen display name's
// normalized forms when exactly one member answers to them. The composer inserts the
// friendly tag only on such an owner, and otherwise keeps inserting the handle every
// gateway has always resolved.

const member = (handle: string, display_name?: string) => ({ handle, display_name })

describe('advertisesRoomMentionAliases', () => {
  it('reads the feature from a capability answer and from a bare feature list', () => {
    expect(advertisesRoomMentionAliases({ features: ['room_identity', 'room_mention_aliases_v1'] })).toBe(true)
    expect(advertisesRoomMentionAliases(['room_mention_aliases_v1'])).toBe(true)
    expect(advertisesRoomMentionAliases(new Set(['room_mention_aliases_v1']))).toBe(true)
  })

  it('is false when the owner does not advertise it, or the answer is not a list', () => {
    expect(advertisesRoomMentionAliases({ features: ['room_identity'] })).toBe(false)
    expect(advertisesRoomMentionAliases(new Set(['room_identity']))).toBe(false)
    expect(advertisesRoomMentionAliases({ methods: ['groups.create'] })).toBe(false)
    expect(advertisesRoomMentionAliases({ features: 'room_mention_aliases_v1' })).toBe(false)
    expect(advertisesRoomMentionAliases(undefined)).toBe(false)
    expect(advertisesRoomMentionAliases(null)).toBe(false)
  })
})

describe('roomMentionTag', () => {
  const primary = member('default-long', 'Scout')
  const second = member('builder')

  it('inserts the frozen display name’s slug on a gateway that accepts aliases', () => {
    expect(roomMentionTag(primary, [primary, second], true)).toBe('scout')
  })

  it('keeps the immutable handle on an owner that does not advertise aliases', () => {
    expect(roomMentionTag(primary, [primary, second], false)).toBe('default-long')
  })

  it('keeps the device suffix of a peer’s frozen name instead of stripping it', () => {
    const peer = member('scout-homelab', 'Scout · Homelab')
    expect(roomMentionTag(peer, [peer, second], true)).toBe('scout-homelab')
  })

  it('never derives a reserved broadcast word from a display name', () => {
    for (const reserved of ['all', 'everyone', 'user', 'default', 'hermes']) {
      const bot = member('default-long', reserved)
      expect(roomMentionTag(bot, [bot, second], true)).toBe('default-long')
    }
  })

  it('falls back to the handle when two members share the alias form', () => {
    const left = member('alpha', 'Scout')
    const right = member('beta', 'Scout')
    expect(roomMentionTag(left, [left, right], true)).toBe('alpha')
    expect(roomMentionTag(right, [left, right], true)).toBe('beta')
  })

  it('lets an exact handle win over another member’s display name', () => {
    const exact = member('scout')
    const renamed = member('default-long', 'Scout')
    expect(roomMentionTag(exact, [exact, renamed], true)).toBe('scout')
    expect(roomMentionTag(renamed, [exact, renamed], true)).toBe('default-long')
  })

  it('does not collide two members whose names normalize to the same form', () => {
    const spaced = member('alpha', 'Scout One')
    const slugged = member('beta', 'Scout.One')
    expect(roomMentionTag(spaced, [spaced, slugged], true)).toBe('alpha')
    expect(roomMentionTag(slugged, [spaced, slugged], true)).toBe('beta')
  })
})

describe('exact handles that are reserved words', () => {
  // A reserved display name derives no alias, so the canonical handle is the only safe tag.
  const bot = member('default', 'Default')
  const hermes = member('hermes')
  const builder = member('builder', 'Builder')

  it('still inserts the member’s canonical handle', () => {
    expect(roomMentionTag(bot, [bot, builder], true)).toBe('default')
    expect(roomMentionTag(hermes, [hermes, builder], true)).toBe('hermes')
  })

  it('never lets a peer’s name capture that handle', () => {
    const captured = member('ops', 'Default')
    expect(roomMentionTag(bot, [bot, captured], true)).toBe('default')
    expect(roomMentionTag(captured, [bot, captured], true)).toBe('ops')
  })
})
