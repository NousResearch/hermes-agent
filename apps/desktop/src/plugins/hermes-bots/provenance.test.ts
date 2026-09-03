/**
 * Causal chain recovery from a durable transcript.
 *
 * The runtime activity feed carries the same `source` edge but is capped at 50
 * events, epoch-scoped and never persisted — so these all work from `log`
 * alone, which is what survives a restart and a gateway sync.
 */

import { describe, expect, it } from 'vitest'

import {
  causedBy,
  causeOf,
  entriesSeenBy,
  hasRecordedCause,
  nextOpenTrail,
  provenanceChain
} from './provenance'
import type { GroupChat, GroupMessage } from './types'

function user(id: string, text: string): GroupMessage {
  return { id, at: 0, from: { kind: 'user', name: 'You' }, text, thread: 'main' }
}

function member(id: string, name: string, text: string, by?: string, saw?: [number, number]): GroupMessage {
  return {
    id,
    at: 0,
    from: { kind: 'member', name },
    text,
    thread: 'main',
    ...(by || saw ? { cause: { ...(by ? { by } : {}), ...(saw ? { saw } : {}) } } : {})
  }
}

/** You → radar → scribe: a two-hop room. */
function room(): GroupChat {
  return {
    log: [
      user('m0', 'summarize the incident'),
      member('m1', 'radar', 'found three candidates', 'You', [0, 1]),
      member('m2', 'scribe', 'written up', 'radar', [1, 2])
    ],
    watermarks: {}
  }
}

describe('entriesSeenBy', () => {
  it('resolves the half-open log range a turn read', () => {
    const r = room()
    const saw = entriesSeenBy(r, r.log[2])

    expect(saw.map(e => e.id)).toEqual(['m1'])
  })

  it('returns nothing for a user send or an unstamped legacy message', () => {
    const r = room()

    expect(entriesSeenBy(r, r.log[0])).toEqual([])
    expect(entriesSeenBy(r, member('x', 'radar', 'legacy'))).toEqual([])
  })

  it('clamps a stale range instead of trusting it', () => {
    // The history trim drops entries from the FRONT, so a range recorded
    // before a trim can point past the end of the log.
    const r = room()
    const stale = member('m9', 'scribe', 'late', 'radar', [5, 99])

    expect(entriesSeenBy({ ...r, log: [...r.log, stale] }, stale)).toEqual([])
  })

  it('survives a malformed range without throwing', () => {
    const r = room()

    for (const saw of [undefined, [] as unknown, [1] as unknown, ['a', 'b'] as unknown]) {
      const bad = { ...member('mx', 'radar', 'bad'), cause: { by: 'You', saw } } as GroupMessage

      expect(entriesSeenBy(r, bad)).toEqual([])
    }
  })
})

describe('causeOf', () => {
  it('resolves the speaker that put this member on turn', () => {
    const r = room()

    expect(causeOf(r, r.log[2])?.id).toBe('m1')
    expect(causeOf(r, r.log[1])?.id).toBe('m0')
  })

  it('returns null at a root — a user send has no cause', () => {
    const r = room()

    expect(causeOf(r, r.log[0])).toBeNull()
  })

  it('picks the newest message from that speaker at or before this one', () => {
    // radar speaks twice; scribe's reply must credit the second, not the first.
    const r: GroupChat = {
      log: [
        user('m0', 'go'),
        member('m1', 'radar', 'first pass', 'You', [0, 1]),
        member('m2', 'radar', 'second pass', 'You', [0, 2]),
        member('m3', 'scribe', 'summary', 'radar', [2, 3])
      ],
      watermarks: {}
    }

    expect(causeOf(r, r.log[3])?.id).toBe('m2')
  })

  it('returns null when the causing message has been trimmed away', () => {
    const orphan = member('m2', 'scribe', 'written up', 'radar', [1, 2])

    expect(causeOf({ log: [orphan], watermarks: {} }, orphan)).toBeNull()
  })
})

describe('provenanceChain', () => {
  it('walks a reply back to the user send that started it', () => {
    const chain = provenanceChain(room(), 'm2')

    expect(chain.map(step => step.message.id)).toEqual(['m2', 'm1', 'm0'])
    expect(chain[0].triggeredBy).toBe('radar')
    expect(chain[1].triggeredBy).toBe('You')
    expect(chain[2].triggeredBy).toBeUndefined()
  })

  it('reports what each hop read', () => {
    const chain = provenanceChain(room(), 'm2')

    expect(chain[0].readCount).toBe(1)
    expect(chain[0].saw.map(e => e.id)).toEqual(['m1'])
    expect(chain[2].readCount).toBe(0)
  })

  it('ends at a user send, and that root is identifiable', () => {
    const chain = provenanceChain(room(), 'm2')
    const root = chain[chain.length - 1]

    expect(root.message.from.kind).toBe('user')
    expect(root.triggeredBy).toBeUndefined()
  })

  it('returns an empty chain for an unknown message id', () => {
    expect(provenanceChain(room(), 'nope')).toEqual([])
  })

  it('stops instead of looping when a trimmed log makes two entries resolve to each other', () => {
    const r: GroupChat = {
      log: [member('a', 'radar', 'one', 'scribe', [0, 1]), member('b', 'scribe', 'two', 'radar', [0, 1])],
      watermarks: {}
    }

    const chain = provenanceChain(r, 'b')

    expect(chain.length).toBeLessThan(5)
    expect(chain.map(step => step.message.id)).toEqual(['b', 'a'])
  })

  it('handles a legacy room with no cause stamps at all', () => {
    const legacy: GroupChat = {
      log: [user('m0', 'go'), member('m1', 'radar', 'reply')],
      watermarks: {}
    }

    const chain = provenanceChain(legacy, 'm1')

    expect(chain.map(step => step.message.id)).toEqual(['m1'])
    expect(chain[0].triggeredBy).toBeUndefined()
  })
})

describe('causedBy', () => {
  it('lists what a message went on to trigger', () => {
    const r = room()

    expect(causedBy(r, 'm1').map(e => e.id)).toEqual(['m2'])
  })

  it('credits the newest message from a speaker, not an older one', () => {
    const r: GroupChat = {
      log: [
        user('m0', 'go'),
        member('m1', 'radar', 'first pass', 'You', [0, 1]),
        member('m2', 'radar', 'second pass', 'You', [0, 2]),
        member('m3', 'scribe', 'summary', 'radar', [2, 3])
      ],
      watermarks: {}
    }

    expect(causedBy(r, 'm2').map(e => e.id)).toEqual(['m3'])
    expect(causedBy(r, 'm1')).toEqual([])
  })

  it('returns nothing for a message that caused nothing, or an unknown id', () => {
    const r = room()

    expect(causedBy(r, 'm2')).toEqual([])
    expect(causedBy(r, 'nope')).toEqual([])
  })
})

describe('hasRecordedCause', () => {
  it('offers the affordance only when there is a chain to show', () => {
    const r = room()

    expect(hasRecordedCause(r.log[2])).toBe(true)
    expect(hasRecordedCause(r.log[1])).toBe(true)
  })

  it('withholds it on a user send and on a pre-provenance legacy message', () => {
    // A "why" button that opens an apology is worse than no button.
    const r = room()

    expect(hasRecordedCause(r.log[0])).toBe(false)
    expect(hasRecordedCause(member('x', 'radar', 'legacy'))).toBe(false)
  })

  it('withholds it on an empty cause, and survives null input', () => {
    const empty = { ...member('y', 'radar', 'hi'), cause: {} } as GroupMessage

    expect(hasRecordedCause(empty)).toBe(false)
    expect(hasRecordedCause(null)).toBe(false)
    expect(hasRecordedCause(undefined)).toBe(false)
  })
})

describe('nextOpenTrail', () => {
  it('opens a trail, and clicking the same one closes it', () => {
    expect(nextOpenTrail(null, 'a')).toBe('a')
    expect(nextOpenTrail('a', 'a')).toBeNull()
  })

  it('moves the trail rather than opening a second', () => {
    expect(nextOpenTrail('a', 'b')).toBe('b')
  })
})
