import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { RENDER_WEIGHT_CHARS } from '@/lib/render-weight'

import {
  advanceSessionTranscriptWindow,
  advanceTranscriptWindow,
  alignToBranchGroup,
  MAX_SESSION_WINDOWS,
  selectTranscriptWindow,
  TRANSCRIPT_WINDOW_BUDGET,
  TRANSCRIPT_WINDOW_MIN_MESSAGES,
  TRANSCRIPT_WINDOW_SLACK
} from './transcript-window'

const message = (id: string, chars: number, branchGroupId?: string): ChatMessage => ({
  id,
  parts: [{ type: 'text', text: 'x'.repeat(chars) }],
  role: id.startsWith('u') ? 'user' : 'assistant',
  ...(branchGroupId ? { branchGroupId } : {})
})

/** Messages of `chars` each, newest last. */
const transcript = (count: number, chars: number): ChatMessage[] =>
  Array.from({ length: count }, (_, i) => message(`m-${i}`, chars))

describe('selectTranscriptWindow', () => {
  it('does not window a transcript that fits the budget', () => {
    const messages = transcript(50, 100)

    const window = selectTranscriptWindow(messages)

    expect(window.windowed).toBe(false)
    // Reference identity preserved — a fresh array would re-render the runtime.
    expect(window.messages).toBe(messages)
  })

  it('windows a HEAVY-but-SHORT transcript that a message-count cap would miss', () => {
    // 40 messages, each a big tool result. Well under any sane count cap, but
    // this is the shape that exhausts the renderer heap (#55191).
    const messages = transcript(40, RENDER_WEIGHT_CHARS * 400)

    const window = selectTranscriptWindow(messages)

    expect(window.windowed).toBe(true)
    expect(window.messages.length).toBeLessThan(messages.length)
    expect(window.messages.at(-1)).toBe(messages.at(-1))
  })

  it('keeps far MORE messages when they are light than when they are heavy', () => {
    // The contract is weight, not count: a message-count cap would treat these
    // two identically. 500 tiny messages are cheaper than 500 tool results, so
    // many more of them survive the same budget.
    const light = selectTranscriptWindow(transcript(500, 20))
    const heavy = selectTranscriptWindow(transcript(500, RENDER_WEIGHT_CHARS * 40))

    expect(light.messages.length).toBeGreaterThan(heavy.messages.length * 10)
  })

  it('leaves a long transcript whole when the whole thing is cheap', () => {
    const messages = transcript(600, 20)

    const window = selectTranscriptWindow(messages)

    expect(window.windowed).toBe(false)
    expect(window.messages).toBe(messages)
  })

  it('keeps a floor of messages when single turns are enormous', () => {
    const messages = transcript(80, RENDER_WEIGHT_CHARS * TRANSCRIPT_WINDOW_BUDGET)

    const window = selectTranscriptWindow(messages)

    expect(window.messages.length).toBeGreaterThanOrEqual(TRANSCRIPT_WINDOW_MIN_MESSAGES)
  })

  it('grows by one budget page per expand and eventually covers everything', () => {
    const messages = transcript(400, RENDER_WEIGHT_CHARS * 40)

    const first = selectTranscriptWindow(messages, 1)
    const second = selectTranscriptWindow(messages, 2)

    expect(first.windowed).toBe(true)
    expect(second.messages.length).toBeGreaterThan(first.messages.length)

    let pages = 1
    let window = selectTranscriptWindow(messages, pages)

    while (window.windowed && pages < 100) {
      window = selectTranscriptWindow(messages, ++pages)
    }

    // Paging terminates at the full transcript — never a dead end.
    expect(window.windowed).toBe(false)
    expect(window.messages).toHaveLength(messages.length)
  })

  it('never cuts inside a branch group, so branches keep their fork point', () => {
    const heavy = RENDER_WEIGHT_CHARS * 200

    // A branch group sits right where a weight-only cut would land.
    const messages: ChatMessage[] = [
      ...transcript(20, heavy),
      message('a-branch-1', heavy, 'group-1'),
      message('a-branch-2', heavy, 'group-1'),
      message('a-branch-3', heavy, 'group-1'),
      ...transcript(20, heavy).map(m => ({ ...m, id: `tail-${m.id}` }))
    ]

    for (let pages = 1; pages <= 6; pages++) {
      const kept = selectTranscriptWindow(messages, pages).messages
      const groupMembers = kept.filter(m => m.branchGroupId === 'group-1')

      // Either the whole group survives or none of it does — never a partial
      // group, which would re-parent the surviving branches.
      expect([0, 3]).toContain(groupMembers.length)
    }
  })

  it('handles an empty transcript', () => {
    const messages: ChatMessage[] = []

    expect(selectTranscriptWindow(messages)).toEqual({ messages, windowed: false })
  })
})

describe('advanceTranscriptWindow', () => {
  const heavyChars = RENDER_WEIGHT_CHARS * 40

  it('matches a fresh walk on first call', () => {
    const messages = transcript(400, heavyChars)

    const state = advanceTranscriptWindow(null, messages)

    expect(state.window).toEqual(selectTranscriptWindow(messages))
    expect(state.anchorId).toBe(state.window.messages[0].id)
  })

  it('holds the cut while a streaming tail grows within slack', () => {
    const messages = transcript(400, heavyChars)
    const state = advanceTranscriptWindow(null, messages)

    // Stream: the tail message grows, no new messages. A fresh weight-walk
    // would slide the cut forward; the sticky window must not.
    const grown = [...messages.slice(0, -1), message('m-399', heavyChars * 2)]
    const next = advanceTranscriptWindow(state, grown)

    expect(next.anchorId).toBe(state.anchorId)
    expect(next.window.messages[0].id).toBe(state.window.messages[0].id)
    // Same message set from the anchor on — every existing row keeps its index.
    expect(next.window.messages.length).toBe(state.window.messages.length)
  })

  it('re-cuts once the tail outgrows budget plus slack', () => {
    const messages = transcript(400, heavyChars)
    let state = advanceTranscriptWindow(null, messages)

    // Keep appending. Each step either HOLDS the cut (identical anchor) or
    // RE-CUTS to exactly what a fresh walk would pick — never something in
    // between — and at least one re-cut must happen before the tail has grown
    // by a full slack's worth of weight.
    const appended = [...messages]
    let recuts = 0

    for (let i = 0; i < TRANSCRIPT_WINDOW_SLACK + 1; i++) {
      appended.push(message(`a-new-${i}`, RENDER_WEIGHT_CHARS))
      const prev = state
      state = advanceTranscriptWindow(state, appended)

      if (state.anchorId === prev.anchorId) {
        expect(state.window.messages[0].id).toBe(prev.window.messages[0].id)
      } else {
        recuts++
        expect(state.window).toEqual(selectTranscriptWindow(appended))
      }
    }

    expect(recuts).toBeGreaterThanOrEqual(1)
    // Streaming causes a re-cut per ~half page of content, not one per flush.
    expect(recuts).toBeLessThan(5)
  })

  it('falls back to a fresh walk when the anchor disappears', () => {
    const messages = transcript(400, heavyChars)
    const state = advanceTranscriptWindow(null, messages)

    // Session swap / compression rewrite: disjoint ids.
    const swapped = transcript(300, heavyChars).map(m => ({ ...m, id: `other-${m.id}` }))
    const next = advanceTranscriptWindow(state, swapped)

    expect(next.window).toEqual(selectTranscriptWindow(swapped))
  })

  it('re-walks when pages change so Show earlier still grows the window', () => {
    const messages = transcript(400, heavyChars)
    const one = advanceTranscriptWindow(null, messages, 1)
    const two = advanceTranscriptWindow(one, messages, 2)

    expect(two.window.messages.length).toBeGreaterThan(one.window.messages.length)
  })

  it('stays pass-through (uncut) while the transcript fits the budget', () => {
    const messages = transcript(50, 100)
    const state = advanceTranscriptWindow(null, messages)

    expect(state.anchorId).toBeNull()
    expect(state.window.messages).toBe(messages)

    const grown = [...messages, message('m-next', 100)]
    const next = advanceTranscriptWindow(state, grown)

    // Still uncut: identity of the full array is preserved for the runtime.
    expect(next.window.messages).toBe(grown)
    expect(next.window.windowed).toBe(false)
  })
})

describe('advanceSessionTranscriptWindow', () => {
  const heavyChars = RENDER_WEIGHT_CHARS * 40

  it('matches a fresh walk on first visit', () => {
    const memos = new Map()
    const messages = transcript(400, heavyChars)

    const state = advanceSessionTranscriptWindow(memos, 'session-a', messages)

    expect(state.window).toEqual(selectTranscriptWindow(messages))
    expect(state.anchorId).toBe(state.window.messages[0].id)
  })

  it('returns the SAME windowed slice by reference on a warm re-visit with an unchanged transcript', () => {
    const memos = new Map()
    const sessionA = transcript(400, heavyChars)
    const sessionB = transcript(300, heavyChars).map(m => ({ ...m, id: `b-${m.id}` }))

    // Visit B, then A, then B again — the exact warm-switch shape of #95595.
    const firstB = advanceSessionTranscriptWindow(memos, 'session-b', sessionB)
    advanceSessionTranscriptWindow(memos, 'session-a', sessionA)
    const secondB = advanceSessionTranscriptWindow(memos, 'session-b', sessionB)

    expect(secondB.window.windowed).toBe(true)
    // THE perf guard: same transcript array => same windowed slice reference,
    // so the runtime repository and every row keep their identity.
    expect(secondB.window.messages).toBe(firstB.window.messages)
    expect(secondB.anchorId).toBe(firstB.anchorId)
  })

  it('holds the sticky cut when a re-visited session grew while away', () => {
    const memos = new Map()
    const messages = transcript(400, heavyChars)
    const state = advanceSessionTranscriptWindow(memos, 'session-a', messages)

    // While away, the session streamed a few light turns (within slack).
    const grown = [...messages, ...transcript(10, 100)]
    const next = advanceSessionTranscriptWindow(memos, 'session-a', grown)

    // Sticky cut survived the switch-away: same anchor, no fresh re-walk.
    expect(next.anchorId).toBe(state.anchorId)
    expect(next.window.messages[0].id).toBe(state.window.messages[0].id)
    // The anchored slice now simply includes the 10 new light turns.
    expect(next.window.messages.length).toBe(state.window.messages.length + 10)
  })

  it('falls back to a fresh walk when the anchor vanished while away', () => {
    const memos = new Map()
    const messages = transcript(400, heavyChars)
    advanceSessionTranscriptWindow(memos, 'session-a', messages)

    // Compression rewrite: disjoint ids while the user was elsewhere.
    const rewritten = transcript(300, heavyChars).map(m => ({ ...m, id: `compressed-${m.id}` }))
    const next = advanceSessionTranscriptWindow(memos, 'session-a', rewritten)

    expect(next.window).toEqual(selectTranscriptWindow(rewritten))
  })

  it('re-walks when pages change on re-entry', () => {
    const memos = new Map()
    const messages = transcript(400, heavyChars)

    const one = advanceSessionTranscriptWindow(memos, 'session-a', messages, 1)
    const two = advanceSessionTranscriptWindow(memos, 'session-a', messages, 2)

    expect(two.window.messages.length).toBeGreaterThan(one.window.messages.length)
  })

  it('keeps sessions independent and evicts the oldest memo past the cap', () => {
    const memos = new Map()

    for (let i = 0; i < MAX_SESSION_WINDOWS + 5; i++) {
      advanceSessionTranscriptWindow(memos, `session-${i}`, transcript(400, heavyChars))
    }

    expect(memos.size).toBeLessThanOrEqual(MAX_SESSION_WINDOWS)
    expect(memos.has('session-0')).toBe(false)
    expect(memos.has(`session-${MAX_SESSION_WINDOWS + 4}`)).toBe(true)
  })
})

describe('alignToBranchGroup', () => {
  const messages = [message('u-1', 10), message('a-1', 10, 'g'), message('a-2', 10, 'g'), message('u-2', 10)]

  it('widens a cut that lands mid-group back to the group start', () => {
    expect(alignToBranchGroup(messages, 2)).toBe(1)
  })

  it('leaves a cut on a non-branch message alone', () => {
    expect(alignToBranchGroup(messages, 3)).toBe(3)
  })

  it('clamps out-of-range indices', () => {
    expect(alignToBranchGroup(messages, -5)).toBe(0)
    expect(alignToBranchGroup(messages, 99)).toBe(messages.length)
  })
})
