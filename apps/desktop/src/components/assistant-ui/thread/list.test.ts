import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  buildGroups,
  firstVisibleGroupIndex,
  HIDDEN_TRANSCRIPT_RENDER_BUDGET,
  LIVE_TAIL_MIN_GROUPS,
  LIVE_TAIL_PARTS,
  liveTailStart,
  type MessageGroup,
  resolveThreadScrollTarget,
  subscribeToThreadForeground,
  transcriptPaneBudget
} from './list'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('subscribeToThreadForeground', () => {
  it('reanchors on focus when an active turn keeps document visibility pinned visible', () => {
    const reanchor = vi.fn()

    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      callback(0)

      return 1
    })

    const unsubscribe = subscribeToThreadForeground(() => true, reanchor)

    window.dispatchEvent(new Event('focus'))

    expect(raf).toHaveBeenCalledOnce()
    expect(reanchor).toHaveBeenCalledOnce()
    unsubscribe()
  })

  it('leaves a scrolled-up reader in place when the window focuses', () => {
    const reanchor = vi.fn()
    const raf = vi.spyOn(window, 'requestAnimationFrame')
    const unsubscribe = subscribeToThreadForeground(() => false, reanchor)

    window.dispatchEvent(new Event('focus'))

    expect(raf).not.toHaveBeenCalled()
    expect(reanchor).not.toHaveBeenCalled()
    unsubscribe()
  })

  it('drops a queued reanchor when the reader scrolls away before the frame', () => {
    const frames: FrameRequestCallback[] = []
    let following = true
    const reanchor = vi.fn()

    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      frames.push(callback)

      return 7
    })

    const unsubscribe = subscribeToThreadForeground(() => following, reanchor)

    window.dispatchEvent(new Event('focus'))
    following = false
    frames[0]?.(0)

    expect(reanchor).not.toHaveBeenCalled()
    unsubscribe()
  })

  it('cancels a queued reanchor when its thread unmounts', () => {
    const cancel = vi.spyOn(window, 'cancelAnimationFrame')
    const reanchor = vi.fn()

    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(9)

    const unsubscribe = subscribeToThreadForeground(() => true, reanchor)

    window.dispatchEvent(new Event('focus'))
    unsubscribe()

    expect(cancel).toHaveBeenCalledWith(9)
    expect(reanchor).not.toHaveBeenCalled()
  })
})

// Signature rows are `${index}:${id}:${role}:${weight}` (see the useAuiState
// selector in list.tsx).
const signature = (rows: [string, string, number][]) =>
  rows.map(([id, role, weight], index) => `${index}:${id}:${role}:${weight}`).join('\n')

describe('transcriptPaneBudget', () => {
  it('uses a fixed live-tail budget while hidden instead of charging every mounted transcript', () => {
    expect(transcriptPaneBudget(1, true)).toBe(HIDDEN_TRANSCRIPT_RENDER_BUDGET)
    expect(transcriptPaneBudget(4, true)).toBe(HIDDEN_TRANSCRIPT_RENDER_BUDGET)
    expect(transcriptPaneBudget(1, false)).toBeGreaterThan(HIDDEN_TRANSCRIPT_RENDER_BUDGET)
  })
})

describe('buildGroups', () => {
  it('returns no groups for an empty signature', () => {
    expect(buildGroups('')).toEqual([])
  })

  it('groups a user message with the assistant turn(s) that follow it', () => {
    const groups = buildGroups(
      signature([
        ['u1', 'user', 1],
        ['a1', 'assistant', 4],
        ['a2', 'assistant', 2],
        ['u2', 'user', 1],
        ['a3', 'assistant', 3]
      ])
    )

    expect(groups).toEqual([
      { id: 'u1', indices: [0, 1, 2], kind: 'turn', weight: 7 },
      { id: 'u2', indices: [3, 4], kind: 'turn', weight: 4 }
    ])
  })

  it('keeps leading non-user messages as standalone groups', () => {
    const groups = buildGroups(
      signature([
        ['s1', 'system', 1],
        ['a0', 'assistant', 2],
        ['u1', 'user', 1],
        ['a1', 'assistant', 5]
      ])
    )

    expect(groups).toEqual([
      { id: 's1', index: 0, kind: 'standalone', weight: 1 },
      { id: 'a0', index: 1, kind: 'standalone', weight: 2 },
      { id: 'u1', indices: [2, 3], kind: 'turn', weight: 6 }
    ])
  })

  it('defaults a missing/zero weight to 1', () => {
    const groups = buildGroups('0:a:assistant:0')

    expect(groups).toEqual([{ id: 'a', index: 0, kind: 'standalone', weight: 1 }])
  })
})

describe('resolveThreadScrollTarget', () => {
  const context = (scrollElement: Pick<HTMLElement, 'scrollTop'>) => ({
    contentElement: document.createElement('div'),
    scrollElement: scrollElement as HTMLElement
  })

  it('settles when the browser clamps the requested bottom within half a CSS pixel', () => {
    let actualScrollTop = 0
    let writes = 0

    const scrollElement = {
      get scrollTop() {
        return actualScrollTop
      },
      set scrollTop(value: number) {
        writes += 1
        actualScrollTop = value - 0.125
      }
    }

    const target = 899

    const requested = resolveThreadScrollTarget(target, context(scrollElement))
    scrollElement.scrollTop = requested
    const settled = resolveThreadScrollTarget(target, context(scrollElement))

    expect(requested).toBe(target)
    expect(actualScrollTop).toBe(898.875)
    expect(settled).toBe(actualScrollTop)
    expect(actualScrollTop < settled).toBe(false)
    expect(writes).toBe(1)
  })

  it('keeps following while more than half a CSS pixel remains', () => {
    const scrollElement = { scrollTop: 898.25 }

    expect(resolveThreadScrollTarget(899, context(scrollElement))).toBe(899)
  })

  it('re-arms after streaming content increases the target', () => {
    const scrollElement = { scrollTop: 898.875 }

    expect(resolveThreadScrollTarget(899, context(scrollElement))).toBe(898.875)
    expect(resolveThreadScrollTarget(999, context(scrollElement))).toBe(999)
  })
})

describe('firstVisibleGroupIndex', () => {
  const group = (id: string, weight: number): MessageGroup => ({ id, index: 0, kind: 'standalone', weight })

  it('shows everything when total weight fits the budget', () => {
    const groups = [group('a', 10), group('b', 10), group('c', 10)]

    expect(firstVisibleGroupIndex(groups, 100)).toBe(0)
  })

  it('walks newest-first and hides everything before the turn that meets the budget', () => {
    const groups = [group('old', 50), group('mid', 30), group('new', 30)]

    // newest-first: 30 (new) < 60, +30 (mid) = 60 >= 60 → mid is the first
    // visible group, old is hidden.
    expect(firstVisibleGroupIndex(groups, 60)).toBe(1)
  })

  it('keeps whole turns intact — the turn that crosses the budget stays visible', () => {
    const groups = [group('old', 5), group('huge', 500)]

    expect(firstVisibleGroupIndex(groups, 60)).toBe(1)
  })

  it('returns groups.length for an empty list', () => {
    expect(firstVisibleGroupIndex([], 60)).toBe(0)
  })

  it('keeps a floor of turns visible however heavy they are', () => {
    // Without the floor a session of enormous turns puts "Show earlier" two
    // turns from the bottom, which reads as broken rather than as paging.
    const groups = Array.from({ length: 20 }, (_, i) => group(`g${i}`, 5_000))

    expect(firstVisibleGroupIndex(groups, 600, 8)).toBe(groups.length - 8)
  })

  it('does not force the floor to hide turns the budget already showed', () => {
    const groups = Array.from({ length: 20 }, (_, i) => group(`g${i}`, 1))

    expect(firstVisibleGroupIndex(groups, 600, 8)).toBe(0)
  })
})

describe('liveTailStart', () => {
  const group = (id: string, weight: number): MessageGroup => ({ id, index: 0, kind: 'standalone', weight })

  it('keeps the newest turns rendered until the parts budget is spent', () => {
    // 10 turns x 10 parts. A 40-part tail covers the newest 4-5 turns.
    const groups = Array.from({ length: 10 }, (_, i) => group(`g${i}`, 10))
    const start = liveTailStart(groups)

    expect(start).toBeGreaterThan(0)
    expect(start).toBeLessThan(groups.length)

    // Everything from `start` onward is the live tail...
    const tailParts = groups.slice(start).reduce((sum, g) => sum + g.weight, 0)
    expect(tailParts).toBeGreaterThan(LIVE_TAIL_PARTS)

    // ...and dropping its oldest member puts it back under budget, i.e. the
    // tail is minimal rather than sprawling.
    const withoutOldest = groups.slice(start + 1).reduce((sum, g) => sum + g.weight, 0)
    expect(withoutOldest).toBeLessThanOrEqual(LIVE_TAIL_PARTS)
  })

  it('virtualizes the old bulk of a long agent transcript', () => {
    // The regression this guards: heavy tool turns. A turn-count tail (6) left
    // NOTHING virtualized on transcripts like this, so every Radix overlay open
    // paid a whole-document style recalc.
    const groups = Array.from({ length: 40 }, (_, i) => group(`g${i}`, 120))

    // Only the min-group floor stays rendered; the other 38 turns skip.
    expect(liveTailStart(groups)).toBe(groups.length - LIVE_TAIL_MIN_GROUPS)
  })

  it('never virtualizes below the min-group floor, however heavy the turns', () => {
    const groups = Array.from({ length: 5 }, (_, i) => group(`g${i}`, 10_000))

    expect(liveTailStart(groups)).toBe(groups.length - LIVE_TAIL_MIN_GROUPS)
  })

  it('keeps every turn rendered when the whole transcript fits in the tail', () => {
    const groups = [group('a', 5), group('b', 5), group('c', 5)]

    expect(liveTailStart(groups)).toBe(0)
  })

  it('handles an empty transcript', () => {
    expect(liveTailStart([])).toBe(0)
  })

  it('honors a custom budget', () => {
    const groups = Array.from({ length: 10 }, (_, i) => group(`g${i}`, 1))

    // A 3-part budget would keep 4 turns, but the max-groups ceiling is not hit
    // here, so the parts budget wins.
    expect(liveTailStart(groups, 3)).toBe(6)
  })

  it('never renders more than the old turn-count tail did, on any shape', () => {
    // Guards the one way a parts budget can regress: a long transcript of tiny
    // turns, where walking back 40 parts reaches further than 6 turns would.
    const shapes = [
      Array.from({ length: 40 }, () => 4), // long chat, tiny turns
      Array.from({ length: 40 }, () => 1), // pathological: 1-part turns
      Array.from({ length: 12 }, () => 6),
      [80, 120, 60, 150, 90, 200, 70], // real agent tile
      [30, 45]
    ]

    for (const weights of shapes) {
      const groups = weights.map((weight, i) => group(`g${i}`, weight))
      const rendered = (start: number) => weights.slice(start).reduce((a, b) => a + b, 0)

      const oldStart = Math.max(0, groups.length - 6)

      expect(rendered(liveTailStart(groups))).toBeLessThanOrEqual(rendered(oldStart))
    }
  })
})
