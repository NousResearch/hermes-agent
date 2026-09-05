import { describe, expect, it } from 'vitest'

import {
  classifySessionScroll,
  createSessionScrollMemory,
  SESSION_SCROLL_MEMORY_CAP,
  sessionScrollOffsetPlaceable,
  sessionScrollTargetTop
} from './session-scroll-memory'

const metrics = (scrollTop: number, scrollHeight = 1000, clientHeight = 200) => ({
  clientHeight,
  scrollHeight,
  scrollTop
})

describe('classifySessionScroll', () => {
  it('treats the tail as bottom, including a few pixels of slack', () => {
    expect(classifySessionScroll(metrics(800))).toEqual({ kind: 'bottom' })
    expect(classifySessionScroll(metrics(792))).toEqual({ kind: 'bottom' })
  })

  it('records distance-from-bottom when the reader is in history', () => {
    expect(classifySessionScroll(metrics(200))).toEqual({ fromBottom: 800, kind: 'offset' })
  })

  it('does not record a position for a zero-size scroller as history', () => {
    expect(classifySessionScroll(metrics(0, 0, 0))).toEqual({ kind: 'bottom' })
  })
})

describe('sessionScrollTargetTop', () => {
  it('pins bottom sessions to the current tail as content grows', () => {
    const state = classifySessionScroll(metrics(800, 1000, 200))

    expect(sessionScrollTargetTop(state, { clientHeight: 200, scrollHeight: 1000 })).toBe(800)
    expect(sessionScrollTargetTop(state, { clientHeight: 200, scrollHeight: 1600 })).toBe(1400)
  })

  it('keeps a mid-read offset as the transcript grows above the viewport', () => {
    const state = classifySessionScroll(metrics(200, 1000, 200))

    expect(state).toEqual({ fromBottom: 800, kind: 'offset' })
    expect(sessionScrollTargetTop(state, { clientHeight: 200, scrollHeight: 1000 })).toBe(200)
    expect(sessionScrollTargetTop(state, { clientHeight: 200, scrollHeight: 1800 })).toBe(1000)
  })

  it('clamps an offset that is still deeper than the painted height', () => {
    const state = { fromBottom: 8000, kind: 'offset' as const }

    expect(sessionScrollTargetTop(state, { clientHeight: 200, scrollHeight: 300 })).toBe(0)
    expect(sessionScrollOffsetPlaceable(state, 300)).toBe(false)
    expect(sessionScrollOffsetPlaceable(state, 8000)).toBe(true)
  })
})

describe('createSessionScrollMemory', () => {
  it('recalls the last position for a session and ignores empty keys', () => {
    const memory = createSessionScrollMemory()
    const offset = { fromBottom: 400, kind: 'offset' as const }

    memory.remember('session-a', offset)
    memory.remember(null, { kind: 'bottom' })
    memory.remember('', { kind: 'bottom' })

    expect(memory.recall('session-a')).toEqual(offset)
    expect(memory.recall(null)).toBeNull()
    expect(memory.recall('')).toBeNull()
    expect(memory.recall('missing')).toBeNull()
  })

  it('evicts the oldest session once the cap is reached', () => {
    const memory = createSessionScrollMemory(2)

    memory.remember('a', { kind: 'bottom' })
    memory.remember('b', { fromBottom: 10, kind: 'offset' })
    memory.remember('c', { fromBottom: 20, kind: 'offset' })

    expect(memory.recall('a')).toBeNull()
    expect(memory.recall('b')).toEqual({ fromBottom: 10, kind: 'offset' })
    expect(memory.recall('c')).toEqual({ fromBottom: 20, kind: 'offset' })
  })

  it('treats a re-remember as recency so it is not evicted next', () => {
    const memory = createSessionScrollMemory(2)

    memory.remember('a', { kind: 'bottom' })
    memory.remember('b', { kind: 'bottom' })
    memory.remember('a', { fromBottom: 50, kind: 'offset' })
    memory.remember('c', { kind: 'bottom' })

    expect(memory.recall('b')).toBeNull()
    expect(memory.recall('a')).toEqual({ fromBottom: 50, kind: 'offset' })
  })

  it('caps at SESSION_SCROLL_MEMORY_CAP by default', () => {
    const memory = createSessionScrollMemory()

    for (let i = 0; i < SESSION_SCROLL_MEMORY_CAP + 5; i += 1) {
      memory.remember(`s${i}`, { kind: 'bottom' })
    }

    expect(memory.size()).toBe(SESSION_SCROLL_MEMORY_CAP)
    expect(memory.recall('s0')).toBeNull()
    expect(memory.recall(`s${SESSION_SCROLL_MEMORY_CAP + 4}`)).toEqual({ kind: 'bottom' })
  })
})
