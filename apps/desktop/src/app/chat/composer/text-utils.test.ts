import { describe, expect, it } from 'vitest'

import { detectTrigger, shouldSkipTriggerRefreshOnKeyUp } from './text-utils'

describe('shouldSkipTriggerRefreshOnKeyUp', () => {
  it('skips the trigger refresh for nav/control keys while a menu is open', () => {
    // These keys are fully handled by the open-trigger keydown branch and
    // never edit text. Refreshing on their keyup resets the highlight to the
    // top (breaking ArrowDown/ArrowUp cycling) and re-opens a menu Escape just
    // closed — the exact bugs this guard prevents.
    for (const key of ['ArrowUp', 'ArrowDown', 'Enter', 'Tab', 'Escape']) {
      expect(shouldSkipTriggerRefreshOnKeyUp(key, true)).toBe(true)
    }
  })

  it('does not skip the refresh when no trigger menu is open', () => {
    for (const key of ['ArrowUp', 'ArrowDown', 'Enter', 'Tab', 'Escape']) {
      expect(shouldSkipTriggerRefreshOnKeyUp(key, false)).toBe(false)
    }
  })

  it('never skips ordinary text-editing keys, so completions still refresh', () => {
    for (const key of ['a', '/', '@', ' ', 'Backspace', 'ArrowLeft', 'ArrowRight']) {
      expect(shouldSkipTriggerRefreshOnKeyUp(key, true)).toBe(false)
    }
  })
})

describe('detectTrigger', () => {
  it('detects a bare slash trigger with an empty query', () => {
    expect(detectTrigger('/')).toEqual({ kind: '/', query: '', tokenLength: 1 })
  })

  it('detects a slash command query', () => {
    expect(detectTrigger('/skill')).toEqual({ kind: '/', query: 'skill', tokenLength: 6 })
  })

  it('detects a bare at-mention trigger with an empty query', () => {
    expect(detectTrigger('@')).toEqual({ kind: '@', query: '', tokenLength: 1 })
  })

  it('detects an at-mention query', () => {
    expect(detectTrigger('@file')).toEqual({ kind: '@', query: 'file', tokenLength: 5 })
  })

  it('returns null for plain text', () => {
    expect(detectTrigger('hello there')).toBeNull()
  })
})
