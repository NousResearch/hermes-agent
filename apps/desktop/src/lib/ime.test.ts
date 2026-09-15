import { describe, expect, it } from 'vitest'

import { IME_COMMIT_GUARD_MS, isImeComposing, isPostCompositionCommitEnter, isSubmitEnter } from './ime'

describe('isImeComposing', () => {
  it('detects composition via nativeEvent.isComposing (React events)', () => {
    expect(isImeComposing({ key: 'Enter', nativeEvent: { isComposing: true } })).toBe(true)
  })

  it('detects composition via isComposing on the event itself (DOM events)', () => {
    expect(isImeComposing({ isComposing: true, key: 'Enter' })).toBe(true)
  })

  it('detects the legacy 229 keyCode on nativeEvent', () => {
    expect(isImeComposing({ key: 'Enter', nativeEvent: { keyCode: 229 } })).toBe(true)
  })

  it('detects the legacy 229 keyCode on a DOM event', () => {
    expect(isImeComposing({ key: 'Enter', keyCode: 229 })).toBe(true)
  })

  it('passes ordinary events', () => {
    expect(isImeComposing({ key: 'Enter', nativeEvent: { isComposing: false, keyCode: 13 } })).toBe(false)
    expect(isImeComposing({ key: 'a', keyCode: 65 })).toBe(false)
  })
})

describe('isSubmitEnter', () => {
  it('accepts a plain Enter', () => {
    expect(isSubmitEnter({ key: 'Enter', nativeEvent: {} })).toBe(true)
  })

  it('rejects Enter during composition', () => {
    expect(isSubmitEnter({ key: 'Enter', nativeEvent: { isComposing: true } })).toBe(false)
  })

  it('rejects the post-compositionend commit Enter still carrying 229', () => {
    expect(isSubmitEnter({ key: 'Enter', nativeEvent: { isComposing: false, keyCode: 229 } })).toBe(false)
  })

  it('rejects non-Enter keys', () => {
    expect(isSubmitEnter({ key: 'Escape', nativeEvent: {} })).toBe(false)
  })
})

describe('isPostCompositionCommitEnter', () => {
  const END = 1_000_000

  it('is false for a field that has never composed', () => {
    expect(isPostCompositionCommitEnter(0, END)).toBe(false)
  })

  it('claims the Enter that lands on the back of a composition end', () => {
    expect(isPostCompositionCommitEnter(END, END)).toBe(true)
    expect(isPostCompositionCommitEnter(END, END + 40)).toBe(true)
    expect(isPostCompositionCommitEnter(END, END + IME_COMMIT_GUARD_MS)).toBe(true)
  })

  it('hands back an Enter that arrives after the window — that one is a send', () => {
    expect(isPostCompositionCommitEnter(END, END + IME_COMMIT_GUARD_MS + 1)).toBe(false)
    expect(isPostCompositionCommitEnter(END, END + 5_000)).toBe(false)
  })

  it('stays inside the window when the clock goes backwards', () => {
    // Date.now() can step back (NTP, sleep/wake). A negative delta means the
    // guard fires rather than a send slipping through on a clock adjustment.
    expect(isPostCompositionCommitEnter(END, END - 50)).toBe(true)
  })

  it('keeps the window far below human send latency', () => {
    // The guard's whole justification is that a commit Enter is a keystroke and
    // a send is a decision. If this ever creeps into human-decision territory
    // the guard would start swallowing deliberate sends.
    expect(IME_COMMIT_GUARD_MS).toBeLessThanOrEqual(300)
  })
})
