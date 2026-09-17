import { describe, expect, it } from 'vitest'

import {
  shouldReapplyFrozenThreadScrollOffset,
  threadScrollTargetTop,
  threadScrollTranscriptHeight
} from './thread-scroll'

const OFFSET = { fromBottom: 800, kind: 'offset' as const }
const BOTTOM = { kind: 'bottom' as const }

describe('shouldReapplyFrozenThreadScrollOffset', () => {
  it('does not re-pin a frozen offset when only composer clearance / viewport box resized', () => {
    const previous = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5000 }
    const next = { clearanceHeight: 200, clientHeight: 520, scrollHeight: 5080 }

    expect(threadScrollTranscriptHeight(previous)).toBe(threadScrollTranscriptHeight(next))
    expect(shouldReapplyFrozenThreadScrollOffset(OFFSET, true, previous, next)).toBe(false)

    // Blind re-apply (the v0.21.1 post-settle RO) would rewrite scrollTop.
    expect(threadScrollTargetTop(OFFSET, previous)).not.toBe(threadScrollTargetTop(OFFSET, next))
  })

  it('does not rewrite when a settled offset sees a no-op / same-transcript resize', () => {
    const metrics = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5000 }

    expect(shouldReapplyFrozenThreadScrollOffset(OFFSET, true, metrics, metrics)).toBe(false)
  })

  it('re-pins when transcript content height grew (streaming, prepend, markdown)', () => {
    const previous = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5000 }
    const next = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5600 }

    expect(shouldReapplyFrozenThreadScrollOffset(OFFSET, true, previous, next)).toBe(true)
    expect(threadScrollTargetTop(OFFSET, next)).toBe(5600 - 600 - 800)
  })

  it('re-pins a bottom-anchored restore on real growth, but not on a no-op resize (#113842)', () => {
    const previous = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5000 }
    const grew = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5600 }

    // Late async growth (images, highlighting) must still follow the bottom.
    expect(shouldReapplyFrozenThreadScrollOffset(BOTTOM, true, previous, grew)).toBe(true)

    // A composer-clearance-only resize leaves transcript height untouched: the
    // old unconditional `target.kind === 'bottom'` arm rewrote scrollTop here.
    const clearanceOnly = { clearanceHeight: 200, clientHeight: 520, scrollHeight: 5080 }

    expect(shouldReapplyFrozenThreadScrollOffset(BOTTOM, true, previous, clearanceOnly)).toBe(false)
    expect(shouldReapplyFrozenThreadScrollOffset(BOTTOM, true, previous, previous)).toBe(false)
  })

  it('does not re-pin while the session-switch settle loop is still running', () => {
    const previous = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5000 }
    const next = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5600 }

    expect(shouldReapplyFrozenThreadScrollOffset(OFFSET, false, previous, next)).toBe(false)
  })
})
