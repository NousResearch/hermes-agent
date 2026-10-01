import { describe, expect, it } from 'vitest'

import {
  shouldReapplyFrozenThreadScrollOffset,
  snapNearBottomScrollOffset,
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

  it('re-pins a settled bottom target only on transcript growth, never on a composer-only resize', () => {
    const previous = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5000 }

    expect(shouldReapplyFrozenThreadScrollOffset(BOTTOM, true, previous, { ...previous, scrollHeight: 5600 })).toBe(
      true
    )
    expect(
      shouldReapplyFrozenThreadScrollOffset(BOTTOM, true, previous, { clearanceHeight: 200, scrollHeight: 5080 })
    ).toBe(false)
    expect(shouldReapplyFrozenThreadScrollOffset(BOTTOM, true, previous, { ...previous, scrollHeight: 4000 })).toBe(
      false
    )
  })

  it('does not re-pin while the session-switch settle loop is still running', () => {
    const previous = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5000 }
    const next = { clearanceHeight: 120, clientHeight: 600, scrollHeight: 5600 }

    expect(shouldReapplyFrozenThreadScrollOffset(OFFSET, false, previous, next)).toBe(false)
  })
})

describe('snapNearBottomScrollOffset', () => {
  it('snaps offsets within one viewport of the tail to the exact bottom', () => {
    expect(snapNearBottomScrollOffset({ fromBottom: 0, kind: 'offset' }, 600)).toEqual(BOTTOM)
    expect(snapNearBottomScrollOffset({ fromBottom: 259.7958984375, kind: 'offset' }, 600)).toEqual(BOTTOM)
    expect(snapNearBottomScrollOffset({ fromBottom: 600, kind: 'offset' }, 600)).toEqual(BOTTOM)
  })

  it('restores deeper reading positions exactly', () => {
    expect(snapNearBottomScrollOffset({ fromBottom: 601, kind: 'offset' }, 600)).toEqual({
      fromBottom: 601,
      kind: 'offset'
    })
    expect(snapNearBottomScrollOffset(OFFSET, 600)).toEqual(OFFSET)
  })

  it('leaves a bottom target untouched', () => {
    expect(snapNearBottomScrollOffset(BOTTOM, 600)).toEqual(BOTTOM)
  })
})
