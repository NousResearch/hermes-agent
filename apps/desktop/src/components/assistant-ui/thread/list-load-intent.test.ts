import { describe, expect, it } from 'vitest'

import { shouldApplyLoadIntent } from './list'

/**
 * The settled-load intent (bottom, or a remembered reading distance) is applied in a
 * commit through `applyRestoreRef`. When the reader moved the viewport themselves, that
 * write re-pins them to the target — measured live as a 1982 → 8649 px jump on a settled
 * transcript change in a long-lived session (#132776). A reader who owns the viewport
 * must keep it.
 */
describe('shouldApplyLoadIntent', () => {
  it('refuses the intent while the reader owns the viewport', () => {
    expect(
      shouldApplyLoadIntent({
        paneVisible: true,
        readerOwnsViewport: true,
        restoreFromBottom: null,
        liveKind: 'offset'
      })
    ).toBe(false)

    expect(
      shouldApplyLoadIntent({
        paneVisible: true,
        readerOwnsViewport: true,
        restoreFromBottom: 500,
        liveKind: 'bottom'
      })
    ).toBe(false)
  })

  it('still applies the intent for a reader parked at the bottom', () => {
    expect(
      shouldApplyLoadIntent({
        paneVisible: true,
        readerOwnsViewport: false,
        restoreFromBottom: null,
        liveKind: 'offset'
      })
    ).toBe(true)

    expect(
      shouldApplyLoadIntent({
        paneVisible: true,
        readerOwnsViewport: false,
        restoreFromBottom: 500,
        liveKind: 'bottom'
      })
    ).toBe(true)
  })

  it('keeps a foreign offset unapplied, even without an owner', () => {
    expect(
      shouldApplyLoadIntent({
        paneVisible: true,
        readerOwnsViewport: false,
        restoreFromBottom: 500,
        liveKind: 'offset'
      })
    ).toBe(false)
  })

  it('never applies into a hidden pane', () => {
    expect(
      shouldApplyLoadIntent({
        paneVisible: false,
        readerOwnsViewport: false,
        restoreFromBottom: null,
        liveKind: 'bottom'
      })
    ).toBe(false)
  })
})
