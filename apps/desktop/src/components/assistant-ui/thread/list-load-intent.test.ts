import { describe, expect, it } from 'vitest'

import { shouldApplyLoadIntent } from './list'

/**
 * The settled-load intent (bottom, or a remembered reading distance) is applied in a
 * commit through `applyRestoreRef`. When the reader moved the viewport themselves, that
 * write re-pins them to the target — measured live as a 1982 → 8649 px jump on a settled
 * transcript change in a long-lived session (#132776). A reader who owns the viewport
 * must keep it, and that ownership has to be read from the session-scoped store as well:
 * a refresh re-creates the transcript, which drops the component-local ref, and the
 * re-pin was measured doing 0 → 14131 px right after such a remount.
 */
const base = { paneVisible: true, readerOwnsViewport: false, scrolledUpInStore: false }

describe('shouldApplyLoadIntent', () => {
  it('refuses the intent while the reader owns the viewport', () => {
    expect(
      shouldApplyLoadIntent({ ...base, readerOwnsViewport: true, restoreFromBottom: null, liveKind: 'offset' })
    ).toBe(false)

    expect(
      shouldApplyLoadIntent({ ...base, readerOwnsViewport: true, restoreFromBottom: 500, liveKind: 'bottom' })
    ).toBe(false)
  })

  it('refuses the intent for a reader the store still remembers after a remount', () => {
    // The ref goes with the re-created transcript; the session-scoped store does not.
    expect(
      shouldApplyLoadIntent({ ...base, scrolledUpInStore: true, restoreFromBottom: null, liveKind: 'offset' })
    ).toBe(false)

    expect(
      shouldApplyLoadIntent({ ...base, scrolledUpInStore: true, restoreFromBottom: 14743, liveKind: 'bottom' })
    ).toBe(false)
  })

  it('still applies the intent for a reader parked at the bottom', () => {
    expect(shouldApplyLoadIntent({ ...base, restoreFromBottom: null, liveKind: 'offset' })).toBe(true)

    expect(shouldApplyLoadIntent({ ...base, restoreFromBottom: 500, liveKind: 'bottom' })).toBe(true)
  })

  it('keeps a foreign offset unapplied, even without an owner', () => {
    expect(shouldApplyLoadIntent({ ...base, restoreFromBottom: 500, liveKind: 'offset' })).toBe(false)
  })

  it('never applies into a hidden pane', () => {
    expect(
      shouldApplyLoadIntent({ ...base, paneVisible: false, restoreFromBottom: null, liveKind: 'bottom' })
    ).toBe(false)
  })
})
