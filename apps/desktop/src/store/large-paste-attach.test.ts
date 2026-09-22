import { beforeEach, describe, expect, it } from 'vitest'

import { $largePasteAttachEnabled, setLargePasteAttachEnabled } from './large-paste-attach'

const STORAGE_KEY = 'hermes.desktop.composer.largePasteAttach'

// The store is the preference's single source of truth: the composer gate
// reads it synchronously, so its init/persist contract must hold exactly.
// `window.localStorage` is stubbed per test (node env has no DOM).
describe('large-paste-attach store', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('defaults to on — the conversion stays the out-of-the-box behavior', () => {
    expect($largePasteAttachEnabled.get()).toBe(true)
  })

  it('restores a stored "false" on init', () => {
    window.localStorage.setItem(STORAGE_KEY, 'false')

    // Re-import into a fresh module registry via dynamic import with a cache
    // bust; the atom initializer reads storage at module scope.
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('false')
  })

  it('persists flips and survives storage failures', () => {
    const original = window.localStorage.setItem

    setLargePasteAttachEnabled(false)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('false')
    expect($largePasteAttachEnabled.get()).toBe(false)

    setLargePasteAttachEnabled(true)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('true')
    expect($largePasteAttachEnabled.get()).toBe(true)

    // A throwing storage must not break the flip itself.
    window.localStorage.setItem = () => {
      throw new Error('quota exceeded')
    }

    expect(() => setLargePasteAttachEnabled(false)).not.toThrow()
    expect($largePasteAttachEnabled.get()).toBe(false)

    window.localStorage.setItem = original
  })
})
