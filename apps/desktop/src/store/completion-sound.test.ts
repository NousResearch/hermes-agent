import { beforeEach, describe, expect, it, vi } from 'vitest'

import { isSharedPreferenceKey } from '@/lib/ui-preferences/keys'

const VOLUME_KEY = 'hermes.desktop.completionSoundVolume'

const loadStore = () => import('./completion-sound')

describe('completion sound volume', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  describe('resolveCompletionSoundVolume', () => {
    it('clamps a number below the floor to the mute floor', async () => {
      const { resolveCompletionSoundVolume } = await loadStore()

      expect(resolveCompletionSoundVolume(-0.01)).toBe(0)
      expect(resolveCompletionSoundVolume(-1)).toBe(0)
    })

    it('clamps a number above the ceiling to the ceiling', async () => {
      const { resolveCompletionSoundVolume } = await loadStore()

      expect(resolveCompletionSoundVolume(3.01)).toBe(3)
      expect(resolveCompletionSoundVolume(100)).toBe(3)
    })

    it('falls back to the default for non-finite values', async () => {
      const { DEFAULT_COMPLETION_SOUND_VOLUME, resolveCompletionSoundVolume } = await loadStore()

      expect(resolveCompletionSoundVolume(Number.NaN)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
      expect(resolveCompletionSoundVolume(Number.POSITIVE_INFINITY)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
      expect(resolveCompletionSoundVolume(Number.NEGATIVE_INFINITY)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    })

    it('passes through values inside the range, including the mute floor', async () => {
      const { resolveCompletionSoundVolume } = await loadStore()

      expect(resolveCompletionSoundVolume(0)).toBe(0)
      expect(resolveCompletionSoundVolume(0.48)).toBe(0.48)
      expect(resolveCompletionSoundVolume(2.5)).toBe(2.5)
      expect(resolveCompletionSoundVolume(3)).toBe(3)
    })
  })

  describe('$completionSoundVolume', () => {
    it('starts at the default when nothing is stored', async () => {
      const { $completionSoundVolume, DEFAULT_COMPLETION_SOUND_VOLUME } = await loadStore()

      expect($completionSoundVolume.get()).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    })

    it('falls back to the default for a stored value that is not a plain decimal', async () => {
      const { DEFAULT_COMPLETION_SOUND_VOLUME } = await loadStore()

      // Each of these would otherwise be laundered into a real gain: max, mute
      // and 1.5 respectively.
      for (const corrupt of ['3.0.1', '0x10', '1.5abc', '', '   ', 'loud']) {
        window.localStorage.setItem(VOLUME_KEY, corrupt)
        vi.resetModules()

        const reloaded = await loadStore()

        expect(reloaded.$completionSoundVolume.get(), corrupt).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
      }
    })

    it('clamps a stored numeric out-of-range value instead of resetting it', async () => {
      window.localStorage.setItem(VOLUME_KEY, '9')

      const { $completionSoundVolume } = await loadStore()

      expect($completionSoundVolume.get()).toBe(3)
    })

    it('keeps a stored value across a reload, including the mute floor and ceiling', async () => {
      for (const stored of ['0', '1.25', '3']) {
        window.localStorage.setItem(VOLUME_KEY, stored)
        vi.resetModules()

        const reloaded = await loadStore()

        // A mute must survive a restart rather than snapping back to the default.
        expect(reloaded.$completionSoundVolume.get(), stored).toBe(Number(stored))
      }
    })

    it('persists a change and validates it through the setter', async () => {
      const store = await loadStore()

      store.setCompletionSoundVolume(2.4)

      expect(store.$completionSoundVolume.get()).toBe(2.4)
      expect(window.localStorage.getItem(VOLUME_KEY)).toBe('2.4')

      store.setCompletionSoundVolume(42)

      expect(store.$completionSoundVolume.get()).toBe(3)
    })

    it('adopts a volume changed in another window', async () => {
      const store = await loadStore()

      window.localStorage.setItem(VOLUME_KEY, '2')
      window.dispatchEvent(new StorageEvent('storage', { key: VOLUME_KEY }))

      expect(store.$completionSoundVolume.get()).toBe(2)

      // A corrupted value from a peer heals to the default rather than muting.
      window.localStorage.setItem(VOLUME_KEY, '0x10')
      window.dispatchEvent(new StorageEvent('storage', { key: VOLUME_KEY }))

      expect(store.$completionSoundVolume.get()).toBe(store.DEFAULT_COMPLETION_SOUND_VOLUME)
    })

    it('is a machine-level preference, so it follows the user like the preset does', () => {
      expect(isSharedPreferenceKey(VOLUME_KEY)).toBe(true)
      expect(isSharedPreferenceKey('hermes.desktop.completionSoundVariantId')).toBe(true)
    })
  })
})
