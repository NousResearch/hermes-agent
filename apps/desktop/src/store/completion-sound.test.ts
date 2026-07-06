import { beforeEach, describe, expect, it, vi } from 'vitest'

const storage = new Map<string, string>()

vi.mock('@/lib/storage', () => ({
  persistString: (key: string, value: null | string) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  },
  storedString: (key: string) => storage.get(key) ?? null
}))

async function loadStore() {
  vi.resetModules()

  return await import('./completion-sound')
}

describe('completion sound preference', () => {
  beforeEach(() => {
    storage.clear()
  })

  it('defaults to silent turn-end cues when no preference is stored', async () => {
    const { $completionSoundVariantId, DEFAULT_COMPLETION_SOUND_VARIANT_ID } = await loadStore()

    expect(DEFAULT_COMPLETION_SOUND_VARIANT_ID).toBe(0)
    expect($completionSoundVariantId.get()).toBe(0)
  })

  it('treats 0 as the explicit off variant', async () => {
    const { resolveCompletionSoundVariantId, setCompletionSoundVariantId, $completionSoundVariantId } =
      await loadStore()

    expect(resolveCompletionSoundVariantId(0)).toBe(0)

    setCompletionSoundVariantId(0)

    expect($completionSoundVariantId.get()).toBe(0)
    expect(storage.get('hermes.desktop.completionSoundVariantId')).toBe('0')
  })

  it('falls back to silent cues for invalid stored values', async () => {
    storage.set('hermes.desktop.completionSoundVariantId', 'loud-ish')

    const { $completionSoundVariantId } = await loadStore()

    expect($completionSoundVariantId.get()).toBe(0)
  })

  it('migrates the legacy auto-saved default chime to silent once', async () => {
    storage.set('hermes.desktop.completionSoundVariantId', '1')

    const firstLoad = await loadStore()

    expect(firstLoad.$completionSoundVariantId.get()).toBe(0)
    expect(storage.get('hermes.desktop.completionSoundLegacyDefaultMigrated')).toBe('true')

    storage.set('hermes.desktop.completionSoundVariantId', '1')

    const secondLoad = await loadStore()

    expect(secondLoad.$completionSoundVariantId.get()).toBe(1)
  })
})
