import { afterEach, describe, expect, it, vi } from 'vitest'

// `IS_MAC` is resolved once when `lib/keybinds/combo` loads from `navigator`, and
// `$comboIndex` runs every default combo through `canonicalizeCombo` (which folds
// `ctrl` → `mod` off macOS). Each platform case overrides the platform and
// re-imports the store fresh so the index is rebuilt for that platform. Mirrors
// `lib/keybinds/combo.test.ts`'s `loadCombo(platform)` harness.
async function loadStore(platform: string) {
  Object.defineProperty(window.navigator, 'platform', { value: platform, configurable: true })
  window.localStorage.clear()
  vi.resetModules()

  return import('./keybinds')
}

afterEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

describe('$comboIndex — session slots survive alongside profile slots', () => {
  it('off macOS: every session slot resolves to its own action, not a profile', async () => {
    // Off macOS a real Ctrl+N keypress is emitted as `mod+N` and profiles own
    // ⌘1…⌘9, so the two families MUST occupy distinct keys — otherwise the
    // first-wins `$comboIndex` guard silently drops the (later) session slots.
    const { $comboIndex } = await loadStore('Win32')
    const index = $comboIndex.get()

    const values = new Set(index.values())

    for (let slot = 1; slot <= 9; slot += 1) {
      // The session slot must be reachable in the index. Before the fix its
      // default `ctrl+N` folded to `mod+N`, collided with `profile.switch.N`
      // (indexed first), and was dropped — so it was absent entirely.
      expect(values.has(`session.slot.${slot}`)).toBe(true)

      // The profile slot must still resolve via ⌘/Ctrl+N, unchanged.
      expect(index.get(`mod+${slot}`)).toBe(`profile.switch.${slot}`)

      // A real Ctrl+N (emitted as `mod+N` off macOS) must NOT resolve to the
      // session slot — that key belongs to the profile switch.
      expect(index.get(`mod+${slot}`)).not.toBe(`session.slot.${slot}`)
    }
  })

  it('on macOS: session slots resolve via literal Control, profiles via Cmd', async () => {
    const { $comboIndex } = await loadStore('MacIntel')
    const index = $comboIndex.get()

    for (let slot = 1; slot <= 9; slot += 1) {
      // Control stays distinct from Cmd on macOS, so both families coexist on
      // their intended chords.
      expect(index.get(`ctrl+${slot}`)).toBe(`session.slot.${slot}`)
      expect(index.get(`mod+${slot}`)).toBe(`profile.switch.${slot}`)
    }
  })
})
