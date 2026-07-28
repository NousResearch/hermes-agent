import { afterEach, describe, expect, it, vi } from 'vitest'

// `IS_MAC` is resolved once at module load from `navigator`, so each platform
// case overrides the platform and re-imports the module fresh.
async function loadCombo(platform: string) {
  Object.defineProperty(window.navigator, 'platform', { value: platform, configurable: true })
  vi.resetModules()

  return import('./combo')
}

function keydown(init: KeyboardEventInit): KeyboardEvent {
  return new KeyboardEvent('keydown', init)
}

afterEach(() => {
  vi.resetModules()
})

describe('comboFromEvent — ctrl as a distinct modifier on macOS', () => {
  it('reports Control+Tab as "ctrl+tab" on macOS (not Cmd)', async () => {
    const { comboFromEvent } = await loadCombo('MacIntel')

    expect(comboFromEvent(keydown({ code: 'Tab', ctrlKey: true }))).toBe('ctrl+tab')
    expect(comboFromEvent(keydown({ code: 'Tab', ctrlKey: true, shiftKey: true }))).toBe('ctrl+shift+tab')
  })

  it('keeps Cmd as "mod" and distinct from Control on macOS', async () => {
    const { comboFromEvent } = await loadCombo('MacIntel')

    expect(comboFromEvent(keydown({ code: 'KeyK', metaKey: true }))).toBe('mod+k')
    expect(comboFromEvent(keydown({ code: 'KeyK', ctrlKey: true }))).toBe('ctrl+k')
  })

  it('uses layout-aware letters for Cmd shortcuts on non-QWERTY layouts', async () => {
    const { comboFromEvent } = await loadCombo('MacIntel')

    expect(comboFromEvent(keydown({ code: 'KeyI', key: 'c', metaKey: true }))).toBe('mod+c')
    expect(comboFromEvent(keydown({ code: 'KeyI', key: 'C', metaKey: true, shiftKey: true }))).toBe('mod+shift+c')
  })

  it('keeps shifted punctuation anchored to the physical key token', async () => {
    const { comboFromEvent } = await loadCombo('MacIntel')

    expect(comboFromEvent(keydown({ code: 'Slash', key: '?', metaKey: true, shiftKey: true }))).toBe('mod+shift+/')
  })

  it('uses layout-aware punctuation for Cmd shortcuts on non-QWERTY layouts', async () => {
    const { comboFromEvent } = await loadCombo('MacIntel')

    // Dvorak puts "." on the physical QWERTY V key — ⌘. must still reach the
    // command center rather than resolving to the physical token.
    expect(comboFromEvent(keydown({ code: 'KeyV', key: '.', metaKey: true }))).toBe('mod+.')
    expect(comboFromEvent(keydown({ code: 'KeyW', key: ',', metaKey: true }))).toBe('mod+,')
    expect(comboFromEvent(keydown({ code: 'BracketLeft', key: '/', metaKey: true }))).toBe('mod+/')
    // AZERTY reaches "," from the physical QWERTY M key.
    expect(comboFromEvent(keydown({ code: 'KeyM', key: ',', metaKey: true }))).toBe('mod+,')
  })

  it("keeps digits physical so AZERTY's shifted number row still binds", async () => {
    const { comboFromEvent } = await loadCombo('MacIntel')

    // On AZERTY the unshifted "1" key types "&", and "1" only with Shift held.
    // Both must resolve to the same `mod+1` the QWERTY user gets.
    expect(comboFromEvent(keydown({ code: 'Digit1', key: '&', metaKey: true }))).toBe('mod+1')
    expect(comboFromEvent(keydown({ code: 'Digit1', key: '1', metaKey: true }))).toBe('mod+1')
  })

  it('falls back to the physical key for glyphs we do not ship as tokens', async () => {
    const { comboFromEvent } = await loadCombo('MacIntel')

    // Option-modified glyphs, dead keys, and non-Latin scripts are not combo
    // tokens, so the physical code keeps the binding reachable.
    expect(comboFromEvent(keydown({ code: 'KeyK', key: '˚', metaKey: true, altKey: true }))).toBe('mod+alt+k')
    expect(comboFromEvent(keydown({ code: 'KeyN', key: 'Dead', metaKey: true, altKey: true }))).toBe('mod+alt+n')
    expect(comboFromEvent(keydown({ code: 'KeyK', key: 'л', metaKey: true }))).toBe('mod+k')
  })

  it('treats Control as the "mod" accelerator off macOS', async () => {
    const { comboFromEvent } = await loadCombo('Win32')

    expect(comboFromEvent(keydown({ code: 'Tab', ctrlKey: true }))).toBe('mod+tab')
    expect(comboFromEvent(keydown({ code: 'Tab', ctrlKey: true, shiftKey: true }))).toBe('mod+shift+tab')
  })
})

describe('canonicalizeCombo', () => {
  it('leaves "ctrl+…" untouched on macOS', async () => {
    const { canonicalizeCombo } = await loadCombo('MacIntel')

    expect(canonicalizeCombo('ctrl+tab')).toBe('ctrl+tab')
    expect(canonicalizeCombo('ctrl+shift+tab')).toBe('ctrl+shift+tab')
  })

  it('folds "ctrl+…" to "mod+…" off macOS so a real Control press resolves', async () => {
    const { canonicalizeCombo } = await loadCombo('Win32')

    expect(canonicalizeCombo('ctrl+tab')).toBe('mod+tab')
    expect(canonicalizeCombo('ctrl+shift+tab')).toBe('mod+shift+tab')
    // Non-ctrl combos are unchanged.
    expect(canonicalizeCombo('mod+k')).toBe('mod+k')
  })
})

describe('formatCombo — honest Control labels', () => {
  it('renders the Control glyph on macOS', async () => {
    const { formatCombo } = await loadCombo('MacIntel')

    expect(formatCombo('ctrl+tab')).toBe('⌃⇥')
    expect(formatCombo('ctrl+shift+tab')).toBe('⌃⇧⇥')
  })

  it('renders "Ctrl+…" off macOS (base key keeps its glyph)', async () => {
    const { formatCombo } = await loadCombo('Win32')

    expect(formatCombo('ctrl+tab')).toBe('Ctrl+⇥')
    expect(formatCombo('ctrl+shift+tab')).toBe('Ctrl+Shift+⇥')
  })
})

describe('comboAllowedInInput', () => {
  it('lets ctrl combos fire while typing (e.g. ⌃Tab from the composer)', async () => {
    const { comboAllowedInInput } = await loadCombo('MacIntel')

    expect(comboAllowedInInput('ctrl+tab')).toBe(true)
    expect(comboAllowedInInput('ctrl+shift+tab')).toBe(true)
    expect(comboAllowedInInput('mod+k')).toBe(true)
    expect(comboAllowedInInput('shift+x')).toBe(false)
  })
})
