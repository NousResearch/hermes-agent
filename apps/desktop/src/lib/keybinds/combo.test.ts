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
