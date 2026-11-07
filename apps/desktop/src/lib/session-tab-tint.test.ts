import { describe, expect, it } from 'vitest'

import { isTintColor, TAB_TINT_ACTIVE_MIX, TAB_TINT_IDLE_MIX, tabTintVars } from './session-tab-tint'

describe('isTintColor', () => {
  it('accepts the hex shapes the swatch grid and a plugin can write', () => {
    expect(isTintColor('#abc')).toBe(true)
    expect(isTintColor('#4a9eff')).toBe(true)
    expect(isTintColor('#4A9EFF')).toBe(true)
    expect(isTintColor('#4a9effcc')).toBe(true)
  })

  it('refuses anything that would invalidate the color-mix() declaration', () => {
    // A malformed value inside a custom property is an invalid-at-computed-value
    // declaration: the tab would lose its surface, hover wash AND seam, not just
    // the tint. These are the shapes a hand-edited store or a plugin can carry.
    for (const junk of ['', 'red', 'rgb(74,158,255)', '#12', '#gggggg', 'var(--x)', null, undefined]) {
      expect(isTintColor(junk)).toBe(false)
    }
  })
})

describe('tabTintVars', () => {
  it('sets the idle and active surfaces SEPARATELY', () => {
    // `PaneTab` picks which var to read per state, so one shared `--tab-bg`
    // inline would paint the active tab with the idle mix.
    const vars = tabTintVars('#4a9eff')

    expect(vars?.['--pane-tab-strip-bg']).toBe(
      `color-mix(in srgb, #4a9eff ${TAB_TINT_IDLE_MIX}%, var(--ui-sidebar-surface-background))`
    )
    expect(vars?.['--pane-tab-active-bg']).toBe(
      `color-mix(in srgb, #4a9eff ${TAB_TINT_ACTIVE_MIX}%, var(--ui-editor-surface-background))`
    )
    expect(vars?.['--pane-tab-active-accent']).toBe('#4a9eff')
  })

  it('keeps the mixes faint enough for a strip of eight', () => {
    expect(TAB_TINT_IDLE_MIX).toBeLessThan(TAB_TINT_ACTIVE_MIX)
    expect(TAB_TINT_ACTIVE_MIX).toBeLessThanOrEqual(22)
  })

  it('returns undefined for a colour it cannot paint', () => {
    expect(tabTintVars('nonsense')).toBeUndefined()
    expect(tabTintVars(null)).toBeUndefined()
  })
})
