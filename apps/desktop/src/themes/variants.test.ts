// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

import type { DesktopTheme, DesktopThemeColors } from './types'
import { activeVariantName, persistThemeVariant, storedThemeVariant, variantForMode, variantsForMode } from './variants'

const colors = (background: string): DesktopThemeColors => ({ background }) as DesktopThemeColors

const family = (): DesktopTheme => ({
  name: 'vsc-kanagawa-flavors',
  label: 'Kanagawa Flavors',
  description: 'VS Code · metaphor.kanagawa-vscode-color-theme',
  colors: colors('#1f1f28'),
  darkColors: colors('#1f1f28'),
  variants: [
    { name: 'wave', label: 'Wave', mode: 'dark', colors: colors('#1f1f28') },
    { name: 'dragon', label: 'Dragon', mode: 'dark', colors: colors('#181616') },
    { name: 'lotus', label: 'Lotus', mode: 'light', colors: colors('#f2ecbc') }
  ]
})

beforeEach(() => window.localStorage.clear())

describe('theme variant selection', () => {
  it('round-trips a per-profile pick and clears it', () => {
    expect(storedThemeVariant('default', 'vsc-kanagawa-flavors')).toBeNull()

    persistThemeVariant('default', 'vsc-kanagawa-flavors', 'dragon')
    expect(storedThemeVariant('default', 'vsc-kanagawa-flavors')).toBe('dragon')

    // Profiles are islands — a named profile never inherits the default's pick.
    expect(storedThemeVariant('research', 'vsc-kanagawa-flavors')).toBeNull()

    persistThemeVariant('default', 'vsc-kanagawa-flavors', null)
    expect(storedThemeVariant('default', 'vsc-kanagawa-flavors')).toBeNull()
  })

  it('resolves a pick only for its own mode, else the mode’s first option', () => {
    const theme = family()

    expect(activeVariantName(theme, 'dark', 'dragon')).toBe('dragon')
    // 'dragon' is dark, so the light surface ignores it and offers its own first.
    expect(activeVariantName(theme, 'light', 'dragon')).toBe('lotus')
    expect(variantForMode(theme, 'light', 'dragon')?.colors.background).toBe('#f2ecbc')
    expect(variantForMode(theme, 'dark', 'dragon')?.colors.background).toBe('#181616')
  })

  it('offers nothing for a family without variants', () => {
    const plain: DesktopTheme = { ...family(), variants: undefined }

    expect(variantsForMode(plain, 'dark')).toEqual([])
    expect(activeVariantName(plain, 'dark', 'wave')).toBeNull()
    expect(variantForMode(plain, 'dark', 'wave')).toBeUndefined()
  })
})
