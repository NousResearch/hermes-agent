// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { ThemeProvider } from '@/themes/context'
import { everforestTheme } from '@/themes/presets'
import { installUserTheme, removeUserTheme } from '@/themes/user-themes'
import { storedThemeVariant } from '@/themes/variants'

import { AppearanceSettings } from './appearance-settings'

const NAME = 'vsc-kanagawa-flavors'
const colorsFor = (background: string) => ({ ...everforestTheme.colors, background })

beforeEach(() => {
  installUserTheme({
    name: NAME,
    label: 'Kanagawa Flavors',
    description: 'VS Code · metaphor.kanagawa-vscode-color-theme',
    colors: colorsFor('#1f1f28'),
    darkColors: colorsFor('#1f1f28'),
    variants: [
      { name: 'wave', label: 'Wave', mode: 'dark', colors: colorsFor('#1f1f28') },
      { name: 'dragon', label: 'Dragon', mode: 'dark', colors: colorsFor('#181616') }
    ]
  })
  // Boot the active family so its card is the one carrying the picker.
  window.localStorage.setItem('hermes-desktop-theme-v2', NAME)
})

afterEach(() => {
  cleanup()
  removeUserTheme(NAME)
  window.localStorage.clear()
})

const renderPage = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ThemeProvider>
        <AppearanceSettings subpage="theme" />
      </ThemeProvider>
    </QueryClientProvider>
  )

describe('AppearanceSettings theme variants', () => {
  it('offers the active family’s flavors and persists the switch', () => {
    renderPage()

    const dragon = screen.getByRole('button', { name: 'Dragon' })
    expect(dragon.getAttribute('aria-pressed')).toBe('false')

    fireEvent.click(dragon)

    expect(screen.getByRole('button', { name: 'Dragon' }).getAttribute('aria-pressed')).toBe('true')
    expect(storedThemeVariant('default', NAME)).toBe('dragon')
  })
})
