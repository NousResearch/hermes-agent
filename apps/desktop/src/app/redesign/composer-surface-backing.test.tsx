import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'
import { type ThemeContextValue, ThemeProvider, useTheme } from '@/themes/context'
import { nousTheme } from '@/themes/presets'
import { installUserTheme, removeUserTheme } from '@/themes/user-themes'

import { ComposerSurfaceBacking } from './composer-surface-backing'

const customName = 'shell-integration-custom'

afterEach(() => {
  cleanup()
  removeUserTheme(customName)
  $shellChoice.set(null)
  localStorage.clear()
})

it('keeps the normal shell across real light, dark and custom skins, with classic rollback', () => {
  $shellChoice.set(null)
  installUserTheme({ ...nousTheme, name: customName, label: 'Custom shell test' })
  let appearance!: ThemeContextValue

  function Surface({ guidedChat = false }) {
    appearance = useTheme()

    return <ComposerSurfaceBacking guidedChat={guidedChat} />
  }

  const view = render(
    <ThemeProvider>
      <Surface />
    </ThemeProvider>
  )
  const edge = () => view.container.querySelector('.portal-composer-edge')

  for (const skin of ['nous', 'portal', customName]) {
    for (const mode of ['light', 'dark'] as const) {
      act(() => {
        appearance.setTheme(skin)
        appearance.setMode(mode)
      })
      expect(appearance.themeName).toBe(skin)
      expect(appearance.resolvedMode).toBe(mode)
      expect(edge()?.getAttribute('aria-hidden')).toBe('true')
    }
  }

  act(() => $shellChoice.set(false))
  expect(edge()).toBeNull()
  expect(appearance.themeName).toBe(customName)
  act(() => $shellChoice.set(true))
  expect(edge()).not.toBeNull()
  view.rerender(
    <ThemeProvider>
      <Surface guidedChat />
    </ThemeProvider>
  )
  expect(edge()).toBeNull()
})
