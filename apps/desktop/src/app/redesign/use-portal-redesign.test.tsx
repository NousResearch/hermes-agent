import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { $agentSidebarChoice } from '@/store/simplified-agent-mode'
import { type ThemeContextValue, ThemeProvider, useTheme } from '@/themes/context'
import { nousTheme } from '@/themes/presets'
import { installUserTheme, removeUserTheme } from '@/themes/user-themes'

import { ComposerSurfaceBacking } from './composer-surface-backing'

const customName = 'shell-integration-custom'

afterEach(() => {
  cleanup()
  removeUserTheme(customName)
  $agentSidebarChoice.set(null)
  localStorage.clear()
})

it('keeps the normal shell across real light, dark and custom skins, with classic rollback', () => {
  $agentSidebarChoice.set(null)
  installUserTheme({ ...nousTheme, name: customName, label: 'Custom shell test' })
  let appearance!: ThemeContextValue
  function Surface({ hudMode = false, guidedChat = false }) {
    appearance = useTheme()
    return <ComposerSurfaceBacking guidedChat={guidedChat} hudMode={hudMode} />
  }
  const view = render(<ThemeProvider><Surface /></ThemeProvider>)
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
  act(() => $agentSidebarChoice.set(false))
  expect(edge()).toBeNull()
  expect(appearance.themeName).toBe(customName)
  act(() => $agentSidebarChoice.set(true))
  expect(edge()).not.toBeNull()
  view.rerender(<ThemeProvider><Surface hudMode /></ThemeProvider>)
  expect(edge()).toBeNull()
  view.rerender(<ThemeProvider><Surface guidedChat /></ThemeProvider>)
  expect(edge()).toBeNull()
})
