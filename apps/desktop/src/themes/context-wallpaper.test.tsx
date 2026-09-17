import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $chatFontFamily } from './chat-font'
import { previewWallpaperThemePalette, restoreWallpaperThemePreview, ThemeProvider } from './context'

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

afterEach(() => {
  cleanup()
  $chatFontFamily.set('')
  vi.restoreAllMocks()
})

it('preserves the current chat font during transient wallpaper preview and restore', () => {
  $chatFontFamily.set('Atkinson Hyperlegible')
  render(
    <ThemeProvider>
      <div />
    </ThemeProvider>
  )

  const originalPrimary = cssVar('--theme-primary')
  const originalFont = cssVar('--dt-font-sans')
  const storageWrite = vi.spyOn(Storage.prototype, 'setItem')

  act(() => previewWallpaperThemePalette({ accent: '#d45b9e', dominant: '#72808f' }))
  expect(cssVar('--theme-primary')).toBe('#d45b9e')
  expect(cssVar('--dt-font-sans')).toBe(originalFont)
  expect(originalFont).toContain('Atkinson Hyperlegible')
  expect(storageWrite).not.toHaveBeenCalled()

  act(() => $chatFontFamily.set('Lexend'))
  const updatedFont = cssVar('--dt-font-sans')
  storageWrite.mockClear()

  act(() => previewWallpaperThemePalette({ accent: '#2468ac', dominant: '#465768' }))
  expect(cssVar('--dt-font-sans')).toBe(updatedFont)
  expect(updatedFont).toContain('Lexend')
  act(() => restoreWallpaperThemePreview())
  expect(cssVar('--dt-font-sans')).toBe(updatedFont)
  expect(cssVar('--theme-primary')).toBe(originalPrimary)
  expect(storageWrite).not.toHaveBeenCalled()
})
