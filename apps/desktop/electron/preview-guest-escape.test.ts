import { describe, expect, it } from 'vitest'

import { hasClosePreviewFlag, previewGuestInputAction } from './preview-guest-escape'

describe('previewGuestInputAction', () => {
  it('exits fullscreen on Escape only while the host window is fullscreen', () => {
    expect(previewGuestInputAction({ key: 'Escape', type: 'keyDown' }, true)).toBe('exit-fullscreen')
    // A pane in normal mode keeps passing Esc through to the page.
    expect(previewGuestInputAction({ key: 'Escape', type: 'keyDown' }, false)).toBeNull()
  })

  it('closes the preview on Ctrl+Shift+W regardless of fullscreen state', () => {
    expect(previewGuestInputAction({ control: true, shift: true, key: 'w', type: 'keyDown' }, true)).toBe(
      'close-preview'
    )
    expect(previewGuestInputAction({ control: true, shift: true, key: 'w', type: 'keyDown' }, false)).toBe(
      'close-preview'
    )
  })

  it('also accepts the Cmd+Shift+W (macOS) form', () => {
    expect(previewGuestInputAction({ meta: true, shift: true, key: 'W', type: 'keyDown' }, true)).toBe('close-preview')
  })

  it('leaves every other key for the page', () => {
    expect(previewGuestInputAction({ key: 'a', type: 'keyDown' }, true)).toBeNull()
    // Bare W / Ctrl+W without Shift is not the close chord.
    expect(previewGuestInputAction({ control: true, key: 'w', type: 'keyDown' }, true)).toBeNull()
    // Shift+W without the modifier is typing, not closing.
    expect(previewGuestInputAction({ shift: true, key: 'W', type: 'keyDown' }, true)).toBeNull()
  })

  it('ignores key releases and auto-repeats', () => {
    expect(previewGuestInputAction({ key: 'Escape', type: 'keyUp' }, true)).toBeNull()
    expect(previewGuestInputAction({ isAutoRepeat: true, key: 'Escape', type: 'keyDown' }, true)).toBeNull()
  })
})

describe('hasClosePreviewFlag', () => {
  it('detects the flag in both forms', () => {
    expect(hasClosePreviewFlag(['Hermes', '--close-preview'])).toBe(true)
    expect(hasClosePreviewFlag(['Hermes', '--close-preview=1'])).toBe(true)
  })

  it('ignores other argv and empty argv', () => {
    expect(hasClosePreviewFlag(['Hermes'])).toBe(false)
    expect(hasClosePreviewFlag(['--close'])).toBe(false)
    expect(hasClosePreviewFlag([])).toBe(false)
  })
})
