import { describe, expect, it, vi } from 'vitest'

import type { BrowserTabId } from '@/store/browser'
import type { RightRailTabId } from '@/store/layout'

import { closeActiveDesktopRailTab, desktopRailHasTabs } from './desktop-rail-state'

describe('desktop rail state', () => {
  it('keeps the preview rail mounted for browser-only tabs', () => {
    expect(
      desktopRailHasTabs({
        browserEnabled: true,
        browserTabCount: 1,
        filePreviewTabCount: 0,
        previewTarget: null
      })
    ).toBe(true)
  })

  it('does not mount the rail for browser-only tabs while Browser is disabled', () => {
    expect(
      desktopRailHasTabs({
        browserEnabled: false,
        browserTabCount: 1,
        filePreviewTabCount: 0,
        previewTarget: null
      })
    ).toBe(false)
  })

  it('closes browser tabs through the browser lifecycle instead of Preview lifecycle', () => {
    const closeBrowserTab = vi.fn()
    const closePreviewTab = vi.fn()

    closeActiveDesktopRailTab('browser:visible' as BrowserTabId, { closeBrowserTab, closePreviewTab })

    expect(closeBrowserTab).toHaveBeenCalledWith('browser:visible')
    expect(closePreviewTab).not.toHaveBeenCalled()
  })

  it('keeps Preview/file close handling for non-browser rail tabs', () => {
    const closeBrowserTab = vi.fn()
    const closePreviewTab = vi.fn()

    closeActiveDesktopRailTab('preview' as RightRailTabId, { closeBrowserTab, closePreviewTab })

    expect(closeBrowserTab).not.toHaveBeenCalled()
    expect(closePreviewTab).toHaveBeenCalled()
  })
})
