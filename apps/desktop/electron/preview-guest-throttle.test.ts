import { describe, expect, it } from 'vitest'

import { applyPreviewGuestAttachPrefs, PREVIEW_GUEST_PARTITION } from './preview-guest-throttle'

describe('applyPreviewGuestAttachPrefs', () => {
  it('keeps the preview guest animating while the Desktop window is in the background', () => {
    const webPreferences: { backgroundThrottling?: boolean; preload?: string } = {}

    applyPreviewGuestAttachPrefs(webPreferences, PREVIEW_GUEST_PARTITION, '/preload.js')

    expect(webPreferences.backgroundThrottling).toBe(false)
    expect(webPreferences.preload).toBe('/preload.js')
  })

  it('leaves every other webview alone', () => {
    const webPreferences: { backgroundThrottling?: boolean; preload?: string } = {}

    applyPreviewGuestAttachPrefs(webPreferences, 'persist:other', '/preload.js')

    expect(webPreferences).toEqual({})
  })
})
