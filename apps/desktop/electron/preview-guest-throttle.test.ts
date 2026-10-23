import { describe, expect, it, vi } from 'vitest'

import { applyPreviewGuestAttachPrefs, PREVIEW_GUEST_PARTITION } from './preview-guest-throttle'

describe('applyPreviewGuestAttachPrefs', () => {
  it('installs explicit device-denial policy only on the preview partition', () => {
    const previewSession = {
      setPermissionCheckHandler: vi.fn(),
      setPermissionRequestHandler: vi.fn()
    }

    const getSession = vi.fn(() => previewSession)
    // The base implementation ignores this optional policy resolver.
    Reflect.apply(applyPreviewGuestAttachPrefs, undefined, [
      {}, PREVIEW_GUEST_PARTITION, '/preload.js', getSession
    ])
    expect(getSession).toHaveBeenCalledWith(PREVIEW_GUEST_PARTITION)
    const request = previewSession.setPermissionRequestHandler.mock.calls[0][0]
    const check = previewSession.setPermissionCheckHandler.mock.calls[0][0]

    for (const permission of ['media', 'audioCapture', 'videoCapture', 'geolocation']) {
      const decision = vi.fn()
      request(null, permission, decision, {})
      expect(decision).toHaveBeenCalledWith(false)
      expect(check(null, permission, 'https://synthetic.example.test', {})).toBe(false)
    }

    expect(check(null, 'fullscreen', 'https://synthetic.example.test', {})).toBe(true)
    getSession.mockClear()
    Reflect.apply(applyPreviewGuestAttachPrefs, undefined, [{}, 'persist:other', '/preload.js', getSession])
    expect(getSession).not.toHaveBeenCalled()
  })

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
