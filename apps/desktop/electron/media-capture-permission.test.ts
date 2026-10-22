import { describe, expect, it, vi } from 'vitest'

import { isDisplayMediaRequestAllowed, isMediaCapturePermission, resolveDisplayMediaRequest } from './media-capture-permission'

describe('isMediaCapturePermission', () => {
  it('allows the fullscreen permissions both handlers can receive', () => {
    // The request handler sees 'fullscreen'; the check handler sees
    // 'automatic-fullscreen'. Both must be allowed or the native fullscreen
    // button on <video controls> does nothing.
    expect(isMediaCapturePermission('fullscreen', undefined)).toBe(true)
    expect(isMediaCapturePermission('automatic-fullscreen', undefined)).toBe(true)
  })

  it('allows the capture permissions without metadata (check-handler shape)', () => {
    // The sync check handler carries no mediaTypes metadata and Chromium sends
    // the capture permission strings directly.
    expect(isMediaCapturePermission('audioCapture', undefined)).toBe(true)
    expect(isMediaCapturePermission('videoCapture', undefined)).toBe(true)
  })

  it('allows a media request with audio or video mediaTypes', () => {
    expect(isMediaCapturePermission('media', { mediaTypes: ['audio'] })).toBe(true)
    expect(isMediaCapturePermission('media', { mediaTypes: ['video'] })).toBe(true)
    expect(isMediaCapturePermission('media', { mediaTypes: ['audio', 'video'] })).toBe(true)
  })

  it('allows a media request with absent or empty metadata (Windows shape)', () => {
    // Chromium on Windows frequently fires the request with an empty or
    // undefined mediaTypes; a strict check denies it and getUserMedia throws
    // NotAllowedError.
    expect(isMediaCapturePermission('media', undefined)).toBe(true)
    expect(isMediaCapturePermission('media', { mediaTypes: [] })).toBe(true)
  })

  it('denies unrelated permissions and media requests for other types', () => {
    expect(isMediaCapturePermission('geolocation', undefined)).toBe(false)
    expect(isMediaCapturePermission('notifications', undefined)).toBe(false)
    expect(isMediaCapturePermission('media', { mediaTypes: ['unknown'] })).toBe(false)
  })

  it('gives both handlers one policy: check-handler strings match the request handler', () => {
    // Invariant for the shared predicate: every permission string the sync
    // check handler can see must resolve identically to the async request
    // handler's decision for the same string with no metadata.
    for (const permission of ['media', 'audioCapture', 'videoCapture', 'fullscreen', 'automatic-fullscreen'] as const) {
      expect(isMediaCapturePermission(permission, undefined)).toBe(true)
    }
  })
})

describe('isDisplayMediaRequestAllowed', () => {
  it('allows a user-initiated display request from the packaged renderer or dev server', () => {
    expect(isDisplayMediaRequestAllowed('file://', true, 'http://127.0.0.1:5174')).toBe(true)
    expect(isDisplayMediaRequestAllowed('http://127.0.0.1:5174', true, 'http://127.0.0.1:5174')).toBe(true)
  })

  it('denies background and foreign-origin display requests', () => {
    expect(isDisplayMediaRequestAllowed('file://', false, 'http://127.0.0.1:5174')).toBe(false)
    expect(isDisplayMediaRequestAllowed('https://example.com', true, 'http://127.0.0.1:5174')).toBe(false)
    expect(isDisplayMediaRequestAllowed('http://127.0.0.1:5174.evil.example', true, 'http://127.0.0.1:5174')).toBe(false)
  })

  it('returns one screen only for an allowed video request and adds loopback only on Windows', async () => {
    const getSources = vi.fn(async () => [{ id: 'screen:0:0', name: 'Primary display' }])
    const request = {
      audioRequested: true,
      securityOrigin: 'file://',
      userGesture: true,
      videoRequested: true
    }

    await expect(resolveDisplayMediaRequest(request, undefined, 'darwin', getSources)).resolves.toEqual({
      video: { id: 'screen:0:0', name: 'Primary display' }
    })
    await expect(resolveDisplayMediaRequest(request, undefined, 'win32', getSources)).resolves.toEqual({
      audio: 'loopback',
      video: { id: 'screen:0:0', name: 'Primary display' }
    })
  })

  it('does not enumerate a screen for a denied request and denies an unavailable source', async () => {
    const getSources = vi.fn(async () => [])

    await expect(resolveDisplayMediaRequest({
      audioRequested: false,
      securityOrigin: 'https://example.com',
      userGesture: true,
      videoRequested: true
    }, undefined, 'linux', getSources)).resolves.toBeNull()
    expect(getSources).not.toHaveBeenCalled()

    await expect(resolveDisplayMediaRequest({
      audioRequested: false,
      securityOrigin: 'file://',
      userGesture: true,
      videoRequested: true
    }, undefined, 'linux', getSources)).resolves.toBeNull()
  })
})
