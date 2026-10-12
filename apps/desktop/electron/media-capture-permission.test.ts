import { describe, expect, it } from 'vitest'

import { isMediaCapturePermission } from './media-capture-permission'

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
