import { describe, expect, it } from 'vitest'

import { extensionForMimeType, MEDIA_MIME_TYPES, mimeTypeForPath } from './media-mime'

// The composer's upload pipeline forwards the data-URL MIME from
// `mimeTypeForPath` verbatim, and the backend's `_decode_chat_image_upload`
// rejects any non-`image/*` MIME BEFORE inspecting the bytes — so a missing
// table entry fails the send even when the backend accepts the format
// (iPhone photos default to HEIC on macOS).
describe('mimeTypeForPath', () => {
  it('labels HEIC/HEIF/AVIF as image MIMEs', () => {
    expect(mimeTypeForPath('/Users/u/Downloads/IMG_2968.HEIC')).toBe('image/heic')
    expect(mimeTypeForPath('/Users/u/Downloads/photo.heic')).toBe('image/heic')
    expect(mimeTypeForPath('/Users/u/Downloads/photo.heif')).toBe('image/heif')
    expect(mimeTypeForPath('/Users/u/Downloads/photo.avif')).toBe('image/avif')
  })

  it('keeps the existing raster and media mappings', () => {
    expect(mimeTypeForPath('/tmp/shot.png')).toBe('image/png')
    expect(mimeTypeForPath('/tmp/shot.jpg')).toBe('image/jpeg')
    expect(mimeTypeForPath('/tmp/clip.mp4')).toBe('video/mp4')
    expect(mimeTypeForPath('/tmp/unknown.xyz')).toBe('application/octet-stream')
    expect(mimeTypeForPath('')).toBe('application/octet-stream')
  })
})

// Persisted blobs take their extension from the MIME (`|| '.png'` fallback);
// an unknown image MIME would store HEIC bytes under a lying `.png` name.
describe('extensionForMimeType', () => {
  it('maps ISO-BMFF image MIMEs to their real extensions', () => {
    expect(extensionForMimeType('image/heic')).toBe('.heic')
    expect(extensionForMimeType('image/heif')).toBe('.heif')
    expect(extensionForMimeType('image/avif')).toBe('.avif')
  })

  it('keeps the existing image mappings and the empty fallback', () => {
    expect(extensionForMimeType('image/png')).toBe('.png')
    expect(extensionForMimeType('image/jpeg')).toBe('.jpg')
    expect(extensionForMimeType('image/webp')).toBe('.webp')
    expect(extensionForMimeType('image/svg+xml')).toBe('.svg')
    expect(extensionForMimeType('application/octet-stream')).toBe('')
    expect(extensionForMimeType('')).toBe('')
  })

  it('ignores MIME parameters', () => {
    expect(extensionForMimeType('image/heic; charset=binary')).toBe('.heic')
  })

  it('keeps the path and extension tables consistent for images', () => {
    for (const [ext, mime] of Object.entries(MEDIA_MIME_TYPES)) {
      if (!mime.startsWith('image/')) {continue}

      // Both .jpg and .jpeg map to image/jpeg; the reverse table is canonical (.jpg).
      if (ext === '.jpeg') {continue}
      expect(extensionForMimeType(mime)).toBe(ext)
    }
  })
})
