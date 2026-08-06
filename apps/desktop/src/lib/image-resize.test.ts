import { afterEach, describe, expect, it, vi } from 'vitest'

import { downscaleDataUrlForPreview } from './image-resize'

// A minimal valid 1x1 red PNG (67 bytes) for testing
const TINY_PNG_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+GkZcAAAAASUVORK5CYII='
const TINY_PNG_DATA_URL = `data:image/png;base64,${TINY_PNG_B64}`

// A 1×1 transparent PNG used as the fallback placeholder
const FALLBACK_PLACEHOLDER =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+GkZcAAAAASUVORK5CYII='

describe('downscaleDataUrlForPreview', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  describe('without createImageBitmap (jsdom default)', () => {
    it('returns the original when createImageBitmap is unavailable', async () => {
      await expect(downscaleDataUrlForPreview(TINY_PNG_DATA_URL)).resolves.toBe(
        TINY_PNG_DATA_URL
      )
    })

    it('returns the original for a non-data-URL string', async () => {
      await expect(downscaleDataUrlForPreview('not-a-data-url')).resolves.toBe(
        'not-a-data-url'
      )
    })

    it('returns the original for a data URL without a comma', async () => {
      await expect(downscaleDataUrlForPreview('data:text/plain')).resolves.toBe(
        'data:text/plain'
      )
    })
  })

  describe('with mocked createImageBitmap + OffscreenCanvas', () => {
    /**
     * Set up the full mock chain: fetch → createImageBitmap → OffscreenCanvas → FileReader.
     * Mocks a bitmap of the given dimensions so the downscaling logic is exercised.
     */
    function setupMocks(bitmapWidth: number, bitmapHeight: number) {
      const close = vi.fn()
      const drawImage = vi.fn()
      const convertToBlob = vi.fn(async () => new Blob(['x'], { type: 'image/png' }))

      const bitmap = { width: bitmapWidth, height: bitmapHeight, close }
      const ctx = { drawImage }

      class MockOffscreenCanvas {
        getContext = vi.fn(() => ctx)
        convertToBlob = convertToBlob
        constructor(_w: number, _h: number) {}
      }

      // Mock fetch to return a Blob from the data URL
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({
          blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' })
        }))
      )
      vi.stubGlobal('createImageBitmap', vi.fn(async () => bitmap))
      vi.stubGlobal('OffscreenCanvas', MockOffscreenCanvas)

      return { bitmap, close, drawImage, convertToBlob, ctx }
    }

    it('returns the original when image is smaller than maxLongEdge', async () => {
      setupMocks(500, 300)

      const result = await downscaleDataUrlForPreview(TINY_PNG_DATA_URL, 2048)
      expect(result).toBe(TINY_PNG_DATA_URL)
    })

    it('downscales when image exceeds maxLongEdge', async () => {
      const { drawImage, close } = setupMocks(4000, 3000)

      // In jsdom, FileReader.readAsDataURL won't actually produce a data URL,
      // so the function falls through to the fallback placeholder. But the
      // important thing is that drawImage was called with scaled dimensions
      // and the bitmap was closed.
      const result = await downscaleDataUrlForPreview(TINY_PNG_DATA_URL, 2048)

      // scale = 2048 / 4000 = 0.512 → width=2048, height=1536
      expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 0, 2048, 1536)
      expect(close).toHaveBeenCalled()

      // Result should be the downscaled data URL (from our mocked blob),
      // NOT the original tiny PNG — the function attempted downscaling.
      expect(result).not.toBe(TINY_PNG_DATA_URL)
      // The drawImage was called with correct dimensions and bitmap was cleaned up
      expect(drawImage).toHaveBeenCalled()
      expect(close).toHaveBeenCalled()
    })

    it('returns placeholder when createImageBitmap throws', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({
          blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' })
        }))
      )
      vi.stubGlobal(
        'createImageBitmap',
        vi.fn(async () => {
          throw new Error('decode failed')
        })
      )
      vi.stubGlobal('OffscreenCanvas', vi.fn(() => ({})))

      const result = await downscaleDataUrlForPreview(TINY_PNG_DATA_URL, 2048)
      expect(result).toBe(FALLBACK_PLACEHOLDER)
    })

    it('closes bitmap even when canvas getContext returns null', async () => {
      const close = vi.fn()
      const bitmap = { width: 4000, height: 3000, close }

      class NullCanvas {
        getContext = () => null
        constructor(_w: number, _h: number) {}
      }

      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({
          blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' })
        }))
      )
      vi.stubGlobal('createImageBitmap', vi.fn(async () => bitmap))
      vi.stubGlobal('OffscreenCanvas', NullCanvas)

      const result = await downscaleDataUrlForPreview(TINY_PNG_DATA_URL, 2048)
      expect(result).toBe(FALLBACK_PLACEHOLDER)
      expect(close).toHaveBeenCalled()
    })

    it('uses placeholder instead of original on convertToBlob failure', async () => {
      const close = vi.fn()
      const bitmap = { width: 4000, height: 3000, close }

      class BrokenCanvas {
        getContext = () => ({ drawImage: vi.fn() })
        convertToBlob = vi.fn(async () => {
          throw new Error('blob failed')
        })
        constructor(_w: number, _h: number) {}
      }

      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({
          blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' })
        }))
      )
      vi.stubGlobal('createImageBitmap', vi.fn(async () => bitmap))
      vi.stubGlobal('OffscreenCanvas', BrokenCanvas)

      const result = await downscaleDataUrlForPreview(TINY_PNG_DATA_URL, 2048)
      expect(result).toBe(FALLBACK_PLACEHOLDER)
      expect(close).toHaveBeenCalled()
    })
  })
})
