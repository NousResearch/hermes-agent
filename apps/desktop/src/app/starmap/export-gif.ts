// Export the live star-map canvas as an animated, looping GIF.
//
// This replicates the video-export flow (export-video.ts) but for a GIF and
// with ZERO dependencies / native binaries. While the caller plays the build-up
// (reveal 0→1), we sample the LIVE canvas on an interval, downsample each frame,
// and feed the frames into the bundled median-cut + LZW GIF encoder. The output
// is pixel-for-pixel what the user sees (scramble core, warp-ins, force-layout
// positions, theme palette) — no re-implementation of the renderer.
//
// Sampling instead of recording: GIFs are palette-based (≤256 colours/frame) and
// best at a modest fps, so we grab ~12 frames/sec at a downscaled size rather
// than 30fps full-res. This keeps the encode fast and the file small while the
// build-up still reads as a smooth, looping animation.

import { pageBackgroundColor } from './export-bg'
import { encodeGif } from './gif-encoder'

export interface GifExport {
  /** Begin sampling the live canvas (expects playback to be running). */
  start: (opts?: { fps?: number; width?: number }) => void
  /** Stop sampling and resolve with the animated GIF blob. */
  finish: () => Promise<Blob>
  /** Abort without producing a file. */
  cancel: () => void
}

export const GIF_MAX_FRAMES = 400 // hard cap to keep encoding bounded

const DEFAULTS = {
  fps: 14,
  width: 720
}

export function createGifExport(canvas: HTMLCanvasElement): GifExport {
  let timer: ReturnType<typeof setInterval> | null = null
  let frames: Array<{ rgba: Uint8Array; width: number; height: number; delayMs: number }> = []
  let stopped = false
  let w0 = 0
  let h0 = 0
  let targetW = DEFAULTS.width
  let targetH = 0
  let resizeCanvas: HTMLCanvasElement | null = null
  let resizeCtx: CanvasRenderingContext2D | null = null
  let sample = 0

  const start = ({ fps = DEFAULTS.fps, width = DEFAULTS.width }: { fps?: number; width?: number } = {}): void => {
    if (timer) {
      return
    }

    // Clear any stale buffers.
    frames = []
    stopped = false
    sample = 0

    w0 = canvas.width
    h0 = canvas.height
    targetW = Math.min(width, w0)
    const scale = targetW / (w0 || 1)
    targetH = Math.max(1, Math.round(h0 * scale))

    // Downsample canvas: a shared offscreen we redraw each sample.
    resizeCanvas = document.createElement('canvas')
    resizeCanvas.width = targetW
    resizeCanvas.height = targetH
    resizeCtx = resizeCanvas.getContext('2d', { willReadFrequently: true })

    const intervalMs = 1000 / fps

    // Sample the live canvas now (and on each tick). Every captured frame is
    // downscaled to targetW×targetH before reading pixels. Keeps sampling until
    // finish()/cancel() — the caller stops it when the build-up completes.
    const tick = (): void => {
      if (stopped || !resizeCtx) {
        return
      }

      // The live canvas is transparent; paint the real page background first so
      // the transparent space becomes the royal-blue backdrop (not black).
      resizeCtx.save()
      resizeCtx.fillStyle = pageBackgroundColor()
      resizeCtx.fillRect(0, 0, targetW, targetH)
      resizeCtx.restore()
      resizeCtx.drawImage(canvas, 0, 0, targetW, targetH)
      const img = resizeCtx.getImageData(0, 0, targetW, targetH)
      frames.push({ rgba: new Uint8Array(img.data.buffer.slice(0)), width: targetW, height: targetH, delayMs: intervalMs })
      sample += 1

      if (sample >= GIF_MAX_FRAMES) {
        if (timer) {
          clearInterval(timer)
          timer = null
        }
      }
    }

    tick()
    timer = setInterval(tick, intervalMs)
  }

  const finish = (): Promise<Blob> =>
    new Promise((resolve, reject) => {
      stopped = true

      if (timer) {
        clearInterval(timer)
        timer = null
      }

      if (!frames.length) {
        reject(new Error('No frames captured'))

        return
      }

      try {
        const bytes = encodeGif(frames)
        const ab = new ArrayBuffer(bytes.byteLength)
        new Uint8Array(ab).set(bytes)
        const blob = new Blob([ab], { type: 'image/gif' })
        resolve(blob)
      } catch (err) {
        reject(err)
      }
    })

  const cancel = (): void => {
    stopped = true

    if (timer) {
      clearInterval(timer)
      timer = null
    }

    frames = []
  }

  return { cancel, finish, start }
}
