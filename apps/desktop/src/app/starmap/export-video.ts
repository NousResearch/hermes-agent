// Export the star map's build-up playback to a video file.
//
// The map already drives a cinematic reveal 0→1 sweep (see StarMap's `playing`
// effect, SWEEP_MS). This module wraps that with a MediaRecorder: it mirrors the
// *live* canvas onto an opaque offscreen canvas every animation frame, records
// that, and on completion hands back the WebM blob for download.
//
// It records the *live* canvas (not a re-implementation) so the output is
// byte-identical to what the user sees — scramble core, warp-in births, camera
// steps — EXCEPT the transparent backdrop is replaced with the real page
// background. The star map's <canvas> is transparent (the blue page sits behind
// it), and a bare `captureStream()` would mux that transparency as black. By
// painting the page background onto the copy canvas first, the exported video
// matches what the user actually sees on screen.

import { pageBackgroundColor } from './export-bg'

export interface VideoExport {
  /** Start recording. Expects playback to already be reset to reveal 0. */
  start: (fps?: number) => void
  /** Finish when playback completes; resolves with the recorded blob. */
  finish: () => Promise<Blob>
  /** Abort without producing a file (drop partial data). */
  cancel: () => void
}

const FALLBACK_MIME = 'video/mp4'

function pickMimeType(): string {
  // Prefer MP4 (H.264) — the universal, play-everywhere format (iPhone, most
  // mail/chat clients, WhatsApp). Chromium's MediaRecorder can mux MP4 natively,
  // so no transcode pipeline is needed. Fall back to WebM only for renderers
  // that can't record MP4 (e.g. older Chromium). The GIF exporter stays webm-free.
  const candidates = [
    'video/mp4;codecs=avc1',
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm'
  ]

  for (const m of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(m)) {
      return m
    }
  }

  return FALLBACK_MIME
}

export function createVideoExport(canvas: HTMLCanvasElement): VideoExport {
  let recorder: MediaRecorder | null = null
  let chunks: BlobPart[] = []
  let mime = FALLBACK_MIME
  let copyCanvas: HTMLCanvasElement | null = null
  let copyCtx: CanvasRenderingContext2D | null = null
  let raf = 0
  let bg = '#000000'

  const paint = (): void => {
    // Mirror the live canvas (background + current pixels) onto the opaque copy
    // canvas that the recorder actually captures.
    if (copyCtx && copyCanvas) {
      copyCtx.save()
      copyCtx.fillStyle = bg
      copyCtx.fillRect(0, 0, copyCanvas.width, copyCanvas.height)
      copyCtx.drawImage(canvas, 0, 0, copyCanvas.width, copyCanvas.height)
      copyCtx.restore()
    }

    raf = requestAnimationFrame(paint)
  }

  const start = (fps = 30) => {
    if (recorder) {
      return
    }

    mime = pickMimeType()
    bg = pageBackgroundColor()

    // Opaque copy canvas at the live canvas' full backing-store resolution.
    copyCanvas = document.createElement('canvas')
    copyCanvas.width = canvas.width
    copyCanvas.height = canvas.height
    copyCtx = copyCanvas.getContext('2d')

    chunks = []
    // captureStream from the COPY canvas, which we keep painted with bg + live.
    const stream = copyCanvas.captureStream(fps)
    recorder = new MediaRecorder(stream, { mimeType: mime })

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunks.push(e.data)
      }
    }

    recorder.start(250)
    raf = requestAnimationFrame(paint)
  }

  const finish = (): Promise<Blob> =>
    new Promise((resolve, reject) => {
      const rec = recorder

      if (!rec || rec.state === 'inactive') {
        reject(new Error('No active recording'))

        return
      }

      rec.onstop = () => {
        recorder = null
        cancelAnimationFrame(raf)

        resolve(new Blob(chunks, { type: mime }))
      }

      rec.stop()
    })

  const cancel = () => {
    const rec = recorder
    cancelAnimationFrame(raf)

    if (rec && rec.state !== 'inactive') {
      rec.onstop = null
      rec.stop()
    }

    recorder = null
  }

  return { cancel, finish, start }
}
