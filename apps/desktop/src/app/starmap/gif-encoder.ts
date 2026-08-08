// GIF encoding for the star-map export — thin wrapper over vendored gifenc.
//
// gifenc (v1.0.3, MIT, vendored at ./gifenc.esm.js) does the heavy lifting:
// per-frame median-cut quantization + correct LZW compression. This wrapper
// keeps the same API the export path already uses (encodeGif(frames)) so the
// feature stays zero-dependency and the output is valid, decodable GIF89a.
//
// We quantize per frame (each frame gets its own adaptive 128-colour palette)
// so the build-up's soft orb gradients don't band — same strategy as before,
// but with a proven encoder instead of the hand-rolled LZW that corrupted files.

import { applyPalette, GIFEncoder, quantize } from './gifenc.esm.js'

export interface GifFrame {
  rgba: Uint8Array
  width: number
  height: number
  delayMs: number
}

const MAX_COLORS = 128

export function encodeGif(frames: GifFrame[]): Uint8Array {
  if (!frames.length) {
    return new Uint8Array(0)
  }

  const f0 = frames[0]!
  const w = f0.width
  const h = f0.height
  const encoder = new GIFEncoder()

  for (const f of frames) {
    // Quantize this frame to an adaptive palette, then map pixels to indices.
    const palette = quantize(f.rgba, MAX_COLORS)
    const index = applyPalette(f.rgba, palette)
    const delay = Math.max(2, Math.round(f.delayMs / 10))
    encoder.writeFrame(index, w, h, { delay, palette })
  }

  encoder.finish()
  const bytes = encoder.bytes()
  const ab = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(ab).set(bytes)

  return new Uint8Array(ab)
}
