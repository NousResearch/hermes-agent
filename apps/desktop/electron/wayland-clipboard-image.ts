// Read a clipboard image from a Wayland compositor via wl-paste (#85782).
// Some compositors (COSMIC's screenshot tool) offer a PNG payload as
// application/octet-stream rather than image/png, which Electron's
// clipboard.readImage() does not recognize — the paste then looks empty.
// wl-paste exposes the raw payload; we validate real PNG bytes before
// returning. Returns a PNG Buffer or null; exec injectable; never throws.

import { execFileSync } from 'node:child_process'
import type { ExecFileSyncOptions } from 'node:child_process'

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

// Same bounding discipline as the WSL reader: a clipboard read must never
// hang the main process or buffer unbounded output.
const EXEC_OPTIONS: ExecFileSyncOptions = {
  encoding: 'buffer',
  timeout: 5000,
  maxBuffer: 64 * 1024 * 1024,
  stdio: ['ignore', 'pipe', 'ignore']
}

// execFileSync returns a Buffer under encoding:'buffer'; tests may hand back
// either a Buffer, a string, or an {stdout} wrapper — normalize all three.
function stdoutBytes(result: unknown): Buffer {
  const raw =
    Buffer.isBuffer(result) || typeof result === 'string'
      ? result
      : (result as { stdout?: Buffer | string } | undefined)?.stdout

  if (raw === undefined || raw === null) {
    return Buffer.alloc(0)
  }

  return Buffer.isBuffer(raw) ? raw : Buffer.from(String(raw), 'utf8')
}

function isPng(buffer: unknown): boolean {
  return (
    Buffer.isBuffer(buffer) &&
    buffer.length >= PNG_SIGNATURE.length &&
    buffer.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)
  )
}

// Read the Wayland clipboard image. Returns a PNG Buffer, or null when there
// is no image, wl-paste is missing, the session is not Wayland, or the bytes
// are not a PNG. Linux-only by contract (caller gates on platform); never
// throws.
function readWaylandClipboardImage({
  exec = execFileSync,
  env = process.env
}: { exec?: typeof execFileSync; env?: NodeJS.ProcessEnv } = {}) {
  // Off a Wayland session there is nothing wl-paste can answer for — don't
  // even probe (X11 and macOS/Windows callers must observe zero spawns).
  if (!env.WAYLAND_DISPLAY) {
    return null
  }

  try {
    const listed = exec('wl-paste', ['--list-types'], EXEC_OPTIONS)

    const types = stdoutBytes(listed)
      .toString('utf8')
      .split('\n')
      .map(t => t.trim())
      .filter(Boolean)

    if (types.length === 0) {
      return null
    }

    // Ask for each non-text offer explicitly so a PNG hidden under
    // application/octet-stream still wins when the clipboard also advertises
    // text/plain. Prefer image/png, then other image types, then opaque binary
    // offers; every candidate must pass the PNG signature check.
    const candidates = [
      ...types.filter(t => t === 'image/png'),
      ...types.filter(t => t !== 'image/png' && t.startsWith('image/')),
      ...types.filter(t => !t.startsWith('image/') && !t.startsWith('text/'))
    ]

    // A failed read falls to the next rung, not out of the ladder: one MIME
    // request failing (offer vanished between --list-types and --type, or a
    // type wl-paste refuses) must not discard candidates not yet tried.
    for (const mime of candidates) {
      let png: Buffer

      try {
        png = stdoutBytes(exec('wl-paste', ['--type', mime], EXEC_OPTIONS))
      } catch {
        continue
      }

      if (isPng(png)) {
        return png
      }
    }

    // Keep an untyped final probe for compositors whose advertised type cannot
    // be requested verbatim. Only real PNG bytes count, so text cannot attach.
    const raw = stdoutBytes(exec('wl-paste', [], EXEC_OPTIONS))

    return isPng(raw) ? raw : null
  } catch {
    // wl-paste absent, timed out, or failed — not an image we can attach.
    return null
  }
}

export { readWaylandClipboardImage }
