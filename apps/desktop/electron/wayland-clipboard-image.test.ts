import assert from 'node:assert/strict'

import { test } from 'vitest'

import { readWaylandClipboardImage } from './wayland-clipboard-image'

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

function fakePngBuffer(extraBytes = 16) {
  return Buffer.concat([PNG_SIGNATURE, Buffer.alloc(extraBytes, 0x42)])
}

const WAYLAND_ENV = { WAYLAND_DISPLAY: 'wayland-0' }

/** exec double that answers the wl-paste probe/extract sequence for a clipboard
 *  offering a PNG image under a list of MIME types. */
function wlPasteExec(results: { types?: string[]; png?: Buffer | null }) {
  const calls: Array<{ cmd: string; args: string[] }> = []

  const exec = ((cmd: string, args: string[]) => {
    calls.push({ cmd, args })

    if (args.includes('--list-types')) {
      return { stdout: (results.types ?? ['image/png']).join('\n'), stderr: '' }
    }

    if (args[0] === '--type') {
      return { stdout: results.png ?? fakePngBuffer(), stderr: '' }
    }

    // Untyped `wl-paste` yields the raw payload of the offered type — this is
    // the `wl-paste | od` path from the #85782 report.
    return { stdout: results.png ?? Buffer.alloc(0), stderr: '' }
  }) as any

  return { exec, calls }
}

test('reads a PNG advertised as image/png', () => {
  const { exec, calls } = wlPasteExec({ types: ['image/png'], png: fakePngBuffer() })
  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.ok(Buffer.isBuffer(result))
  assert.ok(result.equals(fakePngBuffer()))
  assert.deepEqual(calls[0], { cmd: 'wl-paste', args: ['--list-types'] })
})

test('reads a PNG offered as application/octet-stream via PNG-signature fallback', () => {
  // The #85782 report: COSMIC's screenshot tool offers the PNG payload as
  // application/octet-stream, so no image/* MIME is advertised at all.
  const { exec, calls } = wlPasteExec({ types: ['application/octet-stream'], png: fakePngBuffer() })
  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.ok(Buffer.isBuffer(result))
  assert.ok(result.equals(fakePngBuffer()))
  assert.ok(calls.some(c => c.cmd === 'wl-paste' && c.args[0] === '--type' && c.args[1] === 'application/octet-stream'))
})

test('a failing MIME read falls through to the next candidate instead of aborting the ladder', () => {
  // The clipboard can change between --list-types and --type, or wl-paste can
  // refuse one offer — a single failed read must not discard the remaining
  // candidates (the ladder's "failed read falls to the next rung" invariant).
  const png = fakePngBuffer()
  const calls: string[][] = []

  const exec = ((_cmd: string, args: string[]) => {
    calls.push(args)

    if (args.includes('--list-types')) {
      return Buffer.from('image/webp\napplication/octet-stream\n')
    }

    if (args[0] === '--type' && args[1] === 'image/webp') {
      throw new Error('offer vanished')
    }

    if (args[0] === '--type' && args[1] === 'application/octet-stream') {
      return png
    }

    return Buffer.alloc(0)
  }) as any

  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.ok(result?.equals(png))
  assert.ok(calls.some(args => args[0] === '--type' && args[1] === 'image/webp'))
  assert.ok(calls.some(args => args[0] === '--type' && args[1] === 'application/octet-stream'))
})

test('selects an octet-stream PNG when the clipboard also offers plain text', () => {
  const png = fakePngBuffer()
  const calls: string[][] = []

  const exec = ((_cmd: string, args: string[]) => {
    calls.push(args)

    if (args.includes('--list-types')) {
      return Buffer.from('text/plain\napplication/octet-stream\n')
    }

    if (args[0] === '--type' && args[1] === 'application/octet-stream') {
      return png
    }

    return Buffer.from('plain text')
  }) as any

  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.ok(result?.equals(png))
  assert.ok(calls.some(args => args[0] === '--type' && args[1] === 'application/octet-stream'))
})

test('returns null when bytes are not a PNG (no false attach)', () => {
  const { exec } = wlPasteExec({ types: ['application/octet-stream'], png: Buffer.from('not a png') })
  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.equal(result, null)
})

test('returns null when no MIME types are advertised', () => {
  const { exec } = wlPasteExec({ types: [], png: null })
  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.equal(result, null)
})

test('returns null off a Wayland session (WAYLAND_DISPLAY unset)', () => {
  const { exec, calls } = wlPasteExec({ types: ['image/png'], png: fakePngBuffer() })
  const result = readWaylandClipboardImage({ exec, env: {} })
  assert.equal(result, null)
  assert.equal(calls.length, 0)
})

test('returns null when wl-paste is not installed (ENOENT)', () => {
  const exec = (() => {
    const err = new Error('spawn wl-paste ENOENT') as any
    err.code = 'ENOENT'
    throw err
  }) as any

  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.equal(result, null)
})

test('invokes wl-paste directly without shell arguments', () => {
  const { exec, calls } = wlPasteExec({ types: ['image/png'], png: fakePngBuffer() })
  readWaylandClipboardImage({ exec, env: WAYLAND_ENV })

  for (const call of calls) {
    assert.equal(call.cmd, 'wl-paste')
    assert.ok(Array.isArray(call.args))
    assert.equal(call.args.includes('-c'), false)
    assert.equal(call.args.some(arg => /^(?:ba)?sh$/i.test(arg)), false)
  }
})

test('bounded: exec options always carry a timeout and maxBuffer', () => {
  const opts: unknown[] = []

  const exec = ((cmd: string, args: string[], options: any) => {
    opts.push(options)

    return { stdout: '', stderr: '' }
  }) as any

  readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.ok(opts.length > 0)

  for (const o of opts) {
    assert.ok((o as any).timeout > 0 && (o as any).timeout <= 10_000)
    assert.ok((o as any).maxBuffer > 0)
  }
})

test('returns null on generic exec failure instead of throwing', () => {
  const exec = (() => {
    throw new Error('boom')
  }) as any

  const result = readWaylandClipboardImage({ exec, env: WAYLAND_ENV })
  assert.equal(result, null)
})
