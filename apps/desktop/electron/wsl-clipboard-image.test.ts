import assert from 'node:assert/strict'
import test from 'node:test'

import {
  decodeClipboardImageBase64,
  encodePowerShellCommand,
  powershellCandidates,
  readWslWindowsClipboardImage
} from './wsl-clipboard-image'

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

function fakePngBuffer(extraBytes = 16) {
  return Buffer.concat([PNG_SIGNATURE, Buffer.alloc(extraBytes, 0x42)])
}

test('encodePowerShellCommand produces UTF-16LE base64 PowerShell can decode', () => {
  const encoded = encodePowerShellCommand('Write-Output "hi"')
  const roundTripped = Buffer.from(encoded, 'base64').toString('utf16le')
  assert.equal(roundTripped, 'Write-Output "hi"')
})

test('decodeClipboardImageBase64 returns a Buffer for valid PNG base64', () => {
  const png = fakePngBuffer()
  const decoded = decodeClipboardImageBase64(png.toString('base64'))
  assert.ok(Buffer.isBuffer(decoded))
  assert.ok(decoded.equals(png))
})

test('decodeClipboardImageBase64 trims surrounding whitespace before decoding', () => {
  const png = fakePngBuffer()
  const decoded = decodeClipboardImageBase64(`\n  ${png.toString('base64')}  \r\n`)
  assert.ok(decoded && decoded.equals(png))
})

test('decodeClipboardImageBase64 returns null for empty / whitespace input', () => {
  assert.equal(decodeClipboardImageBase64(''), null)
  assert.equal(decodeClipboardImageBase64('   \n  '), null)
  assert.equal(decodeClipboardImageBase64(null), null)
  assert.equal(decodeClipboardImageBase64(undefined), null)
})

test('decodeClipboardImageBase64 rejects base64 without a PNG signature', () => {
  // Valid base64, but the decoded bytes are not a PNG.
  const notPng = Buffer.from('this is not a png at all').toString('base64')
  assert.equal(decodeClipboardImageBase64(notPng), null)
})

test('readWslWindowsClipboardImage decodes the first candidate that returns a PNG', () => {
  const png = fakePngBuffer()
  const calls = []

  const exec = ((cmd, args) => {
    calls.push({ cmd, args })

    return png.toString('base64')
  }) as any

  const result = readWslWindowsClipboardImage({ exec, candidates: ['powershell.exe'] })
  assert.ok(result && result.equals(png))
  assert.equal(calls.length, 1)
  assert.equal(calls[0].cmd, 'powershell.exe')
  // -STA is mandatory for System.Windows.Forms.Clipboard.
  assert.ok(calls[0].args.includes('-STA'))
  assert.ok(calls[0].args.includes('-EncodedCommand'))
})

test('readWslWindowsClipboardImage returns null and stops when stdout is empty (no image)', () => {
  let count = 0

  const exec = (() => {
    count += 1

    return ''
  }) as any

  const result = readWslWindowsClipboardImage({
    exec,
    candidates: ['powershell.exe', '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
  })

  assert.equal(result, null)
  // Empty stdout means "no image on the clipboard" — don't probe further candidates.
  assert.equal(count, 1)
})

test('readWslWindowsClipboardImage falls through to the next candidate when one throws', () => {
  const png = fakePngBuffer()
  const seen = []

  const exec = cmd => {
    seen.push(cmd)

    if (cmd === 'powershell.exe') {
      throw Object.assign(new Error('not found'), { code: 'ENOENT' })
    }

    return png.toString('base64') as any
  }

  const result = readWslWindowsClipboardImage({
    exec,
    candidates: ['powershell.exe', '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
  })

  assert.ok(result && result.equals(png))
  assert.deepEqual(seen, ['powershell.exe', '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'])
})

test('readWslWindowsClipboardImage returns null when every candidate throws', () => {
  const exec = () => {
    throw new Error('boom')
  }

  const result = readWslWindowsClipboardImage({ exec, candidates: ['a', 'b'] })
  assert.equal(result, null)
})

test('powershellCandidates lists the bare name first, then the absolute fallback', () => {
  const candidates = powershellCandidates()
  assert.equal(candidates[0], 'powershell.exe')
  assert.ok(candidates.some(c => c.endsWith('WindowsPowerShell/v1.0/powershell.exe')))
})
