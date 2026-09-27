import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { sanitizeTree } from './sanitize-pe-signatures.mjs'

for (const [magic, countOffset, securityOffset] of [[0x10b, 244, 280], [0x20b, 260, 296]]) {
  test(`tree sanitization preserves all bytes except dangling PE ${magic.toString(16)} directories`, () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pe-sanitize-'))
    // Independent PE specification offsets: e_lfanew=128; optional header=152.
    const rows = [
      ['inside', 400, 32, false], ['at-eof', 400, 64, false], ['absent', 0, 0, false],
      ['starts-at-eof', 464, 24, true], ['crosses-eof', 450, 24, true],
      ['size-only', 0, 24, true], ['offset-only', 400, 0, true]
    ]
    const expected = new Map()
    try {
      for (const [name, offset, size, repair] of rows) {
        const bytes = Buffer.alloc(464, 0xa5)
        bytes.write('MZ', 0, 'latin1')
        bytes.writeUInt32LE(128, 0x3c)
        bytes.writeUInt32LE(0x4550, 128)
        bytes.writeUInt16LE(240, 148)
        bytes.writeUInt16LE(magic, 152)
        bytes.writeUInt32LE(16, countOffset)
        bytes.writeUInt32LE(offset, securityOffset)
        bytes.writeUInt32LE(size, securityOffset + 4)
        fs.writeFileSync(path.join(root, `${name}.dll`), bytes)
        const after = Buffer.from(bytes)
        if (repair) after.fill(0, securityOffset, securityOffset + 8)
        expected.set(`${name}.dll`, after)
      }
      const dos = Buffer.alloc(256)
      dos.write('MZ', 0)
      dos.writeUInt32LE(128, 0x3c)
      for (const [name, bytes] of [['notes.txt', Buffer.from('not PE')], ['dos.exe', dos]]) {
        fs.writeFileSync(path.join(root, name), bytes)
        expected.set(name, bytes)
      }
      const unsupported = Buffer.from(expected.get('inside.dll'))
      unsupported.writeUInt16LE(0x107, 152)
      fs.writeFileSync(path.join(root, 'unsupported.exe'), unsupported)
      expected.set('unsupported.exe', unsupported)
      const result = sanitizeTree(root)
      assert.equal(result.scanned, 7)
      assert.deepEqual(result.repaired.sort(), ['crosses-eof.dll', 'offset-only.dll', 'size-only.dll', 'starts-at-eof.dll'])
      for (const [name, bytes] of expected) assert.deepEqual(fs.readFileSync(path.join(root, name)), bytes, name)
      assert.deepEqual(sanitizeTree(root).repaired, [])
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
}
