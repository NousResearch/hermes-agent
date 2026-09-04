import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, test } from 'vitest'

import {
  classifySecurityDirectory,
  clearSecurityDirectory,
  numberOfRvaAndSizesOffset,
  readSecurityDirectory,
  sanitizeTree,
} from '../scripts/sanitize-pe-signatures.mjs'

// ─── fixture builder: a PE with a chosen security directory ────────────────

const PE_HEADER_OFFSET = 0x80
const OPTIONAL_HEADER_SIZE = 240 // enough for 16 data directories in either width

/**
 * Build the smallest byte sequence that parses as a PE with data
 * directories, then append `trailing` bytes to stand in for a signature.
 * certOffset/certSize are written verbatim so a test can describe a valid
 * table, a dangling one, or none at all.
 */
function peFile({ magic = 0x20b, certOffset = 0, certSize = 0, trailing = 0 } = {}) {
  const headerEnd = PE_HEADER_OFFSET + 24 + OPTIONAL_HEADER_SIZE
  const buf = Buffer.alloc(headerEnd + trailing)
  buf.write('MZ', 0, 'latin1')
  buf.writeUInt32LE(PE_HEADER_OFFSET, 0x3c)
  buf.writeUInt32LE(0x00004550, PE_HEADER_OFFSET) // "PE\0\0"
  buf.writeUInt16LE(magic === 0x20b ? 0xaa64 : 0x014c, PE_HEADER_OFFSET + 4) // machine
  buf.writeUInt16LE(OPTIONAL_HEADER_SIZE, PE_HEADER_OFFSET + 20)
  buf.writeUInt16LE(magic, PE_HEADER_OFFSET + 24)
  const rvaCount = numberOfRvaAndSizesOffset(magic)
  buf.writeUInt32LE(16, PE_HEADER_OFFSET + 24 + rvaCount)
  const secOff = PE_HEADER_OFFSET + 24 + rvaCount + 4 + 4 * 8
  buf.writeUInt32LE(certOffset, secOff)
  buf.writeUInt32LE(certSize, secOff + 4)
  return { buf, securityDirectoryOffset: secOff, headerEnd }
}

let dir
beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pe-sanitize-'))
})
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true })
})

/** @returns {string} absolute path of the written fixture */
function write(name, spec) {
  const file = path.join(dir, name)
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, peFile(spec).buf)
  return file
}

// ─── header arithmetic ──────────────────────────────────────────────────────

test('NumberOfRvaAndSizes sits at 92 for PE32 and 108 for PE32+', () => {
  // The whole repair keys off this offset: read it wrong and a valid
  // signature looks dangling (or the reverse).
  assert.equal(numberOfRvaAndSizesOffset(0x10b), 92)
  assert.equal(numberOfRvaAndSizesOffset(0x20b), 108)
  assert.equal(numberOfRvaAndSizesOffset(0x107), null)
})

// ─── classification ─────────────────────────────────────────────────────────

test('a table wholly inside the file is a real signature', () => {
  assert.equal(classifySecurityDirectory({ certOffset: 100, certSize: 24, fileSize: 200 }), 'present')
  // Exactly reaching EOF is the normal shape: signatures live at the end.
  assert.equal(classifySecurityDirectory({ certOffset: 100, certSize: 100, fileSize: 200 }), 'present')
})

test('a table that cannot fit in the file is dangling', () => {
  // The stripped-DLL shape: offset equals the (now truncated) file size.
  assert.equal(classifySecurityDirectory({ certOffset: 200, certSize: 24, fileSize: 200 }), 'dangling')
  assert.equal(classifySecurityDirectory({ certOffset: 190, certSize: 24, fileSize: 200 }), 'dangling')
  // Half-populated entries name no readable region either.
  assert.equal(classifySecurityDirectory({ certOffset: 0, certSize: 24, fileSize: 200 }), 'dangling')
  assert.equal(classifySecurityDirectory({ certOffset: 100, certSize: 0, fileSize: 200 }), 'dangling')
})

test('an all-zero table means no signature and needs no repair', () => {
  assert.equal(classifySecurityDirectory({ certOffset: 0, certSize: 0, fileSize: 200 }), 'absent')
})

// ─── reading and clearing ───────────────────────────────────────────────────

test('readSecurityDirectory locates the entry in both PE widths', () => {
  for (const magic of [0x10b, 0x20b]) {
    const spec = { magic, certOffset: 1234, certSize: 16 }
    const file = write(`w${magic}.dll`, spec)
    const entry = readSecurityDirectory(file)
    assert.equal(entry.certOffset, 1234)
    assert.equal(entry.certSize, 16)
    assert.equal(entry.offsetInFile, peFile(spec).securityDirectoryOffset)
  }
})

test('readSecurityDirectory ignores files that are not PEs with directories', () => {
  const plain = path.join(dir, 'notes.txt')
  fs.writeFileSync(plain, 'hello')
  assert.equal(readSecurityDirectory(plain), null)

  // MZ with no PE signature: a DOS-era stub, not a PE.
  const stub = path.join(dir, 'stub.exe')
  const buf = Buffer.alloc(0x100)
  buf.write('MZ', 0, 'latin1')
  buf.writeUInt32LE(0x80, 0x3c)
  fs.writeFileSync(stub, buf)
  assert.equal(readSecurityDirectory(stub), null)
})

test('clearSecurityDirectory zeroes the entry and changes nothing else', () => {
  const spec = { magic: 0x20b, certOffset: 4096, certSize: 40 }
  const file = write('cleared.dll', spec)
  const before = fs.readFileSync(file)

  clearSecurityDirectory(file, peFile(spec).securityDirectoryOffset)

  const after = fs.readFileSync(file)
  assert.equal(after.length, before.length)
  const entry = readSecurityDirectory(file)
  assert.equal(entry.certOffset, 0)
  assert.equal(entry.certSize, 0)
  // Byte-for-byte identical everywhere except the eight-byte entry.
  const off = entry.offsetInFile
  assert.deepEqual(after.subarray(0, off), before.subarray(0, off))
  assert.deepEqual(after.subarray(off + 8), before.subarray(off + 8))
})

// ─── the tree pass ──────────────────────────────────────────────────────────

test('sanitizeTree repairs dangling tables and leaves real signatures alone', () => {
  // The payload shape that broke the Windows bundled release: a stripped
  // DLL whose table starts exactly at EOF, beside correctly signed files.
  const stripped = write('python/DLLs/tcl86t.dll', { certOffset: 464, certSize: 12056 })
  const signed = write('python/python.exe', { certOffset: 400, certSize: 64, trailing: 200 })
  const unsigned = write('python/DLLs/_socket.pyd', {})
  fs.writeFileSync(path.join(dir, 'python', 'LICENSE.txt'), 'not a PE')

  const { scanned, repaired } = sanitizeTree(dir)

  assert.equal(scanned, 3) // the three PEs; the text file is skipped
  assert.deepEqual(repaired, [path.join('python', 'DLLs', 'tcl86t.dll')])
  assert.equal(readSecurityDirectory(stripped).certSize, 0)
  assert.equal(readSecurityDirectory(signed).certSize, 64) // untouched
  assert.equal(readSecurityDirectory(unsigned).certSize, 0)
})

test('sanitizeTree is idempotent', () => {
  write('DLLs/tk86t.dll', { certOffset: 464, certSize: 12064 })

  assert.equal(sanitizeTree(dir).repaired.length, 1)
  assert.deepEqual(sanitizeTree(dir).repaired, [])
})
