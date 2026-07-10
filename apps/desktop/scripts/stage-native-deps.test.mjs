import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

import { copyDirRecursive } from '../scripts/stage-native-deps.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))

test('copyDirRecursive reproduces a nested tree with exact bytes', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-stage-'))
  try {
    const src = path.join(tempRoot, 'src')
    const dest = path.join(tempRoot, 'dest')
    // Mirror node-pty's payload shape: top-level files plus a nested conpty/
    // dir holding a binary blob (the .dll/.exe that trip Windows cpSync).
    fs.mkdirSync(path.join(src, 'conpty'), { recursive: true })
    fs.writeFileSync(path.join(src, 'pty.node'), Buffer.from([0, 1, 2, 253, 254, 255]))
    fs.writeFileSync(path.join(src, 'index.js'), 'module.exports = {}\n', 'utf8')
    fs.writeFileSync(path.join(src, 'conpty', 'conpty.dll'), Buffer.from([77, 90, 0, 255]))
    fs.writeFileSync(path.join(src, 'conpty', 'OpenConsole.exe'), Buffer.from([77, 90, 144, 0]))

    copyDirRecursive(src, dest)

    for (const rel of ['pty.node', 'index.js', 'conpty/conpty.dll', 'conpty/OpenConsole.exe']) {
      assert.ok(fs.existsSync(path.join(dest, rel)), `missing ${rel}`)
      assert.deepEqual(
        fs.readFileSync(path.join(dest, rel)),
        fs.readFileSync(path.join(src, rel)),
        `bytes differ for ${rel}`
      )
    }
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('copyDirRecursive creates the destination even for an empty source', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-stage-'))
  try {
    const src = path.join(tempRoot, 'empty-src')
    const dest = path.join(tempRoot, 'empty-dest')
    fs.mkdirSync(src, { recursive: true })

    copyDirRecursive(src, dest)

    assert.equal(fs.existsSync(dest), true)
    assert.deepEqual(fs.readdirSync(dest), [])
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('stage-native-deps uses no recursive cpSync (Windows 0xC0000409 regression)', () => {
  // Node 22.x's native recursive cpSync crashes/hard-faults on Windows while
  // copying the conpty payload; directory copies must go through the manual
  // walk instead. Guard against anyone reintroducing `cpSync(.., recursive)`.
  const source = fs.readFileSync(path.join(here, 'stage-native-deps.mjs'), 'utf8')
  // Drop comments so the guard inspects real calls, not the doc comment that
  // names the very API we're avoiding.
  const code = source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
  const recursiveCpSync = /cpSync\([^\n]*recursive/.test(code)
  assert.equal(recursiveCpSync, false, 'stage-native-deps.mjs must not call cpSync with { recursive: true }')
})
