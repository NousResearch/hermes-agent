import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { test } from 'vitest'

const require = createRequire(import.meta.url)
const signerUtil = path.join(path.dirname(require.resolve('@electron/osx-sign')), 'util.js')

test('the signing walk stays bounded on a truncated protobuf-like resource', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sign-input-'))

  try {
    // The byte sample declares a large length, but the file ends immediately.
    const resource = Buffer.alloc(512, 0x41)
    Buffer.from([0x0a, 0xff, 0xff, 0xff, 0x7f]).copy(resource)
    fs.writeFileSync(path.join(root, 'resource.dat'), resource)
    fs.writeFileSync(path.join(root, 'readme.txt'), 'ordinary text\n')
    const binary = path.join(root, 'library.dylib')
    fs.writeFileSync(binary, Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 0, 0, 0, 0]))

    // Isolate the scanner: a regression must fail the test, not exhaust its runner.
    const code = `import(${JSON.stringify(pathToFileURL(signerUtil).href)})
      .then(async ({ walk }) => console.log(JSON.stringify(await walk(${JSON.stringify(root)}))))
      .catch(error => { console.error(error); process.exitCode = 1 })`
    const result = spawnSync(process.execPath, ['--max-old-space-size=64', '-e', code], {
      encoding: 'utf8', timeout: 10000, maxBuffer: 64 * 1024, windowsHide: true
    })

    assert.equal(result.status, 0, result.error?.message || result.stderr)
    assert.deepEqual(JSON.parse(result.stdout), [binary])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}, 15000)
