import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  hasSoftwareRenderingFlag,
  readChromiumFlags,
  sanitizeChromiumFlags,
  splitChromiumFlag,
  withSoftwareRenderingFlags,
  writeChromiumFlags
} from './chromium-flags'

test('sanitizeChromiumFlags keeps only well-formed switches, deduped', () => {
  assert.deepEqual(
    sanitizeChromiumFlags([
      '--disable-gpu',
      ' --ozone-platform=wayland ',
      '--disable-gpu',
      'disable-gpu',
      '--has space',
      '--=novalue',
      'https://example.com',
      42,
      null,
      '--enable-features=Vulkan,UseSkiaRenderer'
    ]),
    ['--disable-gpu', '--ozone-platform=wayland', '--enable-features=Vulkan,UseSkiaRenderer']
  )
  assert.deepEqual(sanitizeChromiumFlags('--disable-gpu'), [])
})

test('read/write round-trip through chromium-flags.json; missing or malformed files yield none', () => {
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-chromium-flags-'))

  assert.deepEqual(readChromiumFlags(userData), [])

  fs.writeFileSync(path.join(userData, 'chromium-flags.json'), '{ not json')
  assert.deepEqual(readChromiumFlags(userData), [])

  fs.writeFileSync(path.join(userData, 'chromium-flags.json'), JSON.stringify({ flags: 'nope' }))
  assert.deepEqual(readChromiumFlags(userData), [])

  const written = writeChromiumFlags(userData, withSoftwareRenderingFlags(['--ozone-platform=wayland', '--disable-gpu']))

  assert.deepEqual(written, ['--ozone-platform=wayland', '--disable-gpu', '--disable-gpu-compositing'])
  assert.deepEqual(readChromiumFlags(userData), written)
  assert.equal(hasSoftwareRenderingFlag(written), true)
  assert.equal(hasSoftwareRenderingFlag(['--disable-gpu-compositing']), false)
  assert.deepEqual(splitChromiumFlag('--ozone-platform=wayland'), { name: 'ozone-platform', value: 'wayland' })
  assert.deepEqual(splitChromiumFlag('--disable-gpu'), { name: 'disable-gpu' })

  fs.rmSync(userData, { recursive: true, force: true })
})
