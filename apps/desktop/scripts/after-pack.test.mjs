import assert from 'node:assert/strict'
import path from 'node:path'
import { test } from 'vitest'

import { afterPack } from '../scripts/after-pack.mjs'

test('afterPack stamps the packaged Windows executable', async () => {
  const calls = []
  const appOutDir = path.join('C:', 'build', 'win-unpacked')

  await afterPack(
    { electronPlatformName: 'win32', appOutDir, packager: { appInfo: { productFilename: 'Hermes' } } },
    { stamp: async (...args) => calls.push(args) }
  )

  assert.equal(calls.length, 1)
  assert.equal(calls[0][0], path.join(appOutDir, 'Hermes.exe'))
})

test('afterPack rejects a Windows identity-stamping failure', async () => {
  await assert.rejects(
    afterPack(
      { electronPlatformName: 'win32', appOutDir: 'out', packager: { appInfo: { productFilename: 'Hermes' } } },
      { stamp: async () => { throw new Error('rcedit unavailable') } }
    ),
    /rcedit unavailable/
  )
})

test('afterPack skips non-Windows targets', async () => {
  let called = false

  await afterPack({ electronPlatformName: 'darwin', appOutDir: 'out' }, { stamp: async () => { called = true } })

  assert.equal(called, false)
})
