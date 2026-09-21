import assert from 'node:assert/strict'
import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { test, vi } from 'vitest'

const stampExeIdentity = vi.hoisted(() => vi.fn().mockResolvedValue(undefined))
vi.mock('./set-exe-identity.mjs', () => ({ stampExeIdentity }))

const { default: afterExtract } = await import('./after-extract.mjs')

const require = createRequire(import.meta.url)
const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const desktopPackage = require(path.join(desktopRoot, 'package.json'))

// #105629: the identity stamp must run BEFORE electron-builder's ASAR-integrity
// rewrite of the exe (beforeCopyExtraFiles), i.e. from afterExtract, never from
// afterPack — rcedit cannot commit changes to the resedit-rewritten PE. And the
// fix must keep the integrity check on (no disableAsarIntegrity workaround).
test('the exe identity stamp is wired to afterExtract, not afterPack, with ASAR integrity kept on', () => {
  assert.equal(desktopPackage.build.afterExtract, 'scripts/after-extract.mjs')
  assert.equal(desktopPackage.build.afterPack, undefined)
  assert.equal(desktopPackage.build.disableAsarIntegrity, undefined)
})

test('stamps the stock electron.exe on win32 only, before it is renamed to Hermes.exe', async () => {
  stampExeIdentity.mockClear()
  const appOutDir = path.join('tmp', 'win-unpacked')

  await afterExtract({
    appOutDir,
    electronPlatformName: 'win32',
    packager: { appInfo: { productFilename: 'Hermes' } }
  })
  assert.deepEqual(stampExeIdentity.mock.calls, [[path.join(appOutDir, 'electron.exe'), desktopRoot]])

  stampExeIdentity.mockClear()
  await afterExtract({
    appOutDir: path.join('tmp', 'linux-unpacked'),
    electronPlatformName: 'linux',
    packager: { appInfo: { productFilename: 'Hermes' } }
  })
  assert.equal(stampExeIdentity.mock.calls.length, 0)
})
