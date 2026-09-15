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

test('configures executable stamping before electron-builder adds ASAR integrity', () => {
  assert.equal(desktopPackage.build.afterExtract, 'scripts/after-extract.mjs')
  assert.equal(desktopPackage.build.afterPack, undefined)
})

test('stamps the stock Windows executable before it is renamed and rewritten', async () => {
  stampExeIdentity.mockClear()
  const appOutDir = path.join('tmp', 'win-unpacked')

  await afterExtract({
    appOutDir,
    electronPlatformName: 'win32',
    packager: { appInfo: { productFilename: 'Hermes' } }
  })

  assert.deepEqual(stampExeIdentity.mock.calls, [[path.join(appOutDir, 'electron.exe'), desktopRoot]])
})

test('does not edit non-Windows Electron binaries', async () => {
  stampExeIdentity.mockClear()

  await afterExtract({
    appOutDir: path.join('tmp', 'linux-unpacked'),
    electronPlatformName: 'linux',
    packager: { appInfo: { productFilename: 'Hermes' } }
  })

  assert.equal(stampExeIdentity.mock.calls.length, 0)
})
