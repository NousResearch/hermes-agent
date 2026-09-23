import assert from 'node:assert/strict'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { test, vi } from 'vitest'

const stampExeIdentity = vi.hoisted(() => vi.fn().mockResolvedValue(undefined))
vi.mock('./set-exe-identity.mjs', () => ({ stampExeIdentity }))

const { default: afterExtract } = await import('./after-extract.mjs')

const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

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
