import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test, vi } from 'vitest'

import afterPack, { restoreMacLocaleMarkers } from './after-pack.mjs'

function makeMacApp(root, productName = 'Hermes') {
  const contents = path.join(root, `${productName}.app`, 'Contents')
  const frameworkResources = path.join(
    contents,
    'Frameworks',
    'Electron Framework.framework',
    'Versions',
    'A',
    'Resources'
  )

  fs.mkdirSync(frameworkResources, { recursive: true })
  fs.mkdirSync(path.join(contents, 'Resources'), { recursive: true })

  return { contents, frameworkResources }
}

test('restoreMacLocaleMarkers recreates only framework locale directories', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-after-pack-'))
  try {
    const { contents, frameworkResources } = makeMacApp(root)
    fs.mkdirSync(path.join(frameworkResources, 'nb.lproj'))
    fs.mkdirSync(path.join(frameworkResources, 'en_GB.lproj'))
    fs.writeFileSync(path.join(frameworkResources, 'locale.pak'), 'locale data')

    assert.equal(await restoreMacLocaleMarkers(root, 'Hermes'), 2)
    assert.equal(fs.statSync(path.join(contents, 'Resources', 'nb.lproj')).isDirectory(), true)
    assert.equal(fs.statSync(path.join(contents, 'Resources', 'en_GB.lproj')).isDirectory(), true)
    assert.equal(fs.existsSync(path.join(contents, 'Resources', 'locale.pak')), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('restoreMacLocaleMarkers leaves packing intact when framework resources are unavailable', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-after-pack-'))
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
  try {
    assert.equal(await restoreMacLocaleMarkers(root, 'Hermes'), 0)
    assert.equal(warn.mock.calls.length, 1)
    assert.match(warn.mock.calls[0][0], /macOS locale markers were not restored/)
  } finally {
    warn.mockRestore()
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('afterPack restores markers in the configured macOS app bundle', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-after-pack-'))
  try {
    const { contents, frameworkResources } = makeMacApp(root, 'Hermes Preview')
    fs.mkdirSync(path.join(frameworkResources, 'nb.lproj'))

    await afterPack({
      appOutDir: root,
      electronPlatformName: 'darwin',
      packager: { appInfo: { productFilename: 'Hermes Preview' } }
    })

    assert.equal(fs.statSync(path.join(contents, 'Resources', 'nb.lproj')).isDirectory(), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('afterPack leaves non-macOS builds unchanged', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-after-pack-'))
  try {
    await afterPack({ appOutDir: root, electronPlatformName: 'linux' })
    assert.deepEqual(fs.readdirSync(root), [])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
