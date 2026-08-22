import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { createPackageWithOptions } from '@electron/asar'
import { test } from 'vitest'

import afterPack from './after-pack.mjs'
import {
  assertPackagedRendererIntegrity,
  packagedResourcesDir,
  verifyPackagedRenderer
} from './packaged-renderer-integrity.mjs'

test('resolves Electron Builder resource directories on every platform', () => {
  assert.equal(
    packagedResourcesDir('/release/mac-arm64', 'darwin', 'Hermes'),
    path.join('/release/mac-arm64', 'Hermes.app', 'Contents', 'Resources')
  )
  assert.equal(
    packagedResourcesDir('/release/win-unpacked', 'win32', 'Hermes'),
    path.join('/release/win-unpacked', 'resources')
  )
})

async function makePackage(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-renderer-integrity-'))
  const source = path.join(root, 'source')
  const resources = path.join(root, 'app', 'resources')
  fs.mkdirSync(source, { recursive: true })
  fs.mkdirSync(resources, { recursive: true })
  for (const [relative, content] of Object.entries(files)) {
    const target = path.join(source, relative)
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, content)
  }
  await createPackageWithOptions(source, path.join(resources, 'app.asar'), {
    unpackDir: 'dist'
  })
  return { root, appOutDir: path.join(root, 'app'), resources }
}

test('accepts a packaged renderer whose ASAR index matches unpacked files', async () => {
  const fixture = await makePackage({
    'dist/index.html': '<script type="module" src="./assets/index-new.js"></script>',
    'dist/assets/index-new.js': 'console.log("new")',
    'dist/assets/feature-new.js': 'export const feature = true'
  })
  try {
    const result = verifyPackagedRenderer(fixture.resources)

    assert.equal(result.ok, true)
    assert.equal(result.indexedFileCount, 3)
    assert.equal(result.unpackedFileCount, 3)
    await assert.doesNotReject(afterPack({ electronPlatformName: 'linux', appOutDir: fixture.appOutDir }))
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true })
  }
})

test('rejects stale ASAR names paired with a newer unpacked renderer', async () => {
  const fixture = await makePackage({
    'dist/index.html': '<script type="module" src="./assets/index-old.js"></script>',
    'dist/assets/index-old.js': 'console.log("old")'
  })
  try {
    const assets = path.join(fixture.resources, 'app.asar.unpacked', 'dist', 'assets')
    fs.rmSync(path.join(assets, 'index-old.js'))
    fs.writeFileSync(path.join(assets, 'index-new.js'), 'console.log("new")')

    const result = verifyPackagedRenderer(fixture.resources)

    assert.equal(result.ok, false)
    assert.ok(result.errors.some(error => error.includes('absent from ASAR index: dist/assets/index-new.js')))
    assert.ok(result.errors.some(error => error.includes('missing unpacked renderer file: dist/assets/index-old.js')))
    await assert.rejects(
      afterPack({ electronPlatformName: 'linux', appOutDir: fixture.appOutDir }),
      /packaged renderer integrity check failed/
    )
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true })
  }
})

test('rejects content drift even when the renderer filename is unchanged', async () => {
  const fixture = await makePackage({
    'dist/index.html': '<script type="module" src="./assets/index.js"></script>',
    'dist/assets/index.js': 'original bytes'
  })
  try {
    const asset = path.join(fixture.resources, 'app.asar.unpacked', 'dist', 'assets', 'index.js')
    fs.writeFileSync(asset, 'different data')

    assert.throws(
      () => assertPackagedRendererIntegrity(fixture.resources),
      /hash mismatch between ASAR index and disk: dist\/assets\/index.js/
    )
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true })
  }
})

test('allows native dependency bytes to change during post-pack signing', async () => {
  const fixture = await makePackage({
    'dist/index.html': '<main id="root"></main>',
    'dist/node_modules/get-windows/main': 'unsigned native helper'
  })
  try {
    fs.writeFileSync(
      path.join(fixture.resources, 'app.asar.unpacked', 'dist', 'node_modules', 'get-windows', 'main'),
      'signed native helper with a larger code signature'
    )

    assert.equal(verifyPackagedRenderer(fixture.resources).ok, true)
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true })
  }
})
