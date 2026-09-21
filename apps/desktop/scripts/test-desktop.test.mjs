import assert from 'node:assert/strict'
import path from 'node:path'
import test from 'node:test'

import PACKAGE_JSON from '../package.json' with { type: 'json' }
import { expandArtifactName, resolveDesktopAppLayout } from './test-desktop.mjs'

const releaseRoot = path.join('/tmp', 'desktop-release')
const productName = PACKAGE_JSON.productName
const executableName = PACKAGE_JSON.build.executableName

test('resolves packaged app paths from desktop package metadata', () => {
  const mac = resolveDesktopAppLayout({
    releaseRoot,
    platform: 'darwin',
    arch: 'arm64',
    productName,
    executableName
  })
  assert.equal(mac.appPath, path.join(releaseRoot, 'mac-arm64', 'AI Evolution Jarvis.app'))
  assert.equal(
    mac.binary,
    path.join(releaseRoot, 'mac-arm64', 'AI Evolution Jarvis.app', 'Contents', 'MacOS', 'AI Evolution Jarvis')
  )

  const win = resolveDesktopAppLayout({
    releaseRoot,
    platform: 'win32',
    arch: 'x64',
    productName,
    executableName
  })
  assert.equal(win.binary, path.join(releaseRoot, 'win-unpacked', 'AI Evolution Jarvis.exe'))

  const linux = resolveDesktopAppLayout({
    releaseRoot,
    platform: 'linux',
    arch: 'x64',
    productName,
    executableName
  })
  assert.equal(linux.binary, path.join(releaseRoot, 'linux-unpacked', 'AI Evolution Jarvis'))
})

test('expands DMG artifact names from electron-builder artifactName metadata', () => {
  const dmg = expandArtifactName(PACKAGE_JSON.build.artifactName, {
    version: PACKAGE_JSON.version,
    platform: 'darwin',
    arch: 'arm64',
    ext: 'dmg'
  })

  assert.equal(dmg, `AI-Evolution-Jarvis-${PACKAGE_JSON.version}-mac-arm64.dmg`)
  assert.ok(dmg.startsWith('AI-Evolution-Jarvis-'))
})
