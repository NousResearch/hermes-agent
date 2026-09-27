import path from 'node:path'
import { expect, test } from 'vitest'

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
  expect(mac.appPath).toBe(path.join(releaseRoot, 'mac-arm64', `${productName}.app`))
  expect(mac.binary).toBe(
    path.join(releaseRoot, 'mac-arm64', `${productName}.app`, 'Contents', 'MacOS', executableName)
  )

  const win = resolveDesktopAppLayout({
    releaseRoot,
    platform: 'win32',
    arch: 'x64',
    productName,
    executableName
  })
  expect(win.binary).toBe(path.join(releaseRoot, 'win-unpacked', `${executableName}.exe`))

  const linux = resolveDesktopAppLayout({
    releaseRoot,
    platform: 'linux',
    arch: 'x64',
    productName,
    executableName
  })
  expect(linux.binary).toBe(path.join(releaseRoot, 'linux-unpacked', executableName))
})

test('expands DMG artifact names from electron-builder artifactName metadata', () => {
  const dmg = expandArtifactName(PACKAGE_JSON.build.artifactName, {
    version: PACKAGE_JSON.version,
    platform: 'darwin',
    arch: 'arm64',
    ext: 'dmg'
  })

  expect(dmg).toBe(`AI-Evolution-Jarvis-${PACKAGE_JSON.version}-mac-arm64.dmg`)
  expect(dmg.startsWith('AI-Evolution-Jarvis-')).toBe(true)
})
