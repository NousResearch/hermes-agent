import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { appIconCandidates, resolveAppIconPath } from './app-icon'

test('windows candidates prefer resources then assets .ico over apple-touch', () => {
  const candidates = appIconCandidates({
    appRoot: 'C:\\app',
    resourcesPath: 'C:\\resources',
    platform: 'win32'
  })

  assert.equal(candidates[0], path.join('C:\\resources', 'icon.ico'))
  assert.equal(candidates[1], path.join('C:\\app', 'assets', 'icon.ico'))
  assert.ok(candidates.includes(path.join('C:\\app', 'public', 'apple-touch-icon.png')))
  assert.ok(
    candidates.indexOf(path.join('C:\\app', 'assets', 'icon.ico')) <
      candidates.indexOf(path.join('C:\\app', 'public', 'apple-touch-icon.png'))
  )
})

test('darwin prefers native icns/.png ahead of the packaged .ico', () => {
  const candidates = appIconCandidates({
    appRoot: '/Applications/Hermes.app/Contents/Resources/app.asar',
    resourcesPath: '/Applications/Hermes.app/Contents/Resources',
    platform: 'darwin'
  })

  // extraResources always ships resources/icon.ico (the Windows PE-stamp
  // source) on every platform, so the macOS-native .icns/.png must be ordered
  // ahead of it; otherwise .ico would win first-pick on Darwin even when a
  // proper .icns exists.
  const packagedIco = path.join('/Applications/Hermes.app/Contents/Resources', 'icon.ico')
  assert.equal(candidates[0], path.join('/Applications/Hermes.app/Contents/Resources', 'icon.icns'))
  assert.equal(candidates[1], path.join('/Applications/Hermes.app/Contents/Resources', 'icon.png'))
  assert.ok(
    candidates.indexOf(path.join('/Applications/Hermes.app/Contents/Resources/app.asar', 'assets', 'icon.icns')) <
      candidates.indexOf(packagedIco)
  )
  assert.ok(
    candidates.indexOf(
      path.join('/Applications/Hermes.app/Contents/Resources/app.asar', 'public', 'apple-touch-icon.png')
    ) < candidates.indexOf(packagedIco)
  )
})

test('resolveAppIconPath prefers icns when Darwin resources include icns and ico', () => {
  const resourcesPath = '/Applications/Hermes.app/Contents/Resources'
  const icnsPath = path.join(resourcesPath, 'icon.icns')
  const icoPath = path.join(resourcesPath, 'icon.ico')
  const existing = new Set([icnsPath, icoPath])

  const resolved = resolveAppIconPath(
    { appRoot: '/Applications/Hermes.app/Contents/Resources/app.asar', resourcesPath, platform: 'darwin' },
    filePath => existing.has(filePath)
  )

  assert.equal(resolved, icnsPath)
})

test('resolveAppIconPath returns first existing candidate', () => {
  const existing = new Set([
    path.join('C:\\app', 'public', 'apple-touch-icon.png'),
    path.join('C:\\resources', 'icon.ico')
  ])

  const resolved = resolveAppIconPath(
    { appRoot: 'C:\\app', resourcesPath: 'C:\\resources', platform: 'win32' },
    filePath => existing.has(filePath)
  )

  assert.equal(resolved, path.join('C:\\resources', 'icon.ico'))
})

test('resolveAppIconPath falls back to apple-touch when ico missing', () => {
  const existing = new Set([path.join('/app', 'public', 'apple-touch-icon.png')])

  const resolved = resolveAppIconPath({ appRoot: '/app', resourcesPath: '/resources', platform: 'linux' }, filePath =>
    existing.has(filePath)
  )

  assert.equal(resolved, path.join('/app', 'public', 'apple-touch-icon.png'))
})

test('windows still prefers ico when packaged ico and png both exist', () => {
  const resourcesPath = 'C:\\resources'
  const icoPath = path.join(resourcesPath, 'icon.ico')
  const pngPath = path.join('C:\\app', 'assets', 'icon.png')
  const existing = new Set([icoPath, pngPath])

  const resolved = resolveAppIconPath({ appRoot: 'C:\\app', resourcesPath, platform: 'win32' }, filePath =>
    existing.has(filePath)
  )

  assert.equal(resolved, icoPath)
})
