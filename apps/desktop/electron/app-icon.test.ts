import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { appIconCandidates, decodingFileProbe, resolveAppIcon, shouldOverrideDockIcon } from './app-icon'

// Regression: a packaged app.asar can contain a TRUNCATED apple-touch-icon.png
// (interrupted electron-builder run, partial copy). Electron's
// BrowserWindow({ icon }) / app.dock.setIcon() decode synchronously and THROW
// on undecodable bytes, which killed the main process inside createWindow()
// and took the app down mid-session. Icon resolution must fail soft: skip a
// candidate that exists but does not decode, exactly like a missing one.

test('resolveAppIcon skips an existing but undecodable candidate', () => {
  // First candidate "exists" (probe says true) but does not decode; second
  // decodes. The resolver must return the second, not the first.
  const probe = (p: string) => p !== '/packaged/app.asar/public/apple-touch-icon.png'

  const picked = resolveAppIcon(
    ['/packaged/app.asar/public/apple-touch-icon.png', '/packaged/app.asar/dist/apple-touch-icon.png'],
    probe
  )

  assert.equal(picked, '/packaged/app.asar/dist/apple-touch-icon.png')
})

test('resolveAppIcon returns undefined when every candidate fails the probe', () => {
  const picked = resolveAppIcon(['/a.png', '/b.ico'], () => false)
  assert.equal(picked, undefined)
})

test('resolveAppIcon returns the first candidate that passes the probe', () => {
  const picked = resolveAppIcon(['/a.png', '/b.png'], () => true)
  assert.equal(picked, '/a.png')
})

test('decodingFileProbe rejects a missing file', () => {
  const missing = path.join(os.tmpdir(), `hermes-icon-missing-${process.pid}.png`)
  assert.equal(decodingFileProbe(missing), false)
})

test('decodingFileProbe rejects a directory', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-icon-dir-'))

  try {
    assert.equal(decodingFileProbe(dir), false)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('appIconCandidates keeps the documented precedence ladder', () => {
  const mac = appIconCandidates({
    isWindows: false,
    appRoot: '/Applications/Hermes.app/Contents/Resources',
    unpackedPathFor: p => `${p}.unpacked`
  })

  assert.deepEqual(mac, [
    path.join('/Applications/Hermes.app/Contents/Resources', 'public', 'apple-touch-icon.png'),
    path.join('/Applications/Hermes.app/Contents/Resources', 'dist', 'apple-touch-icon.png'),
    path.join('/Applications/Hermes.app/Contents/Resources.unpacked', 'dist', 'apple-touch-icon.png')
  ])

  // Windows prepends the two full-bleed .ico rungs ahead of the PNG ladder.
  const win = appIconCandidates({
    isWindows: true,
    appRoot: 'C:\\app',
    resourcesPath: 'C:\\resources',
    unpackedPathFor: p => `${p}\\unpacked`
  })

  const isIco = win.map(c => c.endsWith('.ico'))
  assert.ok(isIco.lastIndexOf(true) < isIco.indexOf(false), 'every .ico rung precedes the PNG ladder')
  assert.equal(
    win[0],
    path.join('C:\\resources', 'icon.ico'),
    'resources/ icon.ico is the highest-precedence Windows rung'
  )
})

// #73195: a runtime app.dock.setIcon(png) replaces the bundle's icon for the
// life of the process, so macOS 26 cannot apply the Clear/Tinted Liquid Glass
// styles to it. A packaged .app already carries the Hermes icon; only dev runs
// (the stock Electron bundle) need the runtime override.
test('shouldOverrideDockIcon leaves a packaged macOS app on its bundle icon', () => {
  assert.equal(shouldOverrideDockIcon({ platform: 'darwin', isPackaged: true }), false)
})

test('shouldOverrideDockIcon still brands the dock in a macOS dev run', () => {
  assert.equal(shouldOverrideDockIcon({ platform: 'darwin', isPackaged: false }), true)
})

test('shouldOverrideDockIcon never applies off macOS (there is no dock)', () => {
  for (const platform of ['win32', 'linux'] as const) {
    assert.equal(shouldOverrideDockIcon({ platform, isPackaged: false }), false)
    assert.equal(shouldOverrideDockIcon({ platform, isPackaged: true }), false)
  }
})
