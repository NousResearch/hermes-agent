import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { appIconCandidates, decodingFileProbe, resolveAppIcon } from './app-icon'

const require = createRequire(import.meta.url)
const __dirname = path.dirname(fileURLToPath(import.meta.url))

const pkg = require('../package.json') as {
  author: string
  build: {
    appId: string
    artifactName: string
    icon: string
    linux: {
      desktop: { entry: Record<string, string> }
      maintainer: string
      synopsis: string
      target: readonly string[]
    }
    nsis: Record<string, unknown>
    productName: string
    protocols: readonly { name: string; schemes: readonly string[] }[]
  }
  description: string
  productName: string
  repository: {
    type: string
    url: string
  }
}

// Regression: a packaged app.asar can contain a TRUNCATED apple-touch-icon.png
// (interrupted electron-builder run, partial copy). Electron's
// BrowserWindow({ icon }) / app.dock.setIcon() decode synchronously and THROW
// on undecodable bytes, which killed the main process inside createWindow()
// and took the app down mid-session. Icon resolution must fail soft: skip a
// candidate that exists but does not decode, exactly like a missing one.

test('resolveAppIcon skips an existing but undecodable candidate', () => {
  // First candidate "exists" (probe says true) but does not decode; second
  // decodes. The resolver must return the second, not the first.
  const probeCalls: string[] = []

  const probe = (p: string) => {
    probeCalls.push(p)

    return p !== '/packaged/app.asar/public/apple-touch-icon.png'
  }

  const picked = resolveAppIcon(
    ['/packaged/app.asar/public/apple-touch-icon.png', '/packaged/app.asar/dist/apple-touch-icon.png'],
    probe
  )

  assert.equal(picked, '/packaged/app.asar/dist/apple-touch-icon.png')
  assert.deepEqual(probeCalls, [
    '/packaged/app.asar/public/apple-touch-icon.png',
    '/packaged/app.asar/dist/apple-touch-icon.png'
  ])
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

test('decodingFileProbe rejects an existing but empty (0-byte) file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-icon-'))
  const empty = path.join(dir, 'apple-touch-icon.png')
  fs.writeFileSync(empty, Buffer.alloc(0))

  try {
    // 0 bytes exist but decode to an empty image — and without electron in
    // the test runtime the require itself fails. Both paths must be false.
    assert.equal(decodingFileProbe(empty), false)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
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

  assert.equal(win.length, 5)
  assert.equal(win.filter(c => c.endsWith('.ico')).length, 2)
  assert.equal(
    win[0],
    path.join('C:\\resources', 'icon.ico'),
    'resources/ icon.ico is the highest-precedence Windows rung'
  )
  assert.equal(
    win.filter(c => c.endsWith('apple-touch-icon.png')).length,
    3,
    'all three PNG rungs remain after the ico rungs'
  )
})

test('package metadata uses the Agent Czesiek app identity consistently', () => {
  assert.equal(pkg.productName, 'Agent Czesiek')
  assert.equal(pkg.description, 'AI Evolution Jarvis, powered by Hermes Agent.')
  assert.equal(pkg.author, 'AI Evolution')
  assert.equal(pkg.repository.url, 'git+https://github.com/aievolutionpl/AGENT_CZESIEK.git')
  assert.equal(pkg.build.productName, pkg.productName)
  assert.equal(pkg.build.appId, 'pl.aievolution.jarvis')
  assert.equal(pkg.build.linux.maintainer, 'AI Evolution')
  assert.equal(pkg.build.linux.synopsis, pkg.description)

  const schemes = pkg.build.protocols.flatMap(protocol => protocol.schemes)
  assert.ok(schemes.includes('aievolution-jarvis'))
  assert.ok(!schemes.includes('hermes'))

  assert.ok(pkg.build.artifactName.startsWith('AI-Evolution-Jarvis-'))

  for (const token of ['${version}', '${os}', '${arch}', '${ext}']) {
    assert.ok(pkg.build.artifactName.includes(token), `artifactName keeps ${token}`)
  }
})

test('build metadata keeps the existing neutral icon fallback assets present', () => {
  assert.equal(pkg.build.icon, 'assets/icon')

  const desktopRoot = path.resolve(__dirname, '..')
  const iconBase = path.join(desktopRoot, pkg.build.icon)

  for (const ext of ['.png', '.ico', '.icns']) {
    assert.equal(fs.statSync(`${iconBase}${ext}`).isFile(), true)
  }
})

// The "simple install" contract: a user who runs the installer must end up
// with an icon they can double-click, in their own language, without reading
// anything. Each assertion below is one way that has broken before.
test('the Windows installer creates the shortcuts a non-technical install depends on', () => {
  // "always" (not the `true` default) re-creates the icon on a repair/update
  // run after the user deleted it.
  assert.equal(pkg.build.nsis.createDesktopShortcut, 'always')
  assert.equal(pkg.build.nsis.createStartMenuShortcut, true)
  assert.equal(pkg.build.nsis.shortcutName, pkg.productName)
  // Per-user, no elevation prompt, and the app opens when the installer closes.
  assert.equal(pkg.build.nsis.perMachine, false)
  assert.equal(pkg.build.nsis.runAfterFinish, true)
  // Uninstalling must not take the user's sessions and config with it.
  assert.equal(pkg.build.nsis.deleteAppDataOnUninstall, false)

  const languages = pkg.build.nsis.installerLanguages as readonly string[]
  assert.equal(pkg.build.nsis.multiLanguageInstaller, true)
  assert.equal(languages[0], 'pl_PL', 'Polish is the product language, so it leads')
  assert.ok(languages.includes('en_US'))
})

test('the Linux packages register a launcher that menus can actually find', () => {
  const entry = pkg.build.linux.desktop.entry

  assert.equal(entry.Type, 'Application')
  assert.equal(entry.Terminal, 'false')
  // Without StartupWMClass the running window is not grouped with its launcher
  // (a second, generic icon appears in the dock/taskbar instead).
  assert.equal(entry.StartupWMClass, 'AI Evolution Jarvis')
  assert.ok(entry.Categories.includes('Utility;'), 'an uncategorised entry lands in "Other"')
  assert.ok(entry.Keywords.includes('asystent'), 'Polish search terms find it too')

  for (const target of ['AppImage', 'deb', 'rpm']) {
    assert.ok(pkg.build.linux.target.includes(target))
  }
})
