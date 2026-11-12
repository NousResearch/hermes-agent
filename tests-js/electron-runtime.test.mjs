import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { prepareElectronRuntime } from '../scripts/build/electron-runtime.mjs'

const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function json(file, value) {
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, JSON.stringify(value))
}

function lock(version) {
  return ["lockfileVersion: '9.0'", '', 'importers:', '', '  apps/desktop:', '    devDependencies:', '      electron:',
    '        specifier: 42.11.12', `        version: ${version}`, ''].join('\n')
}

// pnpm's isolated layout: the workspace entry links into the virtual store, where
// Electron's own dependencies are its siblings.
function fixture() {
  const source = mkdtempSync(join(tmpdir(), 'electron preparation with spaces-'))
  roots.push(source)
  const slot = join(source, 'node_modules/.pnpm/electron@42.11.12/node_modules')
  const packageRoot = join(slot, 'electron')
  json(join(source, 'apps/desktop/package.json'), { devDependencies: { electron: '42.11.12' } })
  writeFileSync(join(source, 'pnpm-lock.yaml'), lock('42.11.12(supports-color@10.2.2)'))
  json(join(packageRoot, 'package.json'), { name: 'electron', version: '42.11.12', main: 'index.js', dependencies: { '@electron-internal/extract-zip': '1.0.0' } })
  json(join(slot, '@electron-internal/extract-zip/package.json'), { name: '@electron-internal/extract-zip', version: '1.0.0', main: 'index.js' })
  writeFileSync(join(slot, '@electron-internal/extract-zip/index.js'), 'module.exports = true')
  mkdirSync(join(source, 'apps/desktop/node_modules'))
  symlinkSync(packageRoot, join(source, 'apps/desktop/node_modules/electron'), 'junction')
  // Electron 42's lazy package entry point must never execute during validation.
  writeFileSync(join(packageRoot, 'index.js'), `require('node:fs').writeFileSync(__dirname + '/lazy-entry-executed', 'yes'); throw new Error('unexpected lazy entry')`)
  const binaryName = process.platform === 'darwin' ? 'Electron.app/Contents/MacOS/Electron'
    : process.platform === 'win32' ? 'electron.exe' : 'electron'
  writeFileSync(join(packageRoot, 'install.js'), `
    const fs = require('node:fs');
    const path = require('node:path');
    require('@electron-internal/extract-zip');
    if (process.env.ELECTRON_SKIP_BINARY_DOWNLOAD || process.env.ELECTRON_OVERRIDE_DIST_PATH) throw new Error('unexpected installer override');
    if (process.env.ELECTRON_INSTALL_PLATFORM !== process.platform || process.env.ELECTRON_INSTALL_ARCH !== process.arch) throw new Error('wrong host');
    fs.appendFileSync(path.join(__dirname, 'installer-calls'), 'install\\n');
    const binary = path.join(__dirname, 'dist', ${JSON.stringify(binaryName)});
    fs.mkdirSync(path.dirname(binary), { recursive: true });
    fs.writeFileSync(binary, 'synthetic executable');
    fs.writeFileSync(path.join(__dirname, 'dist/version'), 'v42.11.12');
    fs.writeFileSync(path.join(__dirname, 'path.txt'), ${JSON.stringify(binaryName)});
  `)
  return { source, packageRoot, executable: join(packageRoot, 'dist', binaryName) }
}

test('explicitly prepares a fresh locked runtime once, inside the virtual store', () => {
  const { source, packageRoot, executable } = fixture()
  const env = { ...process.env, ELECTRON_SKIP_BINARY_DOWNLOAD: '1', ELECTRON_OVERRIDE_DIST_PATH: 'foreign',
    ELECTRON_INSTALL_PLATFORM: 'foreign', ELECTRON_INSTALL_ARCH: 'foreign' }
  expect(prepareElectronRuntime({ source, env })).toBe(executable)
  expect(readFileSync(join(packageRoot, 'installer-calls'), 'utf8')).toBe('install\n')
  expect(prepareElectronRuntime({ source, install: false })).toBe(executable)
  expect(prepareElectronRuntime({ source })).toBe(executable)
  expect(readFileSync(join(packageRoot, 'installer-calls'), 'utf8')).toBe('install\n')
  expect(existsSync(join(packageRoot, 'lazy-entry-executed'))).toBe(false)
})

test('an Electron outside the desktop workspace is not a prepared dependency', () => {
  const { source, packageRoot } = fixture()
  rmSync(join(source, 'apps/desktop/node_modules/electron'))
  symlinkSync(packageRoot, join(source, 'node_modules/electron'), 'junction')
  expect(() => prepareElectronRuntime({ source })).toThrow(/Missing prepared Electron/)
  expect(existsSync(join(packageRoot, 'installer-calls'))).toBe(false)
})

test('read-only validation refuses missing or stale binaries without executing either entry point', () => {
  const { source, packageRoot, executable } = fixture()
  expect(() => prepareElectronRuntime({ source, install: false })).toThrow(/disabled/)
  expect(existsSync(join(packageRoot, 'installer-calls'))).toBe(false)
  prepareElectronRuntime({ source })
  rmSync(executable)
  expect(() => prepareElectronRuntime({ source, install: false })).toThrow(/disabled/)
  prepareElectronRuntime({ source })
  writeFileSync(join(packageRoot, 'dist/version'), 'v40.10.2')
  expect(() => prepareElectronRuntime({ source, install: false })).toThrow(/disabled/)
  expect(readFileSync(join(packageRoot, 'installer-calls'), 'utf8')).toBe('install\ninstall\n')
  expect(existsSync(join(packageRoot, 'lazy-entry-executed'))).toBe(false)
})

test('propagates installer failure and rejects success without a complete runtime', () => {
  const { source, packageRoot } = fixture()
  writeFileSync(join(packageRoot, 'install.js'), 'process.exit(13)')
  expect(() => prepareElectronRuntime({ source })).toThrow()
  writeFileSync(join(packageRoot, 'install.js'), '')
  expect(() => prepareElectronRuntime({ source })).toThrow(/did not produce/)
  expect(() => prepareElectronRuntime({ source, install: false })).toThrow(/disabled/)
})

test('refuses a foreign installed or locked version before running its installer', () => {
  const { source, packageRoot } = fixture()
  writeFileSync(join(source, 'pnpm-lock.yaml'), lock('41.10.6'))
  expect(() => prepareElectronRuntime({ source })).toThrow(/lock-pinned/)
  writeFileSync(join(source, 'pnpm-lock.yaml'), lock('42.11.12'))
  json(join(packageRoot, 'package.json'), { version: '41.10.6' })
  expect(() => prepareElectronRuntime({ source })).toThrow(/lock-pinned/)
  expect(existsSync(join(packageRoot, 'installer-calls'))).toBe(false)
})
