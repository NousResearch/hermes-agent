import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs'
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

function fixture(layout = 'apps/desktop/node_modules/electron') {
  const source = mkdtempSync(join(tmpdir(), 'electron preparation with spaces-'))
  roots.push(source)
  const packageRoot = join(source, layout)
  json(join(source, 'apps/desktop/package.json'), { devDependencies: { electron: '42.11.12' } })
  json(join(source, 'package-lock.json'), { packages: { [layout]: { version: '42.11.12' } } })
  json(join(packageRoot, 'package.json'), { name: 'electron', version: '42.11.12', main: 'index.js' })
  // Electron 42's lazy package entry point must never execute during validation.
  writeFileSync(join(packageRoot, 'index.js'), `require('node:fs').writeFileSync(__dirname + '/lazy-entry-executed', 'yes'); throw new Error('unexpected lazy entry')`)
  const binaryName = process.platform === 'darwin' ? 'Electron.app/Contents/MacOS/Electron'
    : process.platform === 'win32' ? 'electron.exe' : 'electron'
  writeFileSync(join(packageRoot, 'install.js'), `
    const fs = require('node:fs');
    const path = require('node:path');
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

test.each(['apps/desktop/node_modules/electron', 'node_modules/electron'])('explicitly prepares a fresh locked %s runtime once', layout => {
  const { source, packageRoot, executable } = fixture(layout)
  const env = { ...process.env, ELECTRON_SKIP_BINARY_DOWNLOAD: '1', ELECTRON_OVERRIDE_DIST_PATH: 'foreign',
    ELECTRON_INSTALL_PLATFORM: 'foreign', ELECTRON_INSTALL_ARCH: 'foreign' }
  expect(prepareElectronRuntime({ source, env })).toBe(executable)
  expect(readFileSync(join(packageRoot, 'installer-calls'), 'utf8')).toBe('install\n')
  expect(prepareElectronRuntime({ source, install: false })).toBe(executable)
  expect(prepareElectronRuntime({ source })).toBe(executable)
  expect(readFileSync(join(packageRoot, 'installer-calls'), 'utf8')).toBe('install\n')
  expect(existsSync(join(packageRoot, 'lazy-entry-executed'))).toBe(false)
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

test('refuses a foreign installed version before running its installer', () => {
  const { source, packageRoot } = fixture()
  json(join(packageRoot, 'package.json'), { version: '41.10.6' })
  expect(() => prepareElectronRuntime({ source })).toThrow(/lock-pinned/)
  expect(existsSync(join(packageRoot, 'installer-calls'))).toBe(false)
})
