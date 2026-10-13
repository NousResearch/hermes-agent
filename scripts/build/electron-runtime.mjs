#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

// Do not require('electron'): Electron 42's entry point can acquire a binary.
// Only this explicit preparation phase may execute the installed installer.
/** @param {{ source: string, install?: boolean, env?: NodeJS.ProcessEnv }} options @returns {string} */
export function prepareElectronRuntime({ source, install = true, env = process.env }) {
  source = resolve(source)
  const manifest = [join(source, 'apps/desktop/node_modules/electron/package.json'),
    join(source, 'node_modules/electron/package.json')].find(existsSync)
  if (!manifest) throw new Error('Missing prepared Electron dependency; install the locked desktop workspace first')
  const packageRoot = dirname(manifest)
  const { version } = JSON.parse(readFileSync(manifest, 'utf8'))
  const desktop = JSON.parse(readFileSync(join(source, 'apps/desktop/package.json'), 'utf8'))
  const lock = JSON.parse(readFileSync(join(source, 'package-lock.json'), 'utf8'))
  const entry = lock.packages?.[relative(source, packageRoot).split('\\').join('/')]
  const lockedVersion = entry?.link ? lock.packages[entry.resolved]?.version : entry?.version
  if (version !== desktop.devDependencies?.electron || version !== lockedVersion) {
    throw new Error('Installed Electron is not the desktop lock-pinned version')
  }
  const binaryName = process.platform === 'darwin' ? 'Electron.app/Contents/MacOS/Electron'
    : process.platform === 'win32' ? 'electron.exe' : 'electron'
  const executable = join(packageRoot, 'dist', binaryName)
  const prepared = () => {
    try {
      return readFileSync(join(packageRoot, 'path.txt'), 'utf8').trim() === binaryName
        && readFileSync(join(packageRoot, 'dist/version'), 'utf8').trim().replace(/^v/, '') === version
        && statSync(executable).isFile()
    } catch {
      return false
    }
  }
  if (prepared()) return executable
  if (!install) throw new Error('Electron binary is missing or stale and lazy installs are disabled; run an explicit desktop build/update')
  const excluded = new Set(['electron_skip_binary_download', 'electron_override_dist_path',
    'electron_install_arch', 'electron_install_platform', 'npm_config_arch', 'npm_config_platform'])
  const installerEnv = Object.fromEntries(Object.entries(env).filter(([key]) => !excluded.has(key.toLowerCase())))
  Object.assign(installerEnv, { ELECTRON_INSTALL_PLATFORM: process.platform, ELECTRON_INSTALL_ARCH: process.arch,
    npm_config_platform: process.platform, npm_config_arch: process.arch })
  console.log(`electron-runtime: preparing locked Electron ${version} (${process.platform}-${process.arch})...`)
  execFileSync(process.execPath, [join(packageRoot, 'install.js')], { cwd: source, env: installerEnv, stdio: 'inherit' })
  if (!prepared()) throw new Error('Electron installer did not produce the prepared host runtime')
  return executable
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: {
    source: { type: 'string' }, 'no-install': { type: 'boolean', default: false },
  } })
  if (!values.source) throw new Error('--source is required')
  prepareElectronRuntime({ source: values.source, install: !values['no-install'] })
}
