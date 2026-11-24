#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { copyFileSync, existsSync, readFileSync, realpathSync, statSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

import { lockImporters } from './pnpm-lock.mjs'

// Do not require('electron'): Electron 42's entry point can acquire a binary.
// Only this explicit preparation phase may execute the installed installer.
// `run` runs the installer; node-deps passes one that kills it once its run loses the checkout lock.
const runInstaller = (file, args, options) => execFileSync(file, args, { ...options, stdio: 'inherit' })

/** @param {{ source: string, install?: boolean, env?: NodeJS.ProcessEnv, run?: typeof runInstaller }} options @returns {string} */
export function prepareElectronRuntime({ source, install = true, env = process.env, run = runInstaller }) {
  source = resolve(source)
  const link = join(source, 'apps/desktop/node_modules/electron/package.json')
  if (!existsSync(link)) throw new Error('Missing prepared Electron dependency; install the locked desktop workspace first')
  // The workspace entry is a link into pnpm's virtual store, where Electron's own
  // dependencies (@electron-internal/extract-zip) resolve from.
  const manifest = realpathSync(link)
  const packageRoot = dirname(manifest)
  const { version, dependencies } = JSON.parse(readFileSync(manifest, 'utf8'))
  const desktop = JSON.parse(readFileSync(join(source, 'apps/desktop/package.json'), 'utf8'))
  const locked = lockImporters(readFileSync(join(source, 'pnpm-lock.yaml'), 'utf8'))['apps/desktop']?.devDependencies?.electron
  // A locked version carries its peer resolution as a suffix: 44.9.3(supports-color@10.2.2).
  const lockedVersion = locked?.version?.replace(/\(.*$/, '')
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
  // install.js unpacks with @electron-internal/extract-zip, whose Windows binding imports
  // VCRUNTIME140.dll. A fresh Windows has none, so the load fails (ERR_DLOPEN_FAILED).
  // Node loads bindings with LOAD_WITH_ALTERED_SEARCH_PATH, so a copy beside the binding
  // is found first. HERMES_VC_RUNTIME is the copy our Python ships.
  if (process.platform === 'win32' && env.HERMES_VC_RUNTIME && dependencies?.['@electron-internal/extract-zip']) {
    const binding = dirname(createRequire(manifest).resolve('@electron-internal/extract-zip'))
    copyFileSync(env.HERMES_VC_RUNTIME, join(binding, 'vcruntime140.dll'))
  }
  console.log(`electron-runtime: preparing locked Electron ${version} (${process.platform}-${process.arch})...`)
  run(process.execPath, [join(packageRoot, 'install.js')], { cwd: source, env: installerEnv })
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
