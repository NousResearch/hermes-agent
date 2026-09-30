import fs from 'node:fs'
import path from 'node:path'

import { buildDesktopBackendEnv } from './backend-env'

interface BundledRuntimeOptions {
  isPackaged: boolean
  resourcesPath: string
  hermesHome: string
  args: string[]
  platform?: string
  arch?: string
  currentEnv?: NodeJS.ProcessEnv
}

// The installer owns this immutable runtime. User profiles stay outside the
// application and upgrading it replaces code, never user state.
//
// Windows always ships a runtime. macOS ships one only in DMGs built by
// scripts/build-macos-installer.sh; a DMG without `runtime/manifest.json`
// returns null so the first-run bootstrap (install.sh) still works online.
export function bundledRuntimeBackend({
  isPackaged,
  resourcesPath,
  hermesHome,
  args,
  platform = process.platform,
  arch = process.arch,
  currentEnv = process.env
}: BundledRuntimeOptions) {
  const isWindows = platform === 'win32'

  if (!isPackaged || (!isWindows && platform !== 'darwin')) {
    return null
  }

  const bundle = path.join(resourcesPath, 'runtime')
  const root = path.join(bundle, 'agent')
  const command = isWindows ? path.join(bundle, 'python', 'python.exe') : path.join(bundle, 'python', 'bin', 'python3')
  const manifestPath = path.join(bundle, 'manifest.json')

  if (!isWindows && !fs.existsSync(manifestPath)) {
    return null
  }

  const requiredFiles = [manifestPath, command, path.join(root, 'hermes_cli', 'main.py')]

  if (isWindows) {
    requiredFiles.push(path.join(bundle, 'git', 'bin', 'bash.exe'), path.join(bundle, 'node', 'node.exe'))
  }

  if (!requiredFiles.every(file => fs.existsSync(file))) {
    throw new Error('Brakuje plików silnika Agenta Cześka. Zainstaluj ponownie pełny pakiet aplikacji.')
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))

  if (manifest.schemaVersion !== 1 || manifest.platform !== platform || manifest.arch !== arch) {
    throw new Error('Pakiet silnika Agenta Cześka nie pasuje do tej wersji aplikacji.')
  }

  const environment = buildDesktopBackendEnv({ hermesHome, currentEnv })

  for (const key of Object.keys(environment)) {
    if (key.toUpperCase() === 'PATH') {delete environment[key]}
  }

  // macOS keeps the system bash/git (Xcode tools); only Python and the optional
  // Node are bundled, so they go first and the user's PATH stays behind them.
  const pathEntries = isWindows
    ? [
        path.join(bundle, 'python'),
        path.join(bundle, 'node'),
        path.join(bundle, 'git', 'cmd'),
        path.join(bundle, 'git', 'usr', 'bin')
      ]
    : [path.join(bundle, 'python', 'bin'), path.join(bundle, 'node', 'bin')]

  return {
    kind: 'python',
    label: 'wbudowany silnik Agenta Cześka',
    command,
    args: ['-m', 'hermes_cli.main', ...args],
    root,
    bootstrap: false,
    shell: false,
    env: {
      ...environment,
      PYTHONPATH: root,
      PATH: [...pathEntries, currentEnv.PATH || currentEnv.Path || ''].filter(Boolean).join(path.delimiter),
      ...(isWindows ? { HERMES_GIT_BASH_PATH: path.join(bundle, 'git', 'bin', 'bash.exe') } : {}),
      PYTHONHOME: '',
      VIRTUAL_ENV: '',
      PYTHONNOUSERSITE: '1',
      PYTHONDONTWRITEBYTECODE: '1'
    }
  }
}
