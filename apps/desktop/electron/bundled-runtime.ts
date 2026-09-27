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

// The Windows installer owns this immutable runtime. User profiles stay outside
// Program Files and upgrading the application replaces code, never user state.
export function bundledRuntimeBackend({
  isPackaged,
  resourcesPath,
  hermesHome,
  args,
  platform = process.platform,
  arch = process.arch,
  currentEnv = process.env
}: BundledRuntimeOptions) {
  if (!isPackaged || platform !== 'win32') {
    return null
  }

  const bundle = path.join(resourcesPath, 'runtime')
  const root = path.join(bundle, 'agent')
  const command = path.join(bundle, 'python', 'python.exe')
  const manifestPath = path.join(bundle, 'manifest.json')

  if (
    ![
      manifestPath,
      command,
      path.join(root, 'hermes_cli', 'main.py'),
      path.join(bundle, 'git', 'bin', 'bash.exe'),
      path.join(bundle, 'node', 'node.exe')
    ].every(file => fs.existsSync(file))
  ) {
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
      PATH: [
        path.join(bundle, 'python'),
        path.join(bundle, 'node'),
        path.join(bundle, 'git', 'cmd'),
        path.join(bundle, 'git', 'usr', 'bin'),
        currentEnv.PATH || currentEnv.Path || ''
      ]
        .filter(Boolean)
        .join(path.delimiter),
      HERMES_GIT_BASH_PATH: path.join(bundle, 'git', 'bin', 'bash.exe'),
      PYTHONHOME: '',
      VIRTUAL_ENV: '',
      PYTHONNOUSERSITE: '1',
      PYTHONDONTWRITEBYTECODE: '1'
    }
  }
}
