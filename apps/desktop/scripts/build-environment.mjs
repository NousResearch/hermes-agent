import fs from 'node:fs'
import path from 'node:path'

export const DEFAULT_ELECTRON_MIRROR = 'https://github.com/electron/electron/releases/download/'

export function electronMirror(env = process.env) {
  return env.ELECTRON_MIRROR || env.npm_config_electron_mirror || DEFAULT_ELECTRON_MIRROR
}

export function electronMirrorEnv(env = process.env) {
  const mirror = electronMirror(env)
  return {
    ...env,
    ELECTRON_MIRROR: mirror,
    npm_config_electron_mirror: env.npm_config_electron_mirror || mirror
  }
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function packageVersion(repoRoot, name) {
  const packageFile = path.join(repoRoot, 'node_modules', ...name.split('/'), 'package.json')
  if (!fs.existsSync(packageFile)) return null
  return readJson(packageFile).version || null
}

function electronBinary(repoRoot, platform) {
  const dist = path.join(repoRoot, 'node_modules', 'electron', 'dist')
  if (platform === 'darwin') {
    return path.join(dist, 'Electron.app', 'Contents', 'MacOS', 'Electron')
  }
  if (platform === 'win32') return path.join(dist, 'electron.exe')
  return path.join(dist, 'electron')
}

function nodeVersionSupported(version) {
  const match = String(version || '').match(/^(\d+)\.(\d+)/)
  if (!match) return false
  const major = Number(match[1])
  const minor = Number(match[2])
  return (major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major > 22
}

export function inspectBuildEnvironment({
  repoRoot,
  platform = process.platform,
  nodeVersion = process.versions.node
} = {}) {
  if (!repoRoot) throw new Error('repoRoot is required')

  const problems = []
  const rootPackageFile = path.join(repoRoot, 'package.json')
  const desktopPackageFile = path.join(repoRoot, 'apps', 'desktop', 'package.json')
  const lockFile = path.join(repoRoot, 'package-lock.json')
  const installLockFile = path.join(repoRoot, 'node_modules', '.package-lock.json')

  if (!fs.existsSync(rootPackageFile) || !fs.existsSync(desktopPackageFile)) {
    problems.push({
      code: 'incomplete-checkout',
      cause: 'The build is not running from a complete Hermes repository checkout.',
      fix: `Run the command from the repository root: ${repoRoot}`
    })
    return problems
  }

  if (!nodeVersionSupported(nodeVersion)) {
    problems.push({
      code: 'unsupported-node',
      cause: `Node ${nodeVersion || 'unknown'} is incompatible; Hermes Desktop requires Node ^20.19.0 or >=22.12.0.`,
      fix: 'Install a supported Node release, remove node_modules, then run `npm ci` again.'
    })
  }

  if (!fs.existsSync(lockFile)) {
    problems.push({
      code: 'missing-lockfile',
      cause: 'package-lock.json is missing, so dependency versions cannot be verified.',
      fix: 'Restore package-lock.json from git before installing or building.'
    })
  }

  if (!fs.existsSync(installLockFile)) {
    problems.push({
      code: 'incomplete-node-modules',
      cause: 'node_modules is missing or was created by an incomplete npm install.',
      fix: 'From the repo root, run `npm ci`. If that fails, remove node_modules and retry.'
    })
  }

  const desktopPackage = readJson(desktopPackageFile)
  const required = ['vite', 'electron', 'electron-builder']
  for (const name of required) {
    if (!packageVersion(repoRoot, name)) {
      problems.push({
        code: 'missing-package',
        cause: `Required desktop dependency ${name} is missing from node_modules.`,
        fix: 'From the repo root, run `npm ci` to restore the locked workspace dependencies.'
      })
    }
  }

  const configuredElectron = desktopPackage.build?.electronVersion
  const declaredElectron = desktopPackage.devDependencies?.electron?.replace(/^[^\d]*/, '')
  const installedElectron = packageVersion(repoRoot, 'electron')

  if (configuredElectron && declaredElectron && configuredElectron !== declaredElectron) {
    problems.push({
      code: 'electron-config-mismatch',
      cause: `Electron version mismatch: build.electronVersion is ${configuredElectron}, but devDependencies requests ${desktopPackage.devDependencies.electron}.`,
      fix: 'Make the Electron versions in apps/desktop/package.json match, then refresh package-lock.json.'
    })
  }

  if (configuredElectron && installedElectron && configuredElectron !== installedElectron) {
    problems.push({
      code: 'electron-install-mismatch',
      cause: `Stale node_modules: Electron ${installedElectron} is installed, but the build requires ${configuredElectron}.`,
      fix: 'Remove node_modules and run `npm ci` from the repo root.'
    })
  }

  if (installedElectron && !fs.existsSync(electronBinary(repoRoot, platform))) {
    problems.push({
      code: 'electron-binary-missing',
      cause: `Electron ${installedElectron} is installed, but its runtime binary is missing. The download may have been interrupted or blocked by a mirror.`,
      fix: 'Remove node_modules/electron, then run `npm run install:desktop`. Set ELECTRON_MIRROR if GitHub downloads are unavailable.'
    })
  }

  return problems
}

export function formatProblems(problems, mirror) {
  const details = problems
    .map((problem, index) => `${index + 1}. Likely cause: ${problem.cause}\n   Suggested fix: ${problem.fix}`)
    .join('\n')
  return (
    '[desktop-build] Dependency preflight failed.\n' + `${details}\n` + `[desktop-build] Electron mirror: ${mirror}`
  )
}
