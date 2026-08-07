import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

import { electronMirrorEnv, formatProblems, inspectBuildEnvironment } from './build-environment.mjs'

const repoRoot = path.resolve(import.meta.dirname, '..', '..', '..')
const env = electronMirrorEnv()
const electronRoot = path.join(repoRoot, 'node_modules', 'electron')
const initialProblems = inspectBuildEnvironment({ repoRoot })
const electronNeedsRepair = initialProblems.some(problem =>
  ['electron-install-mismatch', 'electron-binary-missing'].includes(problem.code)
)

if (electronNeedsRepair && fs.existsSync(electronRoot)) {
  console.warn('[desktop-install] Removing the stale or incomplete Electron package before reinstalling.')
  fs.rmSync(electronRoot, { recursive: true, force: true })
}

console.log(`[desktop-install] Electron mirror: ${env.ELECTRON_MIRROR}`)
console.log('[desktop-install] Installing the locked desktop workspace dependencies...')

const result = spawnSync(
  process.platform === 'win32' ? 'npm.cmd' : 'npm',
  ['install', '--workspace', 'apps/desktop', ...process.argv.slice(2)],
  { cwd: repoRoot, env, stdio: 'inherit' }
)

if (result.error) {
  console.error(`[desktop-install] Could not start npm: ${result.error.message}`)
  process.exit(1)
}

if (result.status !== 0) {
  console.error(
    '[desktop-install] npm failed to install the desktop dependencies.\n' +
      'Likely causes: a stale node_modules tree, a package-lock mismatch, or an unreachable Electron mirror.\n' +
      'Suggested fix: remove node_modules, run `npm ci`, and retry. If Electron downloads are blocked, set ELECTRON_MIRROR to a reachable mirror.'
  )
  process.exit(result.status == null ? 1 : result.status)
}

const remainingProblems = inspectBuildEnvironment({ repoRoot })
if (remainingProblems.length > 0) {
  console.error(formatProblems(remainingProblems, env.ELECTRON_MIRROR))
  process.exit(1)
}

console.log('[desktop-install] Desktop dependencies installed successfully.')
