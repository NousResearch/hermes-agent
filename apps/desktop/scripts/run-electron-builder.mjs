// Resolve electronDist at runtime (#38673, #47917): electron-builder 26.8.x can
// re-unpack a broken Electron.app; reusing the installed dist dodges that.
// npm workspace hoisting is non-deterministic — require.resolve finds electron
// wherever it landed. Dist present → -c.electronDist=<abs>/dist; absent → let
// electron-builder fetch via @electron/get (electronVersion + ELECTRON_MIRROR).

import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { createRequire } from 'node:module'

import { electronMirrorEnv, formatProblems, inspectBuildEnvironment } from './build-environment.mjs'

const require = createRequire(import.meta.url)
const repoRoot = path.resolve(import.meta.dirname, '..', '..', '..')
const env = electronMirrorEnv()
const problems = inspectBuildEnvironment({ repoRoot })

if (problems.length > 0) {
  console.error(formatProblems(problems, env.ELECTRON_MIRROR))
  process.exit(1)
}

function electronDistDir() {
  try {
    return path.join(path.dirname(require.resolve('electron/package.json')), 'dist')
  } catch {
    return null
  }
}

function distBinary(dist) {
  if (process.platform === 'darwin') {
    return path.join(dist, 'Electron.app', 'Contents', 'MacOS', 'Electron')
  }
  if (process.platform === 'win32') {
    return path.join(dist, 'electron.exe')
  }
  return path.join(dist, 'electron')
}

function electronBuilderCli() {
  const pkgJson = require.resolve('electron-builder/package.json')
  const bin = require(pkgJson).bin
  const rel = typeof bin === 'string' ? bin : bin['electron-builder']
  return path.join(path.dirname(pkgJson), rel)
}

const dist = electronDistDir()
const args = []
if (dist && fs.existsSync(distBinary(dist))) {
  args.push(`-c.electronDist=${dist}`)
} else {
  console.warn(
    '[run-electron-builder] no local electron dist; electron-builder will fetch ' +
      `via @electron/get using ELECTRON_MIRROR=${env.ELECTRON_MIRROR}.`
  )
}
args.push(...process.argv.slice(2))

const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
  env,
  stdio: 'inherit'
})
if (result.error) {
  console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
  process.exit(1)
}
if (result.status !== 0) {
  console.error(
    '[run-electron-builder] packaging failed.\n' +
      'Likely causes: an incomplete Electron download, stale node_modules, or an Electron version mismatch.\n' +
      'Suggested fix: run `npm run check:desktop-install`, then remove node_modules and run `npm ci` if the preflight reports damage.\n' +
      `Electron mirror: ${env.ELECTRON_MIRROR}`
  )
}
process.exit(result.status == null ? 1 : result.status)
