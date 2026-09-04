// Wraps the electron-builder CLI so the arguments compose in one place, in
// the first spawn with no shell in between.
//
// electron-builder downloads and extracts Electron itself (via electronVersion
// + ELECTRON_MIRROR). Passing the local Electron dist makes v27 copy framework
// links as regular files. Its archive extraction preserves those links.

import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

function electronBuilderCli() {
  const entry = require.resolve('electron-builder')
  let dir = path.dirname(entry)
  while (!fs.existsSync(path.join(dir, 'package.json'))) {
    const parent = path.dirname(dir)
    if (parent === dir) {
      throw new Error('electron-builder package root not found')
    }
    dir = parent
  }
  const bin = require(path.join(dir, 'package.json')).bin
  const rel = typeof bin === 'string' ? bin : bin['electron-builder']
  return path.join(dir, rel)
}

// Resolve electronDist at runtime (#38673, #47917): electron-builder 26.8.x can
// re-unpack a broken Electron.app; reusing the installed dist dodges that.
// npm workspace hoisting is non-deterministic — require.resolve finds electron
// wherever it landed.
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

const args = [...process.argv.slice(2)]

// package.json has no "build" field. Name the config file or electron-builder
// would look for one and silently use defaults.
if (!args.some(a => a === '--config' || a.startsWith('--config='))) {
  args.push('--config', 'electron-builder.config.cjs')
}

// Never let electron-builder publish. On a CI tag build it auto-detects
// GitHub and demands GH_TOKEN after the artifacts are already built.
// The release workflow uploads artifacts in its own step. (Also: the npm
// lifecycle env sets CI=1, and electron-builder treats CI=1 as a signal to
// implicitly resolve a publish target, which reads <projectDir>/.git/config —
// apps/desktop has no .git of its own, so pin publish to "never".)
if (!args.includes('--publish') && !args.some(a => a.startsWith('-p'))) {
  args.push('--publish', 'never')
}

// Reuse the installed Electron dist when present so a broken app is not
// re-unpacked from a fresh download; otherwise electron-builder fetches via
// @electron/get (electronVersion + ELECTRON_MIRROR).
const dist = electronDistDir()
if (dist && fs.existsSync(distBinary(dist))) {
  args.push(`-c.electronDist=${dist}`)
} else {
  console.warn(
    '[run-electron-builder] no local electron dist; electron-builder will fetch ' +
      'via @electron/get (electronVersion + ELECTRON_MIRROR).'
  )
}

if (args.includes('--win') && process.env.AZURE_SIGN_ENDPOINT && process.env.AZURE_CLIENT_ID) {
  console.log(
    `[run-electron-builder] Windows signing: Azure Trusted Signing at ${process.env.AZURE_SIGN_ENDPOINT}`
  )
}

// Cap concurrent fs.open calls in the electron-builder process.
// @electron/osx-sign walks the whole .app with Promise.all and no
// concurrency bound. The payload (lark_oapi alone is thousands of files)
// exhausts the macOS table: EMFILE on a random .py. Raising ulimit only
// moves the ceiling. --require, not an import: isbinaryfile captures
// promisify(fs.open) at its own load.
const preload = path.join(import.meta.dirname, 'fs-open-limit.cjs')

const result = spawnSync(process.execPath, ['--require', preload, electronBuilderCli(), ...args], {
  stdio: 'inherit'
})
if (result.error) {
  console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)
