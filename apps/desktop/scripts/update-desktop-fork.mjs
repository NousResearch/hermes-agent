#!/usr/bin/env node
// ============================================================================
// update-desktop-fork.mjs
// ============================================================================
// Cross-platform (Linux/macOS/Windows) one-shot updater: rebuilds the
// Electron desktop app from the current hermes-agent checkout and hot-swaps
// the built `resources/` (app.asar + app.asar.unpacked) into the ALREADY
// INSTALLED app, without re-running the full installer (.deb/.dmg/.exe)
// every time.
//
// Usage (from apps/desktop/):
//   npm run update:fork                  # build from current worktree
//   npm run update:fork -- --pull        # git pull first
//   npm run update:fork -- --no-relaunch # skip killing/relaunching the app
//   npm run update:fork -- --install-dir /custom/path   # override install dir
//
// What "install dir" means per OS (electron-builder defaults; override with
// --install-dir if you installed somewhere nonstandard):
//   linux:   /opt/Hermes                         (deb/rpm install, root-owned)
//   darwin:  /Applications/Hermes.app/Contents    (Resources/ lives under here)
//   win32:   %LOCALAPPDATA%\Programs\Hermes       (nsis perMachine:false, per-user)
//
// Only the `resources/` directory is swapped — that's where app.asar (your
// code) lives. The Electron/Chromium binary itself is left alone; if you
// bump the `electron` devDependency you need a real full reinstall instead.
// ============================================================================

import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DESKTOP_DIR = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_DIR, '..', '..')
const PLATFORM = process.platform
const ARCH = process.arch

const args = process.argv.slice(2)
const flags = {
  pull: args.includes('--pull'),
  relaunch: !args.includes('--no-relaunch'),
  installDir: (() => {
    const i = args.indexOf('--install-dir')
    return i >= 0 ? args[i + 1] : null
  })(),
}

function log(msg) {
  console.log(`==> ${msg}`)
}

function run(cmd, cmdArgs, opts = {}) {
  execFileSync(cmd, cmdArgs, { stdio: 'inherit', ...opts })
}

// --------------------------------------------------------------------------
// 1. Optional git pull
// --------------------------------------------------------------------------
if (flags.pull) {
  log('git pull --ff-only')
  run('git', ['pull', '--ff-only'], { cwd: REPO_ROOT })
}

// --------------------------------------------------------------------------
// 2. Build (renderer + electron-main + electron-builder --dir, no installer)
// --------------------------------------------------------------------------
log('Building desktop app (npm run pack)')
run(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'pack'], { cwd: DESKTOP_DIR })

// --------------------------------------------------------------------------
// 3. Locate the freshly built resources/ directory
// --------------------------------------------------------------------------
function findBuiltResourcesDir() {
  const releaseDir = path.join(DESKTOP_DIR, 'release')
  const candidates = []
  if (PLATFORM === 'linux') {
    candidates.push(ARCH === 'arm64' ? 'linux-arm64-unpacked' : 'linux-unpacked')
  } else if (PLATFORM === 'win32') {
    candidates.push(
      ARCH === 'arm64' ? 'win-arm64-unpacked' : ARCH === 'ia32' ? 'win-ia32-unpacked' : 'win-unpacked',
    )
  } else if (PLATFORM === 'darwin') {
    candidates.push(ARCH === 'arm64' ? 'mac-arm64' : 'mac')
  }

  for (const c of candidates) {
    const dir = path.join(releaseDir, c)
    if (fs.existsSync(dir)) {
      if (PLATFORM === 'darwin') {
        // release/mac[-arm64]/Hermes.app/Contents/Resources
        const appDirs = fs.readdirSync(dir).filter((f) => f.endsWith('.app'))
        if (appDirs.length > 0) {
          return path.join(dir, appDirs[0], 'Contents', 'Resources')
        }
      } else {
        return path.join(dir, 'resources')
      }
    }
  }
  return null
}

const builtResources = findBuiltResourcesDir()
if (!builtResources || !fs.existsSync(builtResources)) {
  console.error(
    `ERROR: could not find built resources dir under ${path.join(DESKTOP_DIR, 'release')}. ` +
      'Check the electron-builder output directory name for this platform/arch and adjust the script.',
  )
  process.exit(1)
}
log(`Built resources: ${builtResources}`)

// --------------------------------------------------------------------------
// 4. Resolve the installed app's resources/ directory
// --------------------------------------------------------------------------
function defaultInstallResourcesDir() {
  if (flags.installDir) {
    return PLATFORM === 'darwin'
      ? path.join(flags.installDir, 'Contents', 'Resources')
      : path.join(flags.installDir, 'resources')
  }
  if (PLATFORM === 'linux') {
    return '/opt/Hermes/resources'
  }
  if (PLATFORM === 'darwin') {
    return '/Applications/Hermes.app/Contents/Resources'
  }
  if (PLATFORM === 'win32') {
    const localAppData = process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local')
    return path.join(localAppData, 'Programs', 'Hermes', 'resources')
  }
  throw new Error(`Unsupported platform: ${PLATFORM}`)
}

const installResources = defaultInstallResourcesDir()
if (!fs.existsSync(installResources)) {
  console.error(
    `ERROR: installed resources dir not found at ${installResources}.\n` +
      'Pass --install-dir <path to app root> to point at a nonstandard install location.',
  )
  process.exit(1)
}
log(`Install target: ${installResources}`)

// --------------------------------------------------------------------------
// 5. Copy: only app.asar + app.asar.unpacked need to move; leave
//    app-update.yml / icons alone so we don't fight the platform installer's
//    own metadata unnecessarily. Requires write permission on the install
//    dir — on Linux/macOS system installs that means running this whole
//    script with sudo; on Windows per-user (LOCALAPPDATA) installs it does
//    not.
// --------------------------------------------------------------------------
function copyResourceFile(name) {
  const src = path.join(builtResources, name)
  const dest = path.join(installResources, name)
  if (!fs.existsSync(src)) return
  fs.rmSync(dest, { recursive: true, force: true })
  fs.cpSync(src, dest, { recursive: true })
  console.log(`   copied ${name}`)
}

log('Copying app.asar + app.asar.unpacked into install dir')
try {
  copyResourceFile('app.asar')
  copyResourceFile('app.asar.unpacked')
  copyResourceFile('install-stamp.json')
} catch (err) {
  console.error(`ERROR: failed to write to ${installResources}: ${err.message}`)
  if (PLATFORM !== 'win32') {
    console.error('Try re-running this script with sudo (system install dirs are root-owned).')
  }
  process.exit(1)
}

// --------------------------------------------------------------------------
// 6. Report the new build stamp
// --------------------------------------------------------------------------
const stampPath = path.join(installResources, 'install-stamp.json')
if (fs.existsSync(stampPath)) {
  console.log('Installed build info:')
  console.log(fs.readFileSync(stampPath, 'utf8'))
}

// --------------------------------------------------------------------------
// 7. Relaunch
// --------------------------------------------------------------------------
function killAndRelaunch() {
  if (PLATFORM === 'linux') {
    try {
      execFileSync('pkill', ['-f', '^/opt/Hermes/Hermes( |$)'])
    } catch {
      /* not running */
    }
    setTimeout(() => {
      spawn('/opt/Hermes/Hermes', [], { detached: true, stdio: 'ignore' }).unref()
      console.log('Relaunched.')
    }, 1000)
  } else if (PLATFORM === 'darwin') {
    try {
      execFileSync('pkill', ['-f', 'Hermes.app/Contents/MacOS/Hermes'])
    } catch {
      /* not running */
    }
    setTimeout(() => {
      spawn('open', ['-a', 'Hermes'], { detached: true, stdio: 'ignore' }).unref()
      console.log('Relaunched.')
    }, 1000)
  } else if (PLATFORM === 'win32') {
    try {
      execFileSync('taskkill', ['/IM', 'Hermes.exe', '/F'])
    } catch {
      /* not running */
    }
    setTimeout(() => {
      const exe = path.join(path.dirname(installResources), 'Hermes.exe')
      spawn(exe, [], { detached: true, stdio: 'ignore' }).unref()
      console.log('Relaunched.')
    }, 1000)
  }
}

if (flags.relaunch) {
  log('Restarting Hermes desktop app')
  killAndRelaunch()
} else {
  console.log('Skipped relaunch (--no-relaunch). Restart Hermes manually to pick up the update.')
}
