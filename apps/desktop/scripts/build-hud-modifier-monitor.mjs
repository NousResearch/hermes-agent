#!/usr/bin/env node
// Optional, host-native helper. No compiler or development headers at runtime.
import { execFileSync } from 'node:child_process'
import { chmodSync, mkdirSync, renameSync, rmSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const script = fileURLToPath(import.meta.url)
const root = resolve(dirname(script), '..')

export function hudModifierBinaryRelativePath(platform = process.platform, arch = process.arch) {
  return `native/${platform}-${platform === 'darwin' ? 'universal' : arch}/hud-modifier-monitor${platform === 'win32' ? '.exe' : ''}`
}

export function buildHudModifierMonitor({
  distDir = resolve(root, 'dist'),
  platform = process.platform,
  arch = process.arch
} = {}) {
  // Cross-packaging must not ship a host binary under the target's name. The
  // capability stays unavailable unless that target was built on its own host.
  if (platform !== process.platform || (platform !== 'darwin' && arch !== process.arch)) {
    console.warn(`[hud-modifier] ${platform}-${arch} needs a native build; modifier tap unavailable for this target`)
    return null
  }
  if (!['darwin', 'linux', 'win32'].includes(platform)) return null
  const output = resolve(distDir, hudModifierBinaryRelativePath(platform, arch))
  const staging = `${output}.${process.pid}.tmp${platform === 'win32' ? '.exe' : ''}`
  const source = name => resolve(root, 'electron/native', name)
  mkdirSync(dirname(output), { recursive: true })
  try {
    if (platform === 'darwin') {
      execFileSync(
        'xcrun',
        [
          '--sdk',
          'macosx',
          'clang',
          '-arch',
          'arm64',
          '-arch',
          'x86_64',
          '-mmacosx-version-min=11.0',
          '-fobjc-arc',
          '-fblocks',
          '-O2',
          '-Wall',
          '-Wextra',
          '-framework',
          'Cocoa',
          '-framework',
          'CoreGraphics',
          source('hud-modifier-monitor.m'),
          '-o',
          staging
        ],
        { stdio: 'pipe', timeout: 120_000 }
      )
    } else {
      const windows = platform === 'win32'
      execFileSync(
        process.env.CC || (windows ? 'clang' : 'cc'),
        [
          '-std=gnu11',
          '-O2',
          '-Wall',
          '-Wextra',
          source(windows ? 'hud-modifier-monitor-win.c' : 'hud-modifier-monitor-x11.c'),
          '-o',
          staging,
          ...(windows ? ['-luser32'] : ['-lX11', '-lXi'])
        ],
        { stdio: 'pipe', timeout: 120_000 }
      )
    }
    chmodSync(staging, 0o755)
    renameSync(staging, output)
    console.log(`built ${output}`)
    return output
  } catch (error) {
    rmSync(output, { force: true }) // Never keep a stale helper after a failed rebuild.
    if (platform === 'darwin') throw error // The existing macOS build already requires Xcode.
    const prerequisite = platform === 'linux' ? 'a C compiler, libx11-dev and libxi-dev' : 'Clang and the Windows SDK'
    console.warn(`[hud-modifier] unavailable: native build needs ${prerequisite}; desktop packaging continues`)
    console.warn(String(error.stderr || error.message))
    return null
  } finally {
    rmSync(staging, { force: true })
  }
}

if (process.argv[1] && resolve(process.argv[1]) === script) {
  const args = process.argv.slice(2)
  if (args.length && (args.length !== 2 || args[0] !== '--out-dir'))
    throw new Error('Usage: build-hud-modifier-monitor.mjs [--out-dir PATH]')
  buildHudModifierMonitor(args.length ? { distDir: resolve(args[1]) } : {})
}
