#!/usr/bin/env node
// Host-built helper. Windows uses its in-box .NET Framework compiler; no SDK download.
import { execFileSync } from 'node:child_process'
import { chmodSync, existsSync, mkdirSync, renameSync, rmSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { macosSysroot, xcrunClangArgv } from './macos-sysroot.mjs'

const script = fileURLToPath(import.meta.url)
const root = resolve(dirname(script), '..')

export function resolveWindowsFrameworkCompiler() {
  const framework = resolve(process.env.SystemRoot || 'C:\\Windows', 'Microsoft.NET')
  const compiler = ['Framework64', 'Framework']
    .map(dir => resolve(framework, dir, 'v4.0.30319', 'csc.exe'))
    .find(existsSync)
  if (!compiler) throw new Error('The Windows HUD helper needs the .NET Framework included with Windows.')
  return compiler
}

export function hudModifierBinaryRelativePath(platform = process.platform, arch = process.arch) {
  return `native/${platform}-${platform === 'darwin' ? 'universal' : arch}/hud-modifier-monitor${platform === 'win32' ? '.exe' : ''}`
}

export function buildHudModifierMonitor({
  distDir = resolve(root, 'dist'),
  platform = process.platform,
  arch = process.arch,
  sysroot
} = {}) {
  // Cross-packaging must not ship a host binary under the target's name. The
  // capability stays unavailable unless that target was built on its own host.
  if (platform !== process.platform || (platform === 'linux' && arch !== process.arch)) {
    if (platform === 'win32') throw new Error('Build Windows packages on Windows so the HUD helper is included.')
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
          ...xcrunClangArgv(sysroot === undefined ? macosSysroot() : sysroot),
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
    } else if (platform === 'win32') {
      execFileSync(resolveWindowsFrameworkCompiler(), [
        '/nologo', '/target:exe', '/platform:anycpu', '/optimize+', '/warnaserror+',
        '/reference:System.Windows.Forms.dll', `/out:${staging}`,
        source('hud-modifier-monitor-win.cs'), source('hud-modifier-gesture.cs')
      ], { stdio: 'pipe', timeout: 120_000 })
    } else {
      execFileSync(
        process.env.CC || 'cc',
        [
          '-std=gnu11',
          '-O2',
          '-Wall',
          '-Wextra',
          source('hud-modifier-monitor-x11.c'),
          '-o',
          staging,
          '-lX11', '-lXi'
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
    if (platform !== 'linux') throw error
    console.warn('[hud-modifier] unavailable: native build needs a C compiler, libx11-dev and libxi-dev; desktop packaging continues')
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
