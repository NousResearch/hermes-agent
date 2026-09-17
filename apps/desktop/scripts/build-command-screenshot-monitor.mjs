#!/usr/bin/env node
// Build-time only: the shipped app never needs clang or Xcode tools.
import { execFileSync } from 'node:child_process'
import { chmodSync, mkdirSync, renameSync, rmSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const script = fileURLToPath(import.meta.url)
const root = resolve(dirname(script), '..')

export function buildCommandScreenshotMonitor({ distDir = resolve(root, 'dist') } = {}) {
  if (process.platform !== 'darwin') return null
  const output = resolve(distDir, 'native/command-screenshot-monitor')
  const staging = `${output}.${process.pid}.tmp`
  mkdirSync(dirname(output), { recursive: true })
  try {
    // Pin the SDK explicitly: a bare `xcrun clang` inherits the host default
    // SDK, which may be newer than the active linker (unknown-arch .tbd stubs
    // at link time, #113708). `--sdk macosx` names the same default SDK while
    // forcing the driver and linker to agree on it.
    execFileSync('xcrun', [
      '--sdk', 'macosx',
      'clang', '-arch', 'arm64', '-arch', 'x86_64', '-mmacosx-version-min=11.0',
      '-fobjc-arc', '-fblocks', '-O2', '-Wall', '-Wextra',
      '-framework', 'Cocoa', '-framework', 'CoreGraphics',
      resolve(root, 'electron/native/command-screenshot-monitor.m'), '-o', staging,
    ], { stdio: 'inherit', timeout: 120_000 })
    chmodSync(staging, 0o755)
    renameSync(staging, output)
    // electron-builder signs this Mach-O with the app; dist/** is already asarUnpack.
    console.log(`built ${output} (arm64 + x86_64)`)
    return output
  } finally {
    rmSync(staging, { force: true })
  }
}

if (process.argv[1] && resolve(process.argv[1]) === script) {
  const args = process.argv.slice(2)
  if (args.length && (args.length !== 2 || args[0] !== '--out-dir')) {
    throw new Error('Usage: build-command-screenshot-monitor.mjs [--out-dir PATH]')
  }
  buildCommandScreenshotMonitor(args.length ? { distDir: resolve(args[1]) } : {})
}
