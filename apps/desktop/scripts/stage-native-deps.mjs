#!/usr/bin/env node
// stage-native-deps.mjs — stages node-pty's native runtime dependencies
//
// Usage:
//   node scripts/stage-native-deps.mjs                # host platform/arch
//   node scripts/stage-native-deps.mjs win32 arm64     # explicit target
//
// Also exported as `stageNodePty({ platform, arch })` for use from
// before-pack.mjs, where electron-builder gives you the real per-target
// platform/arch during multi-arch builds.

import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, resolve, join } from 'node:path'
import { chmodSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs'
import { isMain } from './utils.mjs'

const here = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(here, '..')
const require = createRequire(import.meta.url)

/**
 * Locate node-pty's package root via real module resolution, so this
 * works whether it's hoisted to a workspace root or local to this app.
 */
export function resolveNodePtyRoot() {
  const pkgJsonPath = require.resolve('node-pty/package.json', {
    paths: [projectRoot]
  })
  return dirname(pkgJsonPath)
}

function copyGlobByExt(srcDir, destDir, extensions) {
  if (!existsSync(srcDir)) return
  mkdirSync(destDir, { recursive: true })
  for (const entry of readdirSync(srcDir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      copyGlobByExt(join(srcDir, entry.name), join(destDir, entry.name), extensions)
      continue
    }
    if (extensions.some(ext => entry.name.endsWith(ext))) {
      mkdirSync(destDir, { recursive: true })
      cpSync(join(srcDir, entry.name), join(destDir, entry.name))
    }
  }
}

/**
 * Recursively copy a directory through per-file cpSync calls. Node 22's
 * recursive cpSync path can hard-crash on Windows while copying conpty.
 */
export function copyDirRecursive(srcDir, destDir) {
  mkdirSync(destDir, { recursive: true })
  for (const entry of readdirSync(srcDir, { withFileTypes: true })) {
    const src = join(srcDir, entry.name)
    const dest = join(destDir, entry.name)
    if (entry.isDirectory()) {
      copyDirRecursive(src, dest)
    } else {
      cpSync(src, dest)
    }
  }
}

function copyNativeFile(src, dest, { executable = false } = {}) {
  cpSync(src, dest)
  if (executable) chmodSync(dest, 0o755)
}

/** Copy a locally compiled Release or Debug payload. */
function copyBuildPayload(srcDir, destDir) {
  if (!existsSync(srcDir)) return
  mkdirSync(destDir, { recursive: true })
  for (const entry of readdirSync(srcDir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      copyDirRecursive(join(srcDir, entry.name), join(destDir, entry.name))
      continue
    }
    if (entry.name === 'spawn-helper' || /\.(node|dll|exe)$/.test(entry.name)) {
      copyNativeFile(join(srcDir, entry.name), join(destDir, entry.name), {
        executable: entry.name === 'spawn-helper'
      })
    }
  }
}

function copyPrebuildPayload(srcDir, destDir) {
  mkdirSync(destDir, { recursive: true })
  for (const entry of readdirSync(srcDir, { withFileTypes: true })) {
    if (entry.name === 'conpty' && entry.isDirectory()) {
      copyDirRecursive(join(srcDir, entry.name), join(destDir, entry.name))
      continue
    }
    if (entry.isFile() && /\.(node|dll|exe)$/.test(entry.name)) {
      copyNativeFile(join(srcDir, entry.name), join(destDir, entry.name))
      continue
    }
    if (entry.name === 'spawn-helper') {
      copyNativeFile(join(srcDir, entry.name), join(destDir, entry.name), { executable: true })
    }
  }
}

export function nodeRuntimeArch(arch) {
  return arch === 'armv7l' ? 'arm' : arch
}

export function isHostTarget(platform, arch, { hostPlatform = process.platform, hostArch = process.arch } = {}) {
  return platform === hostPlatform && nodeRuntimeArch(arch) === nodeRuntimeArch(hostArch)
}

function nativePayloadDirs(root, platform, arch, { includeBuild = true } = {}) {
  // Match node-pty/lib/utils.js. Its loader catches require failures and tries
  // the next directory, so all usable fallbacks must remain in the package.
  const candidates = []
  if (includeBuild) {
    candidates.push(join(root, 'build', 'Release'))
    candidates.push(join(root, 'build', 'Debug'))
  }
  candidates.push(join(root, 'prebuilds', `${platform}-${nodeRuntimeArch(arch)}`))
  return candidates
}

function isNonEmptyRegularFile(path) {
  try {
    const stats = statSync(path)
    return stats.isFile() && stats.size > 0
  } catch {
    return false
  }
}

function isExecutableRegularFile(path) {
  try {
    const stats = statSync(path)
    return stats.isFile() && stats.size > 0 && (stats.mode & 0o111) !== 0
  } catch {
    return false
  }
}

function expectedCpu(platform, arch) {
  const runtimeArch = nodeRuntimeArch(arch)
  const values = {
    linux: { ia32: 3, x64: 62, arm: 40, arm64: 183 },
    win32: { ia32: 0x14c, x64: 0x8664, arm: 0x1c4, arm64: 0xaa64 },
    darwin: { ia32: 7, x64: 0x01000007, arm: 12, arm64: 0x0100000c }
  }
  return values[platform]?.[runtimeArch]
}

function darwinCpuTypes(data) {
  if (data.length < 8) return []
  const magic = data.readUInt32BE(0)
  const thin = new Map([
    [0xfeedface, 'BE'],
    [0xcefaedfe, 'LE'],
    [0xfeedfacf, 'BE'],
    [0xcffaedfe, 'LE']
  ])
  if (thin.has(magic)) {
    return [thin.get(magic) === 'LE' ? data.readUInt32LE(4) : data.readUInt32BE(4)]
  }

  const fat = new Map([
    [0xcafebabe, { endian: 'BE', width: 20 }],
    [0xbebafeca, { endian: 'LE', width: 20 }],
    [0xcafebabf, { endian: 'BE', width: 32 }],
    [0xbfbafeca, { endian: 'LE', width: 32 }]
  ])
  const layout = fat.get(magic)
  if (!layout) return []
  const read32 = layout.endian === 'LE' ? data.readUInt32LE.bind(data) : data.readUInt32BE.bind(data)
  const count = read32(4)
  const cpus = []
  for (let index = 0; index < count; index++) {
    const offset = 8 + index * layout.width
    if (offset + 4 > data.length) return []
    cpus.push(read32(offset))
  }
  return cpus
}

export function nativeBinaryMatchesTarget(path, { platform, arch }) {
  if (!isNonEmptyRegularFile(path)) return false
  const cpu = expectedCpu(platform, arch)
  if (cpu === undefined) return false

  try {
    const data = readFileSync(path)
    if (platform === 'linux') {
      if (data.length < 20 || !data.subarray(0, 4).equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46]))) {
        return false
      }
      const machine = data[5] === 1 ? data.readUInt16LE(18) : data.readUInt16BE(18)
      const expectedClass = ['x64', 'arm64'].includes(nodeRuntimeArch(arch)) ? 2 : 1
      return data[4] === expectedClass && machine === cpu
    }
    if (platform === 'win32') {
      if (data.length < 64 || data[0] !== 0x4d || data[1] !== 0x5a) return false
      const peOffset = data.readUInt32LE(0x3c)
      if (peOffset + 6 > data.length || data.toString('ascii', peOffset, peOffset + 4) !== 'PE\0\0') {
        return false
      }
      return data.readUInt16LE(peOffset + 4) === cpu
    }
    if (platform === 'darwin') return darwinCpuTypes(data).includes(cpu)
  } catch {
    return false
  }
  return false
}

function payloadIsUsable(payloadDir, { platform, arch, requireExecutableHelper = false }) {
  if (!nativeBinaryMatchesTarget(join(payloadDir, 'pty.node'), { platform, arch })) return false
  if (platform !== 'darwin') return true
  const helper = join(payloadDir, 'spawn-helper')
  return requireExecutableHelper ? isExecutableRegularFile(helper) : isNonEmptyRegularFile(helper)
}

export function findNodePtyNativePayloads({
  platform = process.platform,
  arch = process.arch,
  root = resolveNodePtyRoot(),
  includeBuild = isHostTarget(platform, arch),
  requireExecutableHelper = false
} = {}) {
  return nativePayloadDirs(root, platform, arch, { includeBuild }).filter(payloadDir =>
    payloadIsUsable(payloadDir, { platform, arch, requireExecutableHelper })
  )
}

export function findNodePtyNativePayload(options = {}) {
  return findNodePtyNativePayloads(options)[0]
}

export function assertNodePtyNativePayload(root, { platform, arch, requireExecutableHelper = false }) {
  const payloads = findNodePtyNativePayloads({
    root,
    platform,
    arch,
    requireExecutableHelper
  })
  if (payloads.length > 0) return payloads

  const runtimeArch = nodeRuntimeArch(arch)
  const helperRequirement = platform === 'darwin' ? ' plus executable spawn-helper' : ''
  throw new Error(
    `node-pty has no usable native payload for ${platform}-${arch}: expected a compatible pty.node${helperRequirement} ` +
      `under prebuilds/${platform}-${runtimeArch}` +
      (isHostTarget(platform, arch) ? ' or build/Release or build/Debug' : '')
  )
}

export function stageNodePty({
  platform = process.platform,
  arch = process.arch,
  srcRoot = resolveNodePtyRoot(),
  destRoot = resolve(projectRoot, 'dist/node_modules/node-pty')
} = {}) {
  const sourcePayloads = assertNodePtyNativePayload(srcRoot, { platform, arch })

  rmSync(destRoot, { recursive: true, force: true })
  mkdirSync(destRoot, { recursive: true })

  // JS/package files are required before a bare require('node-pty') can reach
  // any native candidate.
  cpSync(join(srcRoot, 'package.json'), join(destRoot, 'package.json'))
  copyGlobByExt(join(srcRoot, 'lib'), join(destRoot, 'lib'), ['.js'])

  const runtimeArch = nodeRuntimeArch(arch)
  const prebuildDir = join(srcRoot, 'prebuilds', `${platform}-${runtimeArch}`)
  for (const sourcePayload of sourcePayloads) {
    if (sourcePayload === prebuildDir) {
      copyPrebuildPayload(sourcePayload, join(destRoot, 'prebuilds', `${platform}-${runtimeArch}`))
      continue
    }
    const configuration = sourcePayload.endsWith(join('build', 'Debug')) ? 'Debug' : 'Release'
    copyBuildPayload(sourcePayload, join(destRoot, 'build', configuration))
  }

  const payloads = assertNodePtyNativePayload(destRoot, {
    platform,
    arch,
    requireExecutableHelper: true
  })
  console.log(`[stage-native-deps] staged node-pty (${platform}-${arch}, ${payloads.join(', ')}) -> ${destRoot}`)
  return destRoot
}

// Allow direct CLI invocation: node scripts/stage-native-deps.mjs [platform] [arch]
if (isMain(import.meta.url)) {
  const [platform, arch] = process.argv.slice(2)
  stageNodePty({ platform, arch })
}
