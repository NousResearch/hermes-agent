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
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs'
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
 * Copies the locally-compiled build/Release output (used when no prebuild
 * was available and node-pty was built from source for the host machine).
 *
 * Filters by name/pattern rather than extension only: macOS builds a
 * separate `spawn-helper` executable (no file extension) that
 * lib/unixTerminal.js requires at a fixed relative path. Filtering this
 * directory by ['.node'] silently drops it — the package then looks
 * fine, ships fine, and crashes the first time a terminal is spawned.
 * Directories are copied wholesale to also cover any nested native
 * payload (e.g. a conpty/ subfolder some build layouts produce).
 */
function copyBuildRelease(srcDir, destDir) {
  if (!existsSync(srcDir)) return
  mkdirSync(destDir, { recursive: true })
  for (const entry of readdirSync(srcDir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      cpSync(join(srcDir, entry.name), join(destDir, entry.name), { recursive: true })
      continue
    }
    if (entry.name === 'spawn-helper' || /\.(node|dll|exe)$/.test(entry.name)) {
      cpSync(join(srcDir, entry.name), join(destDir, entry.name))
    }
  }
}

export function nodeRuntimeArch(arch) {
  return arch === 'armv7l' ? 'arm' : arch
}

export function isHostTarget(platform, arch, { hostPlatform = process.platform, hostArch = process.arch } = {}) {
  return platform === hostPlatform && nodeRuntimeArch(arch) === nodeRuntimeArch(hostArch)
}

function nativePayloadDir(root, platform, arch, { includeBuild = true } = {}) {
  // Match node-pty/lib/utils.js runtime precedence: local build first, then
  // prebuilds/<process.platform>-<process.arch>. Staging and validation must
  // select the same payload the packaged application will load.
  const candidates = []
  if (includeBuild) candidates.push(join(root, 'build', 'Release'))
  candidates.push(join(root, 'prebuilds', `${platform}-${nodeRuntimeArch(arch)}`))
  return candidates.find(candidate => {
    if (!isNonEmptyRegularFile(join(candidate, 'pty.node'))) return false
    return platform !== 'darwin' || isNonEmptyRegularFile(join(candidate, 'spawn-helper'))
  })
}

function isNonEmptyRegularFile(path) {
  try {
    const stats = statSync(path)
    return stats.isFile() && stats.size > 0
  } catch {
    return false
  }
}

export function findNodePtyNativePayload({
  platform = process.platform,
  arch = process.arch,
  root = resolveNodePtyRoot(),
  includeBuild = isHostTarget(platform, arch)
} = {}) {
  return nativePayloadDir(root, platform, arch, { includeBuild })
}

export function assertNodePtyNativePayload(root, { platform, arch }) {
  const payload = findNodePtyNativePayload({ root, platform, arch })
  if (payload) return payload

  const runtimeArch = nodeRuntimeArch(arch)
  const helperRequirement = platform === 'darwin' ? ' plus spawn-helper' : ''
  throw new Error(
    `node-pty has no usable native payload for ${platform}-${arch}: expected pty.node${helperRequirement} ` +
      `under prebuilds/${platform}-${runtimeArch}` +
      (isHostTarget(platform, arch) ? ' or build/Release' : '')
  )
}

export function stageNodePty({
  platform = process.platform,
  arch = process.arch,
  srcRoot = resolveNodePtyRoot(),
  destRoot = resolve(projectRoot, 'dist/node_modules/node-pty')
} = {}) {
  const sourcePayload = assertNodePtyNativePayload(srcRoot, { platform, arch })

  rmSync(destRoot, { recursive: true, force: true })
  mkdirSync(destRoot, { recursive: true })

  // package.json — needed so `require('node-pty')` resolves the package
  // (reads "main") rather than treating it as a directory with no entry.
  cpSync(join(srcRoot, 'package.json'), join(destRoot, 'package.json'))

  // lib/**/*.js — the JS surface node-pty's `main` points into.
  copyGlobByExt(join(srcRoot, 'lib'), join(destRoot, 'lib'), ['.js'])

  const buildDir = join(srcRoot, 'build', 'Release')
  if (sourcePayload === buildDir) {
    // Stage exactly the host build selected by node-pty's runtime precedence.
    copyBuildRelease(buildDir, join(destRoot, 'build', 'Release'))
  } else {
    // Otherwise stage only the selected target prebuild. Explicit extensions
    // skip the ~25MB of Windows .pdb symbols bundled alongside native files.
    const runtimeArch = nodeRuntimeArch(arch)
    const destPrebuild = join(destRoot, 'prebuilds', `${platform}-${runtimeArch}`)
    mkdirSync(destPrebuild, { recursive: true })
    for (const entry of readdirSync(sourcePayload, { withFileTypes: true })) {
      if (entry.name === 'conpty' && entry.isDirectory()) {
        cpSync(join(sourcePayload, 'conpty'), join(destPrebuild, 'conpty'), { recursive: true })
        continue
      }
      if (entry.isFile() && /\.(node|dll|exe)$/.test(entry.name)) {
        cpSync(join(sourcePayload, entry.name), join(destPrebuild, entry.name))
        continue
      }
      if (entry.name === 'spawn-helper') {
        cpSync(join(sourcePayload, entry.name), join(destPrebuild, entry.name))
      }
    }
  }

  const payload = assertNodePtyNativePayload(destRoot, { platform, arch })
  console.log(`[stage-native-deps] staged node-pty (${platform}-${arch}, ${payload}) -> ${destRoot}`)
  return destRoot
}

// Allow direct CLI invocation: node scripts/stage-native-deps.mjs [platform] [arch]
if (isMain(import.meta.url)) {
  const [platform, arch] = process.argv.slice(2)
  stageNodePty({ platform, arch })
}
