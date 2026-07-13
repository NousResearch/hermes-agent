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
import {
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync
} from 'node:fs'
import { isMain } from './utils.mjs'

/**
 * Copy a native helper, re-asserting +x on the destination.
 *
 * cpSync preserves the source mode, and node-pty's published `spawn-helper`
 * can arrive from npm without its execute bit. node-pty posix_spawnp()s that
 * binary on darwin, so a stripped mode surfaces only at runtime, as an opaque
 * "posix_spawnp failed" the first time a terminal is opened. Re-assert the bit
 * rather than trusting whatever mode the package manager left behind.
 */
function copyExecutable(src, dest) {
  cpSync(src, dest)
  try {
    chmodSync(dest, 0o755)
  } catch {
    // Best-effort: a failed chmod is still better than an unhandled throw here.
  }
}

const here = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(here, '..')
const require = createRequire(import.meta.url)

/**
 * Locate node-pty's package root via real module resolution, so this
 * works whether it's hoisted to a workspace root or local to this app.
 */
function resolveNodePtyRoot() {
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
    if (extensions.some((ext) => entry.name.endsWith(ext))) {
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
    if (entry.name === 'spawn-helper') {
      copyExecutable(join(srcDir, entry.name), join(destDir, entry.name))
      continue
    }
    if (/\.(node|dll|exe)$/.test(entry.name)) {
      cpSync(join(srcDir, entry.name), join(destDir, entry.name))
    }
  }
}

export function stageNodePty({ platform = process.platform, arch = process.arch } = {}) {
  const srcRoot = resolveNodePtyRoot()
  const destRoot = resolve(projectRoot, 'dist/node_modules/node-pty')

  rmSync(destRoot, { recursive: true, force: true })
  mkdirSync(destRoot, { recursive: true })

  // package.json — needed so `require('node-pty')` resolves the package
  // (reads "main") rather than treating it as a directory with no entry.
  cpSync(join(srcRoot, 'package.json'), join(destRoot, 'package.json'))

  // lib/**/*.js — the JS surface node-pty's `main` points into.
  copyGlobByExt(join(srcRoot, 'lib'), join(destRoot, 'lib'), ['.js'])

  // build/Release/* — present when node-pty was compiled locally
  // (e.g. no prebuild available for this Electron ABI/platform combo).
  // Some installs won't have this at all if prebuild-install succeeded.
  copyBuildRelease(join(srcRoot, 'build/Release'), join(destRoot, 'build/Release'))

  // prebuilds/<platform>-<arch>/* — the prebuild-install payload for the
  // *target* we're packaging, not necessarily the host running this script.
  // Explicit extensions only, to skip the ~25MB of Windows .pdb symbols
  // prebuild-install bundles alongside the .node/.dll.
  const prebuildDir = join(srcRoot, 'prebuilds', `${platform}-${arch}`)
  if (existsSync(prebuildDir)) {
    const destPrebuild = join(destRoot, 'prebuilds', `${platform}-${arch}`)
    mkdirSync(destPrebuild, { recursive: true })
    for (const entry of readdirSync(prebuildDir, { withFileTypes: true })) {
      if (entry.name === 'conpty' && entry.isDirectory()) {
        cpSync(join(prebuildDir, 'conpty'), join(destPrebuild, 'conpty'), { recursive: true })
        continue
      }
      if (entry.isFile() && /\.(node|dll|exe)$/.test(entry.name)) {
        cpSync(join(prebuildDir, entry.name), join(destPrebuild, entry.name))
        continue
      }
      if (entry.name === 'spawn-helper') {
        copyExecutable(join(prebuildDir, entry.name), join(destPrebuild, entry.name))
      }
    }
  } else {
    console.warn(
      `[stage-native-deps] no prebuild found at prebuilds/${platform}-${arch} for node-pty. ` +
        `If build/Release/* above is also empty, this target will fail at runtime. ` +
        `Run "npx electron-rebuild -w node-pty" for this target, or check that ` +
        `node-pty's published prebuilds cover ${platform}-${arch}.`
    )
  }

  console.log(`[stage-native-deps] staged node-pty (${platform}-${arch}) -> ${destRoot}`)
  return destRoot
}

// Allow direct CLI invocation: node scripts/stage-native-deps.mjs [platform] [arch]
if (isMain(import.meta.url)) {
  const [platform, arch] = process.argv.slice(2)
  stageNodePty({ platform, arch })
}
