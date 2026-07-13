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
  readFileSync,
  rmSync,
  statSync,
  writeFileSync
} from 'node:fs'
import { isMain } from './utils.mjs'

const here = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(here, '..')
const require = createRequire(import.meta.url)
const DARWIN_SHORT_HELPER_RELATIVE_PATH = '../../../bin/spawn-helper'

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
    if (entry.name === 'spawn-helper' || /\.(node|dll|exe)$/.test(entry.name)) {
      copyFileWithMode(join(srcDir, entry.name), join(destDir, entry.name))
    }
  }
}

function copyFileWithMode(srcPath, destPath) {
  cpSync(srcPath, destPath)
  chmodSync(destPath, statSync(srcPath).mode)
}

export function patchUnixTerminalForDarwinShortHelper(source) {
  const marker = "helperPath = helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked');"
  if (!source.includes(marker) || source.includes(DARWIN_SHORT_HELPER_RELATIVE_PATH)) {
    return source
  }

  const injected = `${marker}
if (process.platform === 'darwin') {
    const shortHelperPath = path.resolve(__dirname, '${DARWIN_SHORT_HELPER_RELATIVE_PATH}');
    try {
        if (fs.existsSync(shortHelperPath)) {
            helperPath = shortHelperPath;
        }
    } catch {
        // Ignore fallback lookup failures and keep node-pty's default helper path.
    }
}`
  return source.replace(marker, injected)
}

function patchStagedUnixTerminal(destRoot) {
  const unixTerminalPath = join(destRoot, 'lib', 'unixTerminal.js')
  if (!existsSync(unixTerminalPath)) {
    return false
  }

  const original = readFileSync(unixTerminalPath, 'utf8')
  const patched = patchUnixTerminalForDarwinShortHelper(original)
  if (patched === original) {
    return false
  }

  writeFileSync(unixTerminalPath, patched, 'utf8')
  return true
}

export function stageDarwinShortSpawnHelper(
  destRoot,
  { platform, arch, shortHelperPath = resolve(projectRoot, 'dist/bin/spawn-helper') } = {}
) {
  rmSync(shortHelperPath, { force: true })
  if (platform !== 'darwin') {
    return null
  }

  const candidates = [
    join(destRoot, 'prebuilds', `${platform}-${arch}`, 'spawn-helper'),
    join(destRoot, 'build', 'Release', 'spawn-helper')
  ]
  const sourcePath = candidates.find(existsSync)
  if (!sourcePath) {
    return null
  }

  mkdirSync(dirname(shortHelperPath), { recursive: true })
  copyFileWithMode(sourcePath, shortHelperPath)
  patchStagedUnixTerminal(destRoot)
  return shortHelperPath
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
        copyFileWithMode(join(prebuildDir, entry.name), join(destPrebuild, entry.name))
        continue
      }
      if (entry.name === 'spawn-helper') {
        copyFileWithMode(join(prebuildDir, entry.name), join(destPrebuild, entry.name))
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

  stageDarwinShortSpawnHelper(destRoot, { platform, arch })

  console.log(`[stage-native-deps] staged node-pty (${platform}-${arch}) -> ${destRoot}`)
  return destRoot
}

// Allow direct CLI invocation: node scripts/stage-native-deps.mjs [platform] [arch]
if (isMain(import.meta.url)) {
  const [platform, arch] = process.argv.slice(2)
  stageNodePty({ platform, arch })
}
