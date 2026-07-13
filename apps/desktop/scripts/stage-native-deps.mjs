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
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync
} from 'node:fs'
import { spawnSync } from 'node:child_process'
import { isMain } from './utils.mjs'

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
    if (entry.name === 'spawn-helper' || /\.(node|dll|exe)$/.test(entry.name)) {
      cpSync(join(srcDir, entry.name), join(destDir, entry.name))
    }
  }
}

// ─── binary classification ───────────────────────────────────────────
//
// .node files are shared libraries in the target platform's native binary
// format. By reading the first few bytes (magic) we can determine which
// platform a given .node was compiled for, without shelling out to `file`.
//
//   ELF  (\x7fELF)                         → linux
//   Mach-O 32-bit BE  (feedface)            → darwin
//   Mach-O 64-bit BE  (feedfacf)            → darwin
//   Mach-O 32-bit LE  (cefaedfe — CIGAM)    → darwin
//   Mach-O 64-bit LE  (cffaedfe — CIGAM_64) → darwin
//   Fat/Universal BE (cafebabe)             → darwin
//   Fat/Universal LE (bebafeca — FAT_CIGAM) → darwin
//   PE (MZ DOS header)                      → win32
//
// Mach-O and Fat binaries are stored on disk in the host's native byte
// order. On x64/arm64 Darwin (every Apple Silicon + every Intel Mac that
// ships node-pty prebuilds) that is little-endian, so the on-disk magic is
// the CIGAM byte-swapped form, NOT the big-endian MH_MAGIC form. Checking
// only the BE constants misclassifies every real Darwin prebuild as unknown.
//
// Exported for unit testing.

/**
 * Classify a native binary's target platform from its magic bytes.
 * Returns `'linux'`, `'darwin'`, `'win32'`, or `null` if unrecognized
 * or the file cannot be read.
 */
export function classifyNativeBinary(filePath) {
  let buf
  try {
    buf = readFileSync(filePath, { start: 0, end: 63 }) // first 64 bytes
  } catch {
    return null
  }
  if (buf.length < 4) return null

  // ELF: \x7f E L F
  if (buf[0] === 0x7f && buf[1] === 0x45 && buf[2] === 0x4c && buf[3] === 0x46) {
    return 'linux'
  }
  // Mach-O 32-bit (big-endian / MH_MAGIC): feedface
  if (buf[0] === 0xfe && buf[1] === 0xed && buf[2] === 0xfa && buf[3] === 0xce) {
    return 'darwin'
  }
  // Mach-O 64-bit (big-endian / MH_MAGIC_64): feedfacf
  if (buf[0] === 0xfe && buf[1] === 0xed && buf[2] === 0xfa && buf[3] === 0xcf) {
    return 'darwin'
  }
  // Mach-O 32-bit (little-endian / MH_CIGAM): cefaedfe
  if (buf[0] === 0xce && buf[1] === 0xfa && buf[2] === 0xed && buf[3] === 0xfe) {
    return 'darwin'
  }
  // Mach-O 64-bit (little-endian / MH_CIGAM_64): cffaedfe
  if (buf[0] === 0xcf && buf[1] === 0xfa && buf[2] === 0xed && buf[3] === 0xfe) {
    return 'darwin'
  }
  // Fat/Universal binary (big-endian / FAT_MAGIC): cafebabe
  if (buf[0] === 0xca && buf[1] === 0xfe && buf[2] === 0xba && buf[3] === 0xbe) {
    return 'darwin'
  }
  // Fat/Universal binary (little-endian / FAT_CIGAM): bebafeca
  if (buf[0] === 0xbe && buf[1] === 0xba && buf[2] === 0xfe && buf[3] === 0xca) {
    return 'darwin'
  }
  // PE: MZ DOS header
  if (buf[0] === 0x4d && buf[1] === 0x5a) {
    return 'win32'
  }
  return null
}

/**
 * Scan the staged destination tree for .node files and verify each one's
 * binary platform matches the requested target. Throws on any mismatch.
 *
 * This is the fail-closed safety net: even if a prebuild or build/Release
 * somehow slipped through with the wrong platform, this catches it before
 * the package ships a broken native binary to users.
 */
function validateStagedBinaries(destRoot, targetPlatform) {
  const mismatches = []
  function scan(dir, relPrefix) {
    if (!existsSync(dir)) return
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        scan(join(dir, entry.name), `${relPrefix}${entry.name}/`)
        continue
      }
      if (!entry.name.endsWith('.node')) continue
      const fullPath = join(dir, entry.name)
      const classified = classifyNativeBinary(fullPath)
      if (classified !== targetPlatform) {
        mismatches.push({ file: `${relPrefix}${entry.name}`, classified, expected: targetPlatform })
      }
    }
  }
  scan(join(destRoot, 'prebuilds'), 'prebuilds/')
  scan(join(destRoot, 'build', 'Release'), 'build/Release/')
  if (mismatches.length > 0) {
    throw new Error(
      `[stage-native-deps] native binary platform mismatch (target=${targetPlatform}):\n` +
        mismatches
          .map((m) => `  ${m.file}: expected ${m.expected}, got ${m.classified ?? 'unknown'}`)
          .join('\n') +
        `\nRefusing to stage a binary compiled for the wrong platform.`
    )
  }
}

/**
 * Stage node-pty's native runtime dependencies into `destRoot`.
 *
 * Exported separately from `stageNodePty` so tests can supply a fake
 * node-pty source tree without going through real module resolution.
 *
 * Strategy (fail-closed):
 *
 * 1. Copy the matching prebuild (`prebuilds/<platform>-<arch>/`) if present.
 * 2. Copy `build/Release/` **only when the target matches the host** —
 *    build/Release contains a binary compiled for the host's platform/arch,
 *    so staging it for a different target ships a broken app.
 * 3. If no native binary was staged:
 *    - Same platform as host, different arch → run `electron-rebuild --arch`.
 *    - Different platform from host → throw (cannot cross-compile native
 *      modules; build on the target platform or provide a prebuild).
 * 4. Validate every staged `.node` file's binary platform matches the target.
 */
export function stageNodePtyInto(srcRoot, destRoot, { platform = process.platform, arch = process.arch } = {}) {
  const hostMatch = platform === process.platform && arch === process.arch

  rmSync(destRoot, { recursive: true, force: true })
  mkdirSync(destRoot, { recursive: true })

  // package.json — needed so `require('node-pty')` resolves the package
  // (reads "main") rather than treating it as a directory with no entry.
  cpSync(join(srcRoot, 'package.json'), join(destRoot, 'package.json'))

  // lib/**/*.js — the JS surface node-pty's `main` points into.
  copyGlobByExt(join(srcRoot, 'lib'), join(destRoot, 'lib'), ['.js'])

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
        cpSync(join(prebuildDir, entry.name), join(destPrebuild, entry.name))
      }
    }
  }

  // build/Release/* — present when node-pty was compiled locally
  // (e.g. no prebuild available for this Electron ABI/platform combo).
  // Only stage this when the target matches the host, because
  // build/Release contains a binary compiled for the *host's* platform
  // and architecture. Staging a host binary for a different target (e.g.
  // a macOS Mach-O .node staged for a linux-arm64 target) ships a broken
  // app that crashes the first time a terminal is spawned.
  if (hostMatch) {
    const buildReleaseDir = join(srcRoot, 'build/Release')
    copyBuildRelease(buildReleaseDir, join(destRoot, 'build/Release'))
  }

  // Check whether a native binary for this target was staged.
  const stagedDirs = [
    join(destRoot, 'prebuilds', `${platform}-${arch}`),
    join(destRoot, 'build/Release')
  ]
  const hasNativeBinary = stagedDirs.some((dir) => {
    if (!existsSync(dir)) return false
    return readdirSync(dir, { recursive: true }).some((name) => String(name).endsWith('.node'))
  })

  if (!hasNativeBinary) {
    if (platform !== process.platform) {
      throw new Error(
        `[stage-native-deps] no prebuilt binary for ${platform}-${arch} and ` +
          `cannot cross-compile native modules from ${process.platform}-${process.arch}. ` +
          `Build on the target platform or provide a prebuild.`
      )
    }
    // Same platform, possibly different arch — rebuild from source with
    // the target architecture so electron-rebuild produces the correct
    // binary rather than defaulting to the host's arch.
    console.log(
      `[stage-native-deps] no native binary for ${platform}-${arch}; ` +
        `running electron-rebuild (target arch: ${arch})...`
    )
    const rebuildArgs = [
      '../../node_modules/.bin/electron-rebuild',
      '-f',
      '-w',
      'node-pty',
      '--arch',
      arch
    ]
    const result = spawnSync(process.execPath, rebuildArgs, {
      cwd: projectRoot,
      stdio: 'inherit'
    })
    if (result.status !== 0) {
      throw new Error(
        `electron-rebuild failed for ${platform}-${arch} (exit ${result.status}). ` +
          `Cannot stage node-pty without a native binary.`
      )
    }
    // Re-copy build/Release after electron-rebuild populated it.
    const buildReleaseDir = join(srcRoot, 'build/Release')
    copyBuildRelease(buildReleaseDir, join(destRoot, 'build/Release'))
  }

  // Validate every staged .node binary matches the target platform.
  validateStagedBinaries(destRoot, platform)

  console.log(`[stage-native-deps] staged node-pty (${platform}-${arch}) -> ${destRoot}`)
  return destRoot
}

export function stageNodePty({ platform = process.platform, arch = process.arch } = {}) {
  const srcRoot = resolveNodePtyRoot()
  const destRoot = resolve(projectRoot, 'dist/node_modules/node-pty')
  return stageNodePtyInto(srcRoot, destRoot, { platform, arch })
}

// Allow direct CLI invocation: node scripts/stage-native-deps.mjs [platform] [arch]
if (isMain(import.meta.url)) {
  const [platform, arch] = process.argv.slice(2)
  stageNodePty({ platform, arch })
}
