import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { fileDigest, treeDigest, preparationRequired } from './prepared-packaging.mjs'

// ─── libuv-safe fs primitives ────────────────────────────────────────
//
// Node's native (non-libuv) rewrite of fs.cpSync/fs.rmSync mishandles
// non-ASCII Windows paths (stage-native-deps.mjs has the full write-up: a
// recursive cpSync fails with EIO, an overwriting one with a bogus errno-0
// unlink error, and rmSync silently deletes nothing — leaving a half-staged
// tree). The installer builds on whatever Node the user already has, so the
// packaging copies stick to libuv-backed primitives too: beforePack wipes
// dist/native and re-copies the prepared helpers with these, because the
// silent no-op left electron-builder with no hud-modifier-monitor.exe to
// package (#127420).

/** Recursively delete a path without fs.rmSync (missing paths are fine). */
export function removeTreeSync(target) {
  let stats
  try {
    stats = fs.lstatSync(target)
  } catch {
    return
  }
  if (!stats.isDirectory()) {
    fs.unlinkSync(target)
    return
  }
  for (const entry of fs.readdirSync(target)) {
    removeTreeSync(path.join(target, entry))
  }
  fs.rmdirSync(target)
}

/** Recursively copy a tree without fs.cpSync, following symlinks (cpSync dereference).
 * *skip* receives each absolute source path and excludes it (with its subtree). */
export function copyTreeSync(srcDir, destDir, skip = () => false) {
  fs.mkdirSync(destDir, { recursive: true })
  for (const entry of fs.readdirSync(srcDir)) {
    const src = path.join(srcDir, entry)
    if (skip(src)) continue
    const dest = path.join(destDir, entry)
    if (fs.statSync(src).isDirectory()) {
      copyTreeSync(src, dest, skip)
    } else {
      fs.copyFileSync(src, dest)
    }
  }
}

/** Relative file paths under *root*, for post-copy verification. */
function relativeFiles(root) {
  const found = []
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const full = path.join(root, entry.name)
    if (entry.isDirectory()) {
      for (const relative of relativeFiles(full)) found.push(path.join(entry.name, relative))
    } else {
      found.push(entry.name)
    }
  }
  return found
}

/** @typedef {{ source: string, nativeDeps: string, platform?: string, arch?: string, nativeToolchain?: string }} NativeSelection */
/** @param {string} source @returns {string} */
function nativeIdentity(source) {
  return createHash('sha256').update(JSON.stringify([
    fileDigest(path.join(source, 'package-lock.json')),
    fileDigest(path.join(source, 'apps/desktop/package.json')),
    fileDigest(path.join(import.meta.dirname, 'stage-native-deps.mjs')),
    fileDigest(path.join(import.meta.dirname, 'prepared-native-deps.mjs')),
    ...['build-command-screenshot-monitor.mjs', 'build-hud-modifier-monitor.mjs']
      .map(name => fileDigest(path.join(import.meta.dirname, name))),
    ...['command-screenshot-monitor.m', 'hud-modifier-gesture.h', 'hud-modifier-gesture.cs',
      'hud-modifier-monitor.m', 'hud-modifier-monitor-win.cs', 'hud-modifier-monitor-x11.c']
      .map(name => fileDigest(path.join(source, 'apps/desktop/electron/native', name))),
  ])).digest('hex')
}

/**
 * The sidecar stays outside node_modules so it never ships in the application.
 * @param {{ source: string, out: string, platform: string, arch: string, nativeToolchain?: string }} inputs
 * @returns {void}
 */
export function recordNativeInputs({ source, out, platform, arch, nativeToolchain }) {
  fs.writeFileSync(`${out}.prepared.json`, JSON.stringify({
    schema: 1, source: fs.realpathSync(source), out: fs.realpathSync(out),
    platform, arch, nativeToolchain, identity: nativeIdentity(source), digest: treeDigest(out),
  }) + '\n')
}

/** @param {NativeSelection} inputs @returns {string} */
export function readNativeInputs({ source, nativeDeps, platform = process.platform, arch = process.arch, nativeToolchain }) {
  try {
    const record = JSON.parse(fs.readFileSync(`${nativeDeps}.prepared.json`, 'utf8'))
    if (record.schema !== 1 || record.source !== fs.realpathSync(source) || record.out !== fs.realpathSync(nativeDeps) ||
        record.platform !== platform || record.arch !== arch ||
        (nativeToolchain !== undefined && record.nativeToolchain !== nativeToolchain) ||
        record.identity !== nativeIdentity(source) || record.digest !== treeDigest(nativeDeps)) {
      throw preparationRequired('Stale or foreign native inputs')
    }
    if (!fs.statSync(path.join(nativeDeps, 'node-pty/package.json')).isFile()) throw preparationRequired('Missing prepared node-pty')
    return record.out
  } catch (error) {
    throw preparationRequired(`Cannot consume native inputs: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/** @param {NativeSelection & { out: string }} inputs @returns {void} */
export function copyNativeInputs({ out, ...inputs }) {
  copyNativeTree({ nativeDeps: readNativeInputs(inputs), out })
}

/** Copy admitted modules and executable resources without rebuilding either.
 * @param {{ nativeDeps: string, out: string }} inputs out is the product's node_modules.
 * @returns {void}
 */
export function copyNativeTree({ nativeDeps, out }) {
  nativeDeps = fs.realpathSync(nativeDeps)
  const destination = path.resolve(out)
  const helpers = path.join(path.dirname(destination), 'native')
  for (const target of [destination, helpers]) {
    if (target === nativeDeps || target.startsWith(nativeDeps + path.sep) || nativeDeps.startsWith(target + path.sep)) {
      throw preparationRequired('Native input and product directories overlap')
    }
  }
  removeTreeSync(destination)
  removeTreeSync(helpers)
  const preparedHelpers = path.join(nativeDeps, 'native')
  copyTreeSync(nativeDeps, destination, src => src === preparedHelpers)
  if (fs.existsSync(preparedHelpers)) {
    copyTreeSync(preparedHelpers, helpers)
    // A silent copy failure surfaced only later, as an inexplicable
    // ENOENT inside electron-builder (#127420): verify every prepared
    // helper actually landed before declaring the copy done.
    const missing = relativeFiles(preparedHelpers).filter(relative => !fs.existsSync(path.join(helpers, relative)))
    if (missing.length > 0) {
      throw preparationRequired(
        `Native helper copy is incomplete (missing ${missing.slice(0, 5).join(', ')}${missing.length > 5 ? ', …' : ''}); run preparation again`)
    }
  }
}
