/**
 * before-pack.mjs — electron-builder beforePack hook.
 *
 * Two responsibilities:
 *
 * 1. Removes any stale unpacked app directory (`appOutDir`) before
 *    electron-builder stages the Electron binaries into it.
 *
 * WHY THIS EXISTS
 * ---------------
 * electron-builder's final packaging step copies the stock `electron`
 * binary into `release/<platform>-unpacked/` and then renames it to the
 * product name (`Hermes`). If a PREVIOUS `npm run pack` was interrupted
 * (Ctrl-C, OOM kill, crash, full disk) the unpacked directory is left in a
 * corrupted partial state: it keeps the already-renamed `LICENSE.electron.txt`
 * and the Chromium payload (.pak/.so/icudtl.dat/chrome-sandbox) but is MISSING
 * the `electron` binary itself.
 *
 * On the next run, electron-builder sees the destination directory already
 * populated, skips re-copying the binary it thinks is present, then tries to
 * rename a `electron` file that no longer exists. The build dies with:
 *
 *   ENOENT: no such file or directory, rename
 *   '.../release/linux-unpacked/electron' -> '.../release/linux-unpacked/Hermes'
 *
 * This is a hard failure with no obvious cause for the user — `hermes desktop`
 * just prints "Desktop GUI build failed" and the only fix is to manually
 * `rm -rf` the release directory, which a normal user has no way to know.
 *
 * The packaging step is not idempotent across an interrupted run, so we make
 * it idempotent ourselves: wipe the target unpacked directory up front so
 * electron-builder always stages into a clean tree. This is safe — the
 * directory is a pure build artifact that electron-builder fully recreates
 * on every pack; nothing else depends on its prior contents.
 *
 * Cross-platform: the same partial-state trap exists on macOS
 * (the mac-unpacked Hermes.app bundle) and Windows (win-unpacked), so we
 * clean whatever `appOutDir` electron-builder hands us regardless of platform.
 *
 * Best-effort: a cleanup failure must never mask the real build. We log and
 * resolve rather than throw — worst case electron-builder hits the original
 * ENOENT, which is no worse than not having this hook at all.
 *
 * 2. Copies the target's admitted native tree. Acquisition belongs to native
 * preparation before packaging, never this hook.
 *
 * electron-builder passes a context with:
 *   - appOutDir:            the unpacked app directory about to be staged
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - arch:                 Arch enum (0=ia32, 1=x64, 2=armv7l, 3=arm64, 4=universal)
 */
import { existsSync, rmSync, renameSync } from 'node:fs'
import path from 'node:path'
import { Arch } from 'electron-builder'
import { copyNativeInputs } from './prepared-native-deps.mjs'

/** @param {string | null | undefined} appOutDir @returns {boolean} */
export function cleanStaleAppOutDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') {
    return false
  }
  if (!existsSync(appOutDir)) {
    return false
  }
  // Recursive + force so a half-written tree (read-only bits, partial files)
  // can't block the wipe. retry/maxRetries rides out transient EBUSY on
  // Windows where an AV/indexer may briefly hold a handle.
  rmSync(appOutDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
  // Node's native rmSync silently deletes nothing on non-ASCII Windows
  // paths (nodejs/node#56049, fixed in v24.13.1) — without this check the
  // stale tree survives and the "removed" log below lies. Fall back to
  // the libuv-backed walk, which handles those paths on every version.
  if (existsSync(appOutDir)) {
    removeDirSync(appOutDir)
  }
  return true
}

/**
 * Keep manual recovery material for raw in-place Windows packs (#69179).
 * Preserve `<appOutDir>.bak` only when the output holds the product executable.
 * The CLI builds in a separate staging directory and rejects invalid output
 * before promotion. Its live-app transaction does not consume this backup.
 *
 * Returns true when the tree was preserved (appOutDir no longer exists), false
 * when there was nothing worth preserving (caller falls through to the wipe).
 * A rename failure (AV holding a handle) also returns false — the wipe is the
 * safe fallback and matches pre-#69179 behavior exactly.
 */
/** @param {string | null | undefined} appOutDir @param {string} [productExeName] @returns {boolean} */
export function preserveRollbackBackup(appOutDir, productExeName = 'Hermes.exe') {
  if (!appOutDir || typeof appOutDir !== 'string' || !existsSync(appOutDir)) {
    return false
  }
  if (!existsSync(path.join(appOutDir, productExeName))) {
    // Partial/corrupt tree (interrupted prior pack) — not rollback material.
    return false
  }
  const backupDir = `${appOutDir}.bak`
  try {
    rmSync(backupDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
    // Same non-ASCII rmSync no-op as cleanStaleAppOutDir: a surviving .bak
    // makes the rename fail and the previous build is wiped instead of kept.
    if (existsSync(backupDir)) {
      removeDirSync(backupDir)
    }
    renameSync(appOutDir, backupDir)
    return true
  } catch {
    return false
  }
}

/** @param {import("app-builder-lib").BeforePackContext} context @returns {Promise<void>} */
export default async function beforePack(context) {
  const appOutDir = context && context.appOutDir
  const platformName = context && context.electronPlatformName
  try {
    // Windows: keep the previous working build as rollback material for the
    // post-build integrity gate (#69179) instead of destroying it. Falls
    // through to the plain wipe when the old tree is partial/corrupt or the
    // rename fails.
    const productExe = `${(context && context.packager?.appInfo?.productFilename) || 'Hermes'}.exe`
    if (platformName === 'win32' && preserveRollbackBackup(appOutDir, productExe)) {
      console.log(`[before-pack] preserved previous unpacked dir for rollback: ${appOutDir}.bak`)
    } else if (cleanStaleAppOutDir(appOutDir)) {
      console.log(`[before-pack] removed stale unpacked dir before staging: ${appOutDir}`)
    }
  } catch (err) {
    // Never fail the build over cleanup; surface why so a genuinely stuck
    // directory (permissions, mount) is still diagnosable.
    console.warn(`[before-pack] could not clean ${appOutDir} (${err.message}); continuing`)
  }

  const platform = context && context.electronPlatformName
  const arch = context && typeof context.arch === 'number' ? Arch[context.arch] : undefined
  if (!platform || !arch) return
  const app = context.packager.projectDir
  const source = path.resolve(app, '../..')
  const nativeDeps = process.env.HERMES_PREPARED_NATIVE_DEPS || path.join(app, 'build/native-deps')
  copyNativeInputs({ source, nativeDeps, out: path.join(app, 'dist/node_modules'), platform, arch })
}