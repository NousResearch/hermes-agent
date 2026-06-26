'use strict'

/**
 * before-pack.cjs — electron-builder beforePack hook.
 *
 * Moves any stale unpacked app directory (`appOutDir`) aside into a `.bak`
 * backup before electron-builder stages the Electron binaries into it. The
 * backup is cleaned up by after-pack.cjs on success.
 *
 * WHY RENAME INSTEAD OF DELETE
 * ----------------------------
 * Previously this hook `fs.rmSync`'d the directory (see git history). That
 * left zero recovery path when the build after the cleanup failed — git merge
 * conflict, npm error, OOM, Ctrl-C, etc. The user's running Hermes.exe was
 * already shut down (electron-builder needs the file unlocked), the unpacked
 * tree was gone, and the new one never materialised. Result: Hermes.exe
 * vanished until a successful rebuild.
 *
 * Renaming to `<appOutDir>.bak` preserves the last-known-good build so:
 *  - A build failure leaves the `.bak` intact — the user can keep working
 *    with `Hermes.exe.bak` or rename it back manually.
 *  - The `hermes desktop` / `--update` path can detect the `.bak` as a
 *    fallback exe and recover automatically (future follow-up).
 *  - The `after-pack.cjs` hook removes the `.bak` only after the new build
 *    completes, which is safe because `appOutDir` now holds a fresh, complete
 *    Electron tree (stamped + identity-applied).
 *
 * Cross-platform: rename(2) on the same filesystem is atomic on Linux/macOS
 * and near-atomic on NTFS. All platforms benefit; this path runs for every
 * `beforePack` regardless of `electronPlatformName`.
 *
 * Best-effort: if rename fails (cross-device mount, permissions, EBUSY) we
 * fall back to the old `rmSync` so the build can still proceed — but we log
 * a warning so the user knows they have no backup.
 *
 * electron-builder passes a context with:
 *   - appOutDir:            the unpacked app directory about to be staged
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 */

const fs = require('node:fs')
const path = require('node:path')

/**
 * Derive the backup directory path from the app output directory.
 * @param {string} appOutDir
 * @returns {string}
 */
function staleBackupPath(appOutDir) {
  return `${appOutDir}.bak`
}

/**
 * Move the stale unpacked directory aside into a `.bak` backup.
 * Falls back to rmSync if rename is impossible (cross-device, permissions).
 *
 * @param {string} appOutDir
 * @returns {{ removed: boolean, backedUp: boolean }}
 */
function cleanStaleAppOutDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') {
    return { removed: false, backedUp: false }
  }
  if (!fs.existsSync(appOutDir)) {
    return { removed: false, backedUp: false }
  }

  const backupDir = staleBackupPath(appOutDir)

  // Remove a leftover .bak from a *previous* failed build so we can rename
  // the current directory cleanly.
  if (fs.existsSync(backupDir)) {
    try {
      fs.rmSync(backupDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 })
    } catch (_) {
      // If we can't even clean the old backup, fall through to rmSync below.
    }
  }

  // Try rename first (non-destructive, preserves the last good build).
  try {
    fs.renameSync(appOutDir, backupDir)
    return { removed: true, backedUp: true }
  } catch (renameErr) {
    console.warn(
      `[before-pack] rename to backup failed (${renameErr.message}); ` +
        `falling back to rmSync — no backup will be available`
    )
  }

  // Fallback: delete outright so electron-builder can proceed.
  try {
    fs.rmSync(appOutDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
    return { removed: true, backedUp: false }
  } catch (rmErr) {
    console.warn(`[before-pack] could not clean ${appOutDir} (${rmErr.message}); continuing`)
    return { removed: false, backedUp: false }
  }
}

exports.staleBackupPath = staleBackupPath
exports.cleanStaleAppOutDir = cleanStaleAppOutDir

exports.default = async function beforePack(context) {
  const appOutDir = context && context.appOutDir
  try {
    const { removed } = cleanStaleAppOutDir(appOutDir)
    if (removed) {
      console.log(`[before-pack] moved stale unpacked dir aside before staging: ${appOutDir}`)
    }
  } catch (err) {
    // Never fail the build over cleanup; surface why so a genuinely stuck
    // directory (permissions, mount) is still diagnosable.
    console.warn(`[before-pack] error cleaning ${appOutDir} (${err.message}); continuing`)
  }
}
