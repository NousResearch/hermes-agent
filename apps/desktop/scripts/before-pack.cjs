'use strict'

/**
 * before-pack.cjs — electron-builder beforePack hook.
 *
 * Moves any stale unpacked app directory (`appOutDir`) aside into a nested
 * backup under `release/.rebuild-backup/<dirname>` before electron-builder
 * stages the Electron binaries into it. The backup is cleaned up by
 * after-pack.cjs on success, and restored by the CLI build wrapper on failure.
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
 * Renaming preserves the last-known-good build so:
 *  - `hermes_cli/main.py` can restore the backup automatically on build
 *    failure — the desktop shortcut keeps working without user intervention.
 *  - `after-pack.cjs` removes the backup only after the new build completes.
 *
 * WHY NESTED UNDER `.rebuild-backup/`
 * -----------------------------------
 * Sibling renames (e.g. `win-unpacked.bak`) risk being matched by:
 *  - `_purge_electron_build_cache`'s `release/*-unpacked` glob
 *  - `_desktop_packaged_executable`'s macOS `mac*` glob, which would treat
 *    `mac-arm64.bak/Hermes.app/...` as a candidate executable and skip the
 *    corrupt-download retry that `_cmd_desktop` needs to perform.
 *
 * Nesting under `release/.rebuild-backup/` (a dot-directory) avoids both
 * globs: the `*-unpacked` glob won't enter the dot-directory, and the `mac*`
 * glob won't match a path starting with `.rebuild-backup`.
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

/** Subdirectory under ``release/`` where backups are parked. */
const REBUILD_BACKUP_DIRNAME = '.rebuild-backup'

/**
 * Derive the nested backup directory path from the app output directory.
 *
 * ``appOutDir`` is e.g. ``/build/release/win-unpacked``.
 * Returns ``/build/release/.rebuild-backup/win-unpacked``.
 *
 * @param {string} appOutDir
 * @returns {string}
 */
function staleBackupPath(appOutDir) {
  return path.join(
    path.dirname(appOutDir),
    REBUILD_BACKUP_DIRNAME,
    path.basename(appOutDir)
  )
}

/**
 * Move the stale unpacked directory aside into the nested backup.
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

  // Remove a leftover backup from a *previous* failed build so we can rename
  // the current directory cleanly.
  if (fs.existsSync(backupDir)) {
    try {
      fs.rmSync(backupDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 })
    } catch (_) {
      // If we can't even clean the old backup, fall through to rmSync below.
    }
  }

  // Ensure the parent .rebuild-backup/ directory exists.
  try {
    fs.mkdirSync(path.dirname(backupDir), { recursive: true })
  } catch (_) {
    // Best-effort; renameSync will fail with its own error if parent missing.
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

exports.REBUILD_BACKUP_DIRNAME = REBUILD_BACKUP_DIRNAME
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
