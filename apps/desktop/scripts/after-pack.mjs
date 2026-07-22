/**
 * after-pack.mjs — electron-builder afterPack hook.
 *
 * 1. Cleans up the nested backup directory under
 *    `release/.rebuild-backup/<dirname>` left by before-pack.mjs — the build
 *    succeeded so the backup is no longer needed. Also removes the empty
 *    `.rebuild-backup/` parent when possible.
 * 2. Stamps the Hermes icon + identity onto the packed Windows Hermes.exe via
 *    rcedit (delegated to set-exe-identity.mjs).
 *
 * WHY THE BACKUP CLEANUP IS HERE
 * ------------------------------
 * before-pack.mjs renames the old `appOutDir` into
 * `release/.rebuild-backup/<dirname>` to preserve a fallback exe when the
 * build fails. This hook runs only after electron-builder has staged a fresh,
 * complete tree — so it is now safe to discard the backup.
 *
 * Best-effort: a cleanup failure must never fail an otherwise-good build
 * (worst case is disk waste, not a broken app), so we log and resolve.
 *
 * Windows-only for the exe stamp: rcedit edits PE resources, irrelevant on
 * macOS/Linux where the app identity comes from the bundle Info.plist /
 * desktop entry. Best-effort: a stamp failure must never fail an
 * otherwise-good build (worst case is the stock icon, not a broken app),
 * so we log and resolve rather than throw.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */
import { existsSync, rmSync, rmdirSync } from 'node:fs'
import path from 'node:path'

import { staleBackupPath, REBUILD_BACKUP_DIRNAME } from './before-pack.mjs'
import { stampExeIdentity } from './set-exe-identity.mjs'

/**
 * Remove the nested backup directory left by before-pack.mjs.
 * Best-effort: never throws.
 *
 * @param {string} appOutDir
 */
export function cleanStaleBackupDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') {
    return
  }
  const backupDir = staleBackupPath(appOutDir)
  if (!existsSync(backupDir)) {
    return
  }
  try {
    rmSync(backupDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
    console.log(`[after-pack] removed backup: ${backupDir}`)
  } catch (err) {
    console.warn(
      `[after-pack] could not remove backup ${backupDir} (${err.message}); ` +
        `safe to delete manually`
    )
    return  // don't try to rmdir parent if child removal failed
  }

  // Remove the .rebuild-backup/ parent if it's now empty.
  const parentDir = path.dirname(backupDir)
  try {
    rmdirSync(parentDir)
  } catch (_) {
    // Directory not empty (other platform backups may still exist), or
    // permissions — ignore; the empty dir is harmless.
  }
}

export default async function afterPack(context) {
  // NOTE: backup cleanup is deferred to the CLI layer (hermes_cli/main.py).
  // after-pack runs inside electron-builder before the CLI regains control,
  // so cleaning the backup here would destroy the last-good build before
  // the CLI can verify that a fresh executable was actually produced
  // (pack may return 0 but leave no launchable app — misconfigured targets).
  // The CLI checks _desktop_packaged_executable() after the pack returns
  // and either restores the backup or discards it explicitly.

  // Windows-only: stamp the exe icon + identity.
  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); Hermes.exe keeps the stock Electron icon`)
  }
}
