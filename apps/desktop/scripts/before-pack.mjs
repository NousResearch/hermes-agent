/**
 * before-pack.mjs — electron-builder beforePack hook.
 *
 * Two responsibilities:
 *
 * 1. Moves any stale unpacked app directory (`appOutDir`) aside into a nested
 *    backup under `release/.rebuild-backup/<dirname>` before electron-builder
 *    stages the Electron binaries into it. The backup is cleaned up by
 *    after-pack.mjs on success, and restored by the CLI build wrapper on
 *    failure.
 *
 * WHY RENAME INSTEAD OF DELETE
 * ----------------------------
 * Previously this hook `rmSync`'d the directory (see git history). That left
 * zero recovery path when the build after the cleanup failed — git merge
 * conflict, npm error, OOM, Ctrl-C, etc. The user's Hermes.exe was already
 * locked by electron-builder, the unpacked tree was gone, and the new one
 * never materialised. Result: Hermes.exe vanished until a successful rebuild.
 *
 * Renaming preserves the last-known-good build so:
 *  - `hermes_cli/main.py` can restore the backup automatically on build
 *    failure — the desktop shortcut keeps working without user intervention.
 *  - `after-pack.mjs` removes the backup only after the new build completes.
 *
 * WHY NESTED UNDER `.rebuild-backup/`
 * -----------------------------------
 * Sibling renames (e.g. `win-unpacked.bak`) risk being matched by:
 *  - `_purge_electron_build_cache`'s `release/*-unpacked` glob
 *  - `_desktop_packaged_executable`'s macOS `mac*` glob
 *
 * Nesting under `release/.rebuild-backup/` (a dot-directory) avoids both.
 *
 * RETRY-SAFE BACKUP PRESERVATION
 * ------------------------------
 * When a backup already exists AND appOutDir exists, we preserve the
 * existing backup and delete only the current (possibly partial) output.
 * This fixes the race where `cmd_gui` retries pack after failure.
 *
 * 2. Re-stages node-pty's native files for the ACTUAL target platform/arch.
 */
import { existsSync, mkdirSync, renameSync, rmSync } from 'node:fs'
import path from 'node:path'
import { Arch } from 'electron-builder'
import { stageNodePty } from './stage-native-deps.mjs'

export const REBUILD_BACKUP_DIRNAME = '.rebuild-backup'

export function staleBackupPath(appOutDir) {
  return path.join(path.dirname(appOutDir), REBUILD_BACKUP_DIRNAME, path.basename(appOutDir))
}

export function cleanStaleAppOutDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') {
    return { removed: false, backedUp: false }
  }
  if (!existsSync(appOutDir)) {
    return { removed: false, backedUp: false }
  }

  const backupDir = staleBackupPath(appOutDir)

  // RETRY-SAFE: preserve existing backup, only delete current partial output.
  if (existsSync(backupDir)) {
    try {
      rmSync(appOutDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
      return { removed: true, backedUp: true }
    } catch (rmErr) {
      console.warn(
        `[before-pack] could not clean partial output ${appOutDir} (${rmErr.message}); ` +
          `continuing — existing backup preserved`
      )
      return { removed: false, backedUp: true }
    }
  }

  // Ensure parent .rebuild-backup/ exists.
  try { mkdirSync(path.dirname(backupDir), { recursive: true }) } catch (_) {}

  // Try rename first (non-destructive).
  try {
    renameSync(appOutDir, backupDir)
    return { removed: true, backedUp: true }
  } catch (renameErr) {
    console.warn(
      `[before-pack] rename to backup failed (${renameErr.message}); ` +
        `falling back to rmSync — no backup will be available`
    )
  }

  // Fallback: delete so electron-builder can proceed.
  try {
    rmSync(appOutDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
    return { removed: true, backedUp: false }
  } catch (rmErr) {
    console.warn(`[before-pack] could not clean ${appOutDir} (${rmErr.message}); continuing`)
    return { removed: false, backedUp: false }
  }
}

export default async function beforePack(context) {
  const appOutDir = context && context.appOutDir
  try {
    const { removed } = cleanStaleAppOutDir(appOutDir)
    if (removed) {
      console.log(`[before-pack] moved stale unpacked dir aside before staging: ${appOutDir}`)
    }
  } catch (err) {
    console.warn(`[before-pack] error cleaning ${appOutDir} (${err.message}); continuing`)
  }

  try {
    const platform = context && context.electronPlatformName
    const archName = context && typeof context.arch === 'number' ? Arch[context.arch] : undefined
    if (platform && archName) {
      if (archName === 'universal') {
        console.warn(
          '[before-pack] target arch is "universal" — node-pty has no universal prebuild; ' +
            'staged binary will be whichever single-arch copy npm run build left behind.'
        )
      } else {
        await stageNodePty({ platform, arch: archName })
        console.log(`[before-pack] re-staged node-pty for target ${platform}-${archName}`)
      }
    }
  } catch (err) {
    throw new Error(`[before-pack] failed to stage node-pty for this target: ${err.message}`)
  }
}
