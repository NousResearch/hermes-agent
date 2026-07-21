/**
 * before-pack.mjs — electron-builder beforePack hook.
 *
 * Two responsibilities:
 *
 * 1. Moves any stale unpacked app directory (`appOutDir`) aside into a nested
 *    backup under `release/.rebuild-backup/<dirname>` before electron-builder
 *    stages the Electron binaries into it. The backup is cleaned up by
 *    after-pack.mjs on success, and restored by the CLI build wrapper on failure.
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
 *  - `after-pack.mjs` removes the backup only after the new build completes.
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
 * 2. Re-stages node-pty's native files for the ACTUAL target platform/arch
 *    of this pack. `npm run build` already staged node-pty once for the
 *    host machine (see scripts/stage-native-deps.mjs), which is correct for
 *    single-arch builds matching the host. But electron-builder can target
 *    a different arch than the host (cross-build), or pack multiple archs
 *    from one `npm run build` (e.g. `dist:mac` => x64 + arm64). Only this
 *    hook knows the real per-target arch, via `context.arch` /
 *    `context.electronPlatformName` — so it re-stages on top of whatever
 *    `npm run build` left behind, per target, right before files are read
 *    for packing.
 *
 * electron-builder passes a context with:
 *   - appOutDir:            the unpacked app directory about to be staged
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - arch:                 Arch enum (0=ia32, 1=x64, 2=armv7l, 3=arm64, 4=universal)
 */
import { existsSync, mkdirSync, renameSync, rmSync, rmdirSync } from 'node:fs'
import path from 'node:path'
import { Arch } from 'electron-builder'
import { stageNodePty } from './stage-native-deps.mjs'

/** Subdirectory under ``release/`` where backups are parked. */
export const REBUILD_BACKUP_DIRNAME = '.rebuild-backup'

/**
 * Derive the nested backup directory path from the app output directory.
 *
 * ``appOutDir`` is e.g. ``/build/release/win-unpacked``.
 * Returns ``/build/release/.rebuild-backup/win-unpacked``.
 *
 * @param {string} appOutDir
 * @returns {string}
 */
export function staleBackupPath(appOutDir) {
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
export function cleanStaleAppOutDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') {
    return { removed: false, backedUp: false }
  }
  if (!existsSync(appOutDir)) {
    return { removed: false, backedUp: false }
  }

  const backupDir = staleBackupPath(appOutDir)

  // Remove a leftover backup from a *previous* failed build so we can rename
  // the current directory cleanly.
  if (existsSync(backupDir)) {
    try {
      rmSync(backupDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 })
    } catch (_) {
      // If we can't even clean the old backup, fall through to rmSync below.
    }
  }

  // Ensure the parent .rebuild-backup/ directory exists.
  try {
    mkdirSync(path.dirname(backupDir), { recursive: true })
  } catch (_) {
    // Best-effort; renameSync will fail with its own error if parent missing.
  }

  // Try rename first (non-destructive, preserves the last good build).
  try {
    renameSync(appOutDir, backupDir)
    return { removed: true, backedUp: true }
  } catch (renameErr) {
    console.warn(
      `[before-pack] rename to backup failed (${renameErr.message}); ` +
        `falling back to rmSync — no backup will be available`
    )
  }

  // Fallback: delete outright so electron-builder can proceed.
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
    // Never fail the build over cleanup; surface why so a genuinely stuck
    // directory (permissions, mount) is still diagnosable.
    console.warn(`[before-pack] error cleaning ${appOutDir} (${err.message}); continuing`)
  }

  try {
    const platform = context && context.electronPlatformName
    const archName = context && typeof context.arch === 'number' ? Arch[context.arch] : undefined
    if (platform && archName) {
      if (archName === 'universal') {
        console.warn(
          '[before-pack] target arch is "universal" — node-pty has no universal prebuild; ' +
            'staged binary will be whichever single-arch copy npm run build left behind. ' +
            'lipo-merge x64/arm64 .node files manually if you need a true universal build.'
        )
      } else {
        await stageNodePty({ platform, arch: archName })
        console.log(`[before-pack] re-staged node-pty for target ${platform}-${archName}`)
      }
    }
  } catch (err) {
    // This one SHOULD fail the build — a missing/wrong native binary for the
    // target arch means a broken package shipped to users, which is worse
    // than a build that fails loudly here.
    throw new Error(`[before-pack] failed to stage node-pty for this target: ${err.message}`)
  }
}
