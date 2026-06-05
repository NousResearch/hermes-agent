'use strict'

/**
 * before-pack.cjs — electron-builder beforePack hook.
 *
 * Renames any existing unpacked app directory (`appOutDir`) to a backup
 * before electron-builder stages the Electron binaries. On success the
 * after-pack hook removes the backup; on failure the backup remains so the
 * previously working app is not lost.
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
 * WHY RENAME INSTEAD OF DELETE
 * ----------------------------
 * If the build/compile step fails after beforePack has removed the old
 * unpacked dir, the user is left with NO working app (the prior version
 * is gone and the new one never materialized). By renaming to a backup,
 * we preserve the previous build — the backup is only cleaned up AFTER
 * a successful pack completes (via after-pack.cjs). If the process is
 * interrupted or fails, the next run's beforePack will remove any stale
 * backup before proceeding again.
 *
 * Cross-platform: the same problem exists on macOS (mac-unpacked Hermes.app)
 * and Windows (win-unpacked), so we handle whatever `appOutDir` electron-builder
 * hands us regardless of platform.
 *
 * Best-effort: a rename/cleanup failure must never mask the real build.
 * We log and resolve rather than throw.
 *
 * electron-builder passes a context with:
 *   - appOutDir:            the unpacked app directory about to be staged
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 */

const fs = require('node:fs')

/**
 * Rename an existing appOutDir to a backup so electron-builder gets a clean
 * staging target, but the previous build is preserved in case this pack fails.
 *
 * Returns the backup path if a rename occurred, false otherwise.
 */
function stashAppOutDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') {
    return false
  }
  if (!fs.existsSync(appOutDir)) {
    return false
  }

  const backupDir = appOutDir + '.backup'

  // Remove any leftover backup from a prior interrupted run before we rename.
  if (fs.existsSync(backupDir)) {
    fs.rmSync(backupDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
  }

  // Rename is near-atomic on the same filesystem — much safer than a
  // delete-then-rebuild pattern.
  fs.renameSync(appOutDir, backupDir)
  return backupDir
}

exports.stashAppOutDir = stashAppOutDir

exports.default = async function beforePack(context) {
  const appOutDir = context && context.appOutDir
  try {
    const backupDir = stashAppOutDir(appOutDir)
    if (backupDir) {
      console.log(`[before-pack] stashed previous unpacked dir as backup: ${backupDir}`)
    }
  } catch (err) {
    // Never fail the build over cleanup; surface why so a genuinely stuck
    // directory (permissions, mount) is still diagnosable.
    console.warn(`[before-pack] could not stash ${appOutDir} (${err.message}); continuing`)
  }
}
