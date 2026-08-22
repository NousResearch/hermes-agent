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
 * electron-builder always stages into a clean tree. This is safe for stale or
 * structurally incomplete output. On Windows, however, a valid current app is
 * user rollback material: destructive replacement is allowed only after that
 * generation has been acquired transactionally.
 *
 * Cross-platform: the same partial-state trap exists on macOS
 * (the mac-unpacked Hermes.app bundle) and Windows (win-unpacked), so we
 * clean whatever `appOutDir` electron-builder hands us regardless of platform.
 *
 * Best-effort cleanup applies to stale/partial trees. Failure to acquire
 * rollback authority for a valid Windows app is different: the hook fails
 * closed rather than deleting the current working generation.
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
import { existsSync, rmSync, renameSync } from 'node:fs'
import path from 'node:path'
import { Arch } from 'electron-builder'
import { stageNodePty, stageGetWindows } from './stage-native-deps.mjs'
import {
  PACK_SESSION_ENV,
  clearRollbackSession,
  readRollbackSession,
  writeRollbackSession
} from './desktop-pack-transaction.mjs'

export const ROLLBACK_ACQUISITION_STATUS = Object.freeze({
  PRESERVED: 'preserved',
  SAFE_TO_CLEAN: 'safe-to-clean',
  BLOCKED: 'blocked'
})

function rollbackResult(status, reason, error, details = {}) {
  return {
    status,
    reason,
    ...(error ? { error } : {}),
    ...details
  }
}

function rollbackOperations(overrides) {
  const supplied = overrides && typeof overrides === 'object' ? overrides : {}
  return {
    existsSync: supplied.existsSync ?? existsSync,
    rmSync: supplied.rmSync ?? rmSync,
    renameSync: supplied.renameSync ?? renameSync,
    clearRollbackSession: supplied.clearRollbackSession ?? clearRollbackSession,
    readRollbackSession: supplied.readRollbackSession ?? readRollbackSession,
    writeRollbackSession: supplied.writeRollbackSession ?? writeRollbackSession
  }
}

function removeTree(rm, target) {
  rm(target, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
}

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
  removeTree(rmSync, appOutDir)
  return true
}

/**
 * Windows rollback material (#69179): before wiping the previous unpacked
 * tree, preserve it as `<appOutDir>.bak` — but ONLY when it holds the product
 * exe (i.e. it is a previously-working build, not the corrupted partial state
 * cleanStaleAppOutDir exists to remove). If the fresh pack then produces a
 * Hermes.exe that Windows can't load (truncated PE from a corrupt cached
 * Electron zip, wrong arch), the builder transaction restores this .bak
 * instead of leaving the user with "This app can't run on your computer".
 *
 * One electron-builder invocation may run beforePack more than once (multiple
 * Windows targets/architectures). The wrapper supplies one pack-session ID.
 * A matching `<appOutDir>.bak.session` proves the backup already belongs to
 * this invocation, so later targets clean their intermediate output without
 * overwriting the original rollback generation.
 *
 * The result is deliberately multi-state:
 *
 * - `preserved`: rollback authority exists for this generation;
 * - `safe-to-clean`: the current tree is absent/partial and may be wiped;
 * - `blocked`: a valid current or backup generation exists, but rollback
 *   authority could not be acquired. The caller must fail closed.
 */
export function preserveRollbackBackup(
  appOutDir,
  productExeName = 'Hermes.exe',
  sessionId = process.env[PACK_SESSION_ENV],
  operationOverrides
) {
  const operations = rollbackOperations(operationOverrides)
  if (!appOutDir || typeof appOutDir !== 'string') {
    return rollbackResult(
      ROLLBACK_ACQUISITION_STATUS.SAFE_TO_CLEAN,
      'invalid-or-missing-app-output'
    )
  }

  const backupDir = `${appOutDir}.bak`
  const currentExe = path.join(appOutDir, productExeName)
  const backupExe = path.join(backupDir, productExeName)

  if (!operations.existsSync(currentExe)) {
    // Partial/corrupt tree (interrupted prior pack) — not rollback material.
    // A valid backup from an interrupted older invocation remains useful, but
    // it must be adopted into this generation before packaging may proceed.
    if (sessionId && operations.existsSync(backupExe)) {
      try {
        operations.writeRollbackSession(backupDir, sessionId)
      } catch (error) {
        return rollbackResult(
          ROLLBACK_ACQUISITION_STATUS.BLOCKED,
          'existing-backup-adoption-failed',
          error,
          { backupDir }
        )
      }
    }
    return rollbackResult(
      ROLLBACK_ACQUISITION_STATUS.SAFE_TO_CLEAN,
      'current-package-is-partial-or-absent',
      undefined,
      { backupDir }
    )
  }

  const sameSessionBackup =
    Boolean(sessionId) &&
    operations.existsSync(backupExe) &&
    operations.readRollbackSession(backupDir) === sessionId

  if (sameSessionBackup) {
    // Multi-target pack: keep the first (pre-build) generation as authority.
    // The current tree is output from an earlier target in this same builder
    // process and must not replace the rollback generation.
    try {
      removeTree(operations.rmSync, appOutDir)
      return rollbackResult(
        ROLLBACK_ACQUISITION_STATUS.PRESERVED,
        'same-session-backup-retained',
        undefined,
        { backupDir }
      )
    } catch (error) {
      return rollbackResult(
        ROLLBACK_ACQUISITION_STATUS.BLOCKED,
        'same-session-output-cleanup-failed',
        error,
        { backupDir }
      )
    }
  }

  try {
    // Do not touch the valid current app unless the prior rollback slot and
    // marker can first be retired. A locked marker therefore blocks packaging.
    operations.clearRollbackSession(backupDir)
    removeTree(operations.rmSync, backupDir)
  } catch (error) {
    return rollbackResult(
      ROLLBACK_ACQUISITION_STATUS.BLOCKED,
      'rollback-slot-retirement-failed',
      error,
      { backupDir }
    )
  }

  // Stage the generation identity before moving the current package. If marker
  // creation fails, the live app has not been touched. If the subsequent
  // directory rename fails, marker cleanup is best-effort but the live app
  // still remains at appOutDir.
  if (sessionId) {
    try {
      operations.writeRollbackSession(backupDir, sessionId)
    } catch (error) {
      return rollbackResult(
        ROLLBACK_ACQUISITION_STATUS.BLOCKED,
        'rollback-session-write-failed',
        error,
        { backupDir, currentPackageUntouched: true }
      )
    }
  }

  try {
    operations.renameSync(appOutDir, backupDir)
  } catch (error) {
    let markerCleanupError
    if (sessionId) {
      try {
        operations.clearRollbackSession(backupDir)
      } catch (cleanupError) {
        markerCleanupError = cleanupError
      }
    }
    return rollbackResult(
      ROLLBACK_ACQUISITION_STATUS.BLOCKED,
      'current-package-preservation-failed',
      error,
      {
        backupDir,
        currentPackageUntouched: true,
        ...(markerCleanupError ? { markerCleanupError } : {})
      }
    )
  }

  return rollbackResult(
    ROLLBACK_ACQUISITION_STATUS.PRESERVED,
    'current-package-preserved',
    undefined,
    { backupDir }
  )
}

function rollbackBlockMessage(appOutDir, acquisition) {
  const primary =
    acquisition.error instanceof Error ? acquisition.error.message : String(acquisition.error || '')
  const markerCleanup =
    acquisition.markerCleanupError instanceof Error
      ? `; cleaning the staged rollback marker also failed: ${acquisition.markerCleanupError.message}`
      : ''
  const detail = primary ? `: ${primary}` : ''
  return (
    `[before-pack] refusing destructive Windows package replacement for ${appOutDir}: ` +
    `rollback acquisition blocked (${acquisition.reason})${detail}${markerCleanup}`
  )
}

export default async function beforePack(
  context,
  {
    rollbackOperations: operationOverrides,
    rollbackSessionId = process.env[PACK_SESSION_ENV]
  } = {}
) {
  const appOutDir = context && context.appOutDir
  const platformName = context && context.electronPlatformName
  const productExe = `${(context && context.packager?.appInfo?.productFilename) || 'Hermes'}.exe`

  if (platformName === 'win32') {
    const acquisition = preserveRollbackBackup(
      appOutDir,
      productExe,
      rollbackSessionId,
      operationOverrides
    )
    if (acquisition.status === ROLLBACK_ACQUISITION_STATUS.BLOCKED) {
      throw new Error(rollbackBlockMessage(appOutDir, acquisition))
    }
    if (acquisition.status === ROLLBACK_ACQUISITION_STATUS.PRESERVED) {
      console.log(`[before-pack] preserved previous unpacked dir for rollback: ${appOutDir}.bak`)
    } else {
      try {
        if (cleanStaleAppOutDir(appOutDir)) {
          console.log(`[before-pack] removed stale unpacked dir before staging: ${appOutDir}`)
        }
      } catch (err) {
        // A stale/partial tree is not rollback authority. Keep cleanup
        // best-effort so electron-builder can surface its canonical failure.
        console.warn(`[before-pack] could not clean ${appOutDir} (${err.message}); continuing`)
      }
    }
  } else {
    try {
      if (cleanStaleAppOutDir(appOutDir)) {
        console.log(`[before-pack] removed stale unpacked dir before staging: ${appOutDir}`)
      }
    } catch (err) {
      // Non-Windows cleanup remains best-effort.
      console.warn(`[before-pack] could not clean ${appOutDir} (${err.message}); continuing`)
    }
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
      // The macOS helper is universal, while Windows bindings are arch-specific.
      // Pass the target arch so an ARM64 package never stages an x64 binding.
      stageGetWindows({ platform, arch: archName })
      console.log(`[before-pack] re-staged get-windows for target ${platform}-${archName}`)
    }
  } catch (err) {
    // This one SHOULD fail the build — a missing/wrong native binary for the
    // target arch means a broken package shipped to users, which is worse
    // than a build that fails loudly here.
    throw new Error(`[before-pack] failed to stage native deps for this target: ${err.message}`)
  }
}
