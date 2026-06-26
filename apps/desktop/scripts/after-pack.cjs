/**
 * after-pack.cjs — electron-builder afterPack hook.
 *
 * 1. Cleans up the `.bak` backup directory left by before-pack.cjs — the
 *    build succeeded so the backup is no longer needed.
 * 2. Stamps the Hermes icon + identity onto the packed Windows Hermes.exe via
 *    rcedit (delegated to set-exe-identity.cjs).
 *
 * WHY THE BACKUP CLEANUP IS HERE
 * ------------------------------
 * before-pack.cjs renames the old `appOutDir` to `<appOutDir>.bak` to
 * preserve a fallback exe when the build fails. This hook runs only after
 * electron-builder has staged a fresh, complete tree — so it is now safe to
 * discard the backup.
 *
 * Best-effort: a cleanup failure must never fail an otherwise-good build
 * (worst case is disk waste, not a broken app), so we log and resolve.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

const fs = require('node:fs')
const path = require('node:path')

const { staleBackupPath } = require('./before-pack.cjs')
const { stampExeIdentity } = require('./set-exe-identity.cjs')

/**
 * Remove the `.bak` backup directory left by before-pack.cjs.
 * Best-effort: never throws.
 *
 * @param {string} appOutDir
 */
function cleanStaleBackupDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') {
    return
  }
  const backupDir = staleBackupPath(appOutDir)
  if (!fs.existsSync(backupDir)) {
    return
  }
  try {
    fs.rmSync(backupDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
    console.log(`[after-pack] removed backup: ${backupDir}`)
  } catch (err) {
    console.warn(
      `[after-pack] could not remove backup ${backupDir} (${err.message}); ` +
        `safe to delete manually`
    )
  }
}

exports.cleanStaleBackupDir = cleanStaleBackupDir

exports.default = async function afterPack(context) {
  // Clean up the backup from before-pack now that the build succeeded.
  // This runs for ALL platforms — the backup rename is cross-platform.
  try {
    cleanStaleBackupDir(context.appOutDir)
  } catch (_) {
    // Never fail the build over cleanup.
  }

  // Windows-only: stamp the exe icon + identity.
  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(__dirname, '..')

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); Hermes.exe keeps the stock Electron icon`)
  }
}
