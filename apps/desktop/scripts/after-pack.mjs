/**
 * after-pack.mjs — electron-builder afterPack hook.
 *
 * 1. Cleans up the nested backup under release/.rebuild-backup/<dirname>
 *    left by before-pack.mjs — the build succeeded.
 * 2. Stamps the Hermes icon + identity onto Windows Hermes.exe via rcedit.
 */
import { existsSync, rmSync, rmdirSync } from 'node:fs'
import path from 'node:path'

import { staleBackupPath } from './before-pack.mjs'
import { stampExeIdentity } from './set-exe-identity.mjs'

export function cleanStaleBackupDir(appOutDir) {
  if (!appOutDir || typeof appOutDir !== 'string') return
  const backupDir = staleBackupPath(appOutDir)
  if (!existsSync(backupDir)) return
  try {
    rmSync(backupDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
    console.log(`[after-pack] removed backup: ${backupDir}`)
  } catch (err) {
    console.warn(`[after-pack] could not remove backup ${backupDir} (${err.message}); safe to delete manually`)
    return
  }
  try { rmdirSync(path.dirname(backupDir)) } catch (_) {}
}

export default async function afterPack(context) {
  try { cleanStaleBackupDir(context.appOutDir) } catch (_) {}
  if (context.electronPlatformName !== 'win32') return
  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')
  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    console.warn(`[after-pack] exe identity stamp failed (${err.message})`)
  }
}
