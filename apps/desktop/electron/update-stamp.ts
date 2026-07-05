/**
 * Helpers for comparing the running desktop bundle's install stamp against
 * the source checkout. Kept outside main.ts so the stamp path stays covered
 * by node --test without booting Electron.
 */

import fs from 'node:fs'
import path from 'node:path'

export function readInstallStamp(bundlePath, fsImpl = fs) {
  if (!bundlePath) return null

  const stampPath = path.join(bundlePath, 'Contents', 'Resources', 'install-stamp.json')
  if (!fsImpl.existsSync(stampPath)) return null

  return JSON.parse(fsImpl.readFileSync(stampPath, 'utf8'))
}

export function bundleNeedsRebuild(bundlePath, currentSha, fsImpl = fs) {
  if (!bundlePath || !currentSha) return false

  try {
    const stamp = readInstallStamp(bundlePath, fsImpl)
    const stampCommit = String(stamp?.commit || '').trim()
    return Boolean(stampCommit && stampCommit !== currentSha)
  } catch {
    return false
  }
}
