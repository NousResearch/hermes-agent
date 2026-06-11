/**
 * Helpers for comparing the running desktop bundle's install stamp against
 * the source checkout. Kept outside main.cjs so the stamp path stays covered
 * by node --test without booting Electron.
 */

const fs = require('node:fs')
const path = require('node:path')

function readInstallStamp(bundlePath, fsImpl = fs) {
  if (!bundlePath) return null

  const stampPath = path.join(bundlePath, 'Contents', 'Resources', 'install-stamp.json')
  if (!fsImpl.existsSync(stampPath)) return null

  return JSON.parse(fsImpl.readFileSync(stampPath, 'utf8'))
}

function bundleNeedsRebuild(bundlePath, currentSha, fsImpl = fs) {
  if (!bundlePath || !currentSha) return false

  try {
    const stamp = readInstallStamp(bundlePath, fsImpl)
    const stampCommit = String(stamp?.commit || '').trim()
    return Boolean(stampCommit && stampCommit !== currentSha)
  } catch {
    return false
  }
}

module.exports = {
  bundleNeedsRebuild,
  readInstallStamp
}
