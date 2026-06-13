/**
 * Helpers for comparing the running desktop bundle's install stamp against
 * the source checkout. Kept outside main.cjs so the stamp path stays covered
 * by node --test without booting Electron.
 */

const fs = require('node:fs')
const path = require('node:path')
const {
  EMPTY_DESKTOP_SOURCE_STATE_HASH,
  currentDesktopSourceStateHash,
  hashText
} = require('./source-state-hash.cjs')

const EMPTY_TRACKED_SOURCE_DIFF_HASH = EMPTY_DESKTOP_SOURCE_STATE_HASH
const currentTrackedSourceDiffHash = currentDesktopSourceStateHash

function readInstallStamp(bundlePath, fsImpl = fs) {
  if (!bundlePath) return null

  const stampPath = path.join(bundlePath, 'Contents', 'Resources', 'install-stamp.json')
  if (!fsImpl.existsSync(stampPath)) return null

  return JSON.parse(fsImpl.readFileSync(stampPath, 'utf8'))
}

function bundleNeedsRebuild(bundlePath, currentSha, fsImpl = fs, options = {}) {
  if (!bundlePath || !currentSha) return false

  try {
    const stamp = readInstallStamp(bundlePath, fsImpl)
    const stampCommit = String(stamp?.commit || '').trim()
    if (!stampCommit) return false
    if (stampCommit !== currentSha) return true

    const currentDiffHash =
      typeof options.currentTrackedSourceDiffHash === 'string'
        ? options.currentTrackedSourceDiffHash
        : currentTrackedSourceDiffHash(options.repoRoot)
    if (!currentDiffHash) return false

    if (typeof stamp.trackedSourceDiffHash === 'string') {
      return stamp.trackedSourceDiffHash !== currentDiffHash
    }

    return currentDiffHash !== EMPTY_TRACKED_SOURCE_DIFF_HASH
  } catch {
    return false
  }
}

module.exports = {
  EMPTY_TRACKED_SOURCE_DIFF_HASH,
  bundleNeedsRebuild,
  currentTrackedSourceDiffHash,
  hashText,
  readInstallStamp
}
