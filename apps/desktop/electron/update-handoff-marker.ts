/**
 * update-handoff-marker.ts — durable handshake for the Tauri updater hand-off.
 *
 * When the desktop app clicks "Update now" and a staged `hermes-setup` binary
 * exists, it spawns that updater detached and quits to release the venv shim
 * (#57645). The swap/relaunch is supposed to happen inside hermes-setup, but
 * if it fails silently — the updater crashes, is blocked by Gatekeeper, or
 * simply never completes — the desktop relaunches into the SAME old version
 * with no indication anything went wrong. The user sees the "update
 * available" pill again as if they never clicked Update.
 *
 * This module provides a durable, self-healing marker so the relaunched
 * instance can detect that a hand-off was attempted and whether the version
 * actually changed:
 *
 *   1. Before quitting, writeHermesUpdateHandoff() records the pre-hand-off
 *      git HEAD SHA, the Hermes version string, and a timestamp into
 *      HERMES_HOME/.hermes-update-handoff.json.
 *   2. On next boot, readUpdateHandoffResult() reads that marker and compares
 *      its recorded SHA against the current git HEAD. If they match, the
 *      update did NOT take → return { failed: true, ... } so the renderer
 *      can land on a closeable error state instead of silently reverting.
 *   3. If the SHA changed (update succeeded) or no marker exists, return
 *      { failed: false }.
 *   4. The marker self-heals: it is deleted after being read, and a stale
 *      marker (older than the age ceiling) is treated as "no hand-off" so a
 *      long-dormant marker from a crashed update doesn't false-positive
 *      weeks later.
 *
 * Pure-ish (file I/O + injectable git SHA resolver) so it is unit-testable
 * without booting Electron.
 */

import fs from 'fs'
import path from 'path'

// A hand-off marker older than this is treated as stale and pruned. A real
// update (git pull + pip + desktop rebuild) completes in minutes; if the app
// is relaunched more than this long after the hand-off, either the update
// succeeded (and we'd see a new SHA) or it's been so long that surfacing a
// stale failure is more confusing than helpful.
export const HANDOFF_MARKER_MAX_AGE_MS = 30 * 60 * 1000

export function handoffMarkerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-handoff.json')
}

/**
 * Read the current git HEAD SHA from the given update root.
 * Returns '' if git is unavailable or the checkout has no HEAD.
 * Injectable via `resolveSha` for unit tests.
 */
export function defaultResolveSha(updateRoot) {
  try {
    const { execFileSync } = require('child_process')
    const sha = execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: updateRoot,
      encoding: 'utf8',
      timeout: 5000,
      stdio: ['pipe', 'pipe', 'pipe']
    }).trim()
    return sha
  } catch {
    return ''
  }
}

/**
 * Write the hand-off marker before quitting for the updater.
 *
 * @param {string} hermesHome
 * @param {object} payload
 * @param {string} payload.sha       git HEAD SHA at hand-off time ('' if unknown)
 * @param {string} payload.version   Hermes version string at hand-off time
 * @param {number} [payload.at]      unix-ms timestamp (defaults to now)
 */
export function writeHermesUpdateHandoff(hermesHome, { sha, version, at }: any = {}) {
  const markerPath = handoffMarkerPath(hermesHome)
  try {
    fs.mkdirSync(path.dirname(markerPath), { recursive: true })
  } catch {
    // directory likely exists already
  }
  const body = JSON.stringify({
    sha: sha || '',
    version: version || '',
    at: at || Date.now()
  })
  // Atomic write (temp + rename) so a crash mid-write can't leave a
  // half-written marker that would be misparsed on next boot.
  const tmp = markerPath + '.tmp'
  try {
    fs.writeFileSync(tmp, body, 'utf8')
    fs.renameSync(tmp, markerPath)
  } catch {
    // If we can't write the marker, the worst case is we lose the
    // post-relaunch validation — the update still proceeds normally.
    try { fs.unlinkSync(tmp) } catch { /* ignore */ }
  }
}

/**
 * Read and interpret the hand-off marker on boot.
 *
 * Returns { pending: false } when there is no actionable hand-off marker
 * (absent, unreadable, malformed, or stale). When a valid marker exists:
 *
 *   - If the current SHA differs from the recorded SHA → the update took;
 *     return { pending: false, succeeded: true } and prune the marker.
 *   - If the current SHA matches the recorded SHA (or the SHA can't be
 *     resolved and the marker is fresh) → the update did NOT take;
 *     return { pending: true, failed: true, ... } so the renderer can
 *     surface a closeable error. The marker is pruned either way.
 *
 * @param {string} hermesHome
 * @param {string} updateRoot      git checkout root for SHA comparison
 * @param {object} [opts]
 * @param {function} [opts.resolveSha]  injectable SHA resolver (tests)
 * @param {function} [opts.now]         injectable clock (tests)
 * @param {number}   [opts.maxAgeMs]    injectable staleness ceiling (tests)
 */
export function readUpdateHandoffResult(hermesHome, updateRoot, opts: any = {}) {
  const resolveSha = opts.resolveSha || defaultResolveSha
  const now = opts.now || Date.now
  const maxAgeMs = opts.maxAgeMs || HANDOFF_MARKER_MAX_AGE_MS

  const markerPath = handoffMarkerPath(hermesHome)
  let raw
  try {
    raw = fs.readFileSync(markerPath, 'utf8')
  } catch {
    return { pending: false }
  }

  let marker
  try {
    marker = JSON.parse(raw)
  } catch {
    // Malformed marker — prune and move on.
    try { fs.unlinkSync(markerPath) } catch { /* ignore */ }
    return { pending: false }
  }

  if (!marker || typeof marker !== 'object') {
    try { fs.unlinkSync(markerPath) } catch { /* ignore */ }
    return { pending: false }
  }

  const at = Number(marker.at)
  const ageMs = Number.isFinite(at) ? now() - at : Infinity

  // Stale marker — the hand-off was too long ago to act on. Prune so it
  // doesn't false-positive on a much later relaunch.
  if (ageMs > maxAgeMs) {
    try { fs.unlinkSync(markerPath) } catch { /* ignore */ }
    return { pending: false }
  }

  const recordedSha = String(marker.sha || '')
  const currentSha = resolveSha(updateRoot)

  // Prune the marker regardless of outcome — it's a one-shot.
  try { fs.unlinkSync(markerPath) } catch { /* ignore */ }

  // If we can resolve the current SHA and it changed, the update succeeded.
  if (currentSha && recordedSha && currentSha !== recordedSha) {
    return { pending: false, succeeded: true }
  }

  // SHA unchanged (or couldn't be resolved) within the freshness window →
  // the update did not take. Surface a closeable error to the user instead
  // of silently reverting to the old version.
  return {
    pending: true,
    failed: true,
    recordedSha,
    currentSha,
    recordedVersion: String(marker.version || ''),
    ageMs
  }
}
