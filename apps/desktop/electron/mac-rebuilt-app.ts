/**
 * Resolve the rebuilt macOS `.app` bundle produced by the in-app update's
 * `hermes desktop --build-only` step, so the swap/relaunch can install it.
 *
 * electron-builder writes the bundle under a platform/arch-specific directory:
 *   - `release/mac-arm64/Hermes.app`  (Apple Silicon)
 *   - `release/mac-x64/Hermes.app`    (Intel, when the build pins the arch)
 *   - `release/mac/Hermes.app`        (host-arch default, no explicit arch)
 *
 * The updater only checked `mac-arm64` and `mac`, so an Intel rebuild landing in
 * `release/mac-x64` was missed: the swap/relaunch found no bundle and the update
 * silently degraded to "Restart Hermes to load the new version" instead of
 * installing the rebuilt app (issue #48160).
 *
 * Resolve the running process's arch first — the same assumption the desktop
 * build/validation path makes (`scripts/test-desktop.mjs` maps arm64 ->
 * `mac-arm64`, else `mac-x64`) — then fall back to the generic `mac` dir.
 * Preferring the host arch also stops an Intel host from selecting a stale
 * `mac-arm64` bundle left by an earlier cross-build, and a non-x64/non-arm64
 * `process.arch` falls back to `mac` only rather than guessing an arch dir.
 */

import fs from 'node:fs'
import path from 'node:path'

function directoryExists(candidate) {
  try {
    return fs.statSync(candidate).isDirectory()
  } catch {
    return false
  }
}

/**
 * Return the rebuilt `Hermes.app` for this host, or undefined when none exists.
 * `arch` and `exists` are injectable so the resolution stays unit-testable
 * without a real release tree on disk.
 */
function resolveRebuiltMacApp(updateRoot, { arch = process.arch, exists = directoryExists } = {}) {
  const releaseDir = path.join(updateRoot, 'apps', 'desktop', 'release')
  const archDir = arch === 'arm64' ? 'mac-arm64' : arch === 'x64' ? 'mac-x64' : null
  const candidates = archDir ? [archDir, 'mac'] : ['mac']

  return candidates.map(dir => path.join(releaseDir, dir, 'Hermes.app')).find(exists)
}

export { resolveRebuiltMacApp }
