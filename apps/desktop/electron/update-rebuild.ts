/**
 * Retry-once policy for the desktop `--build-only` rebuild during self-update.
 *
 * The first rebuild can return nonzero on a still-settling post-update tree or a
 * network-blocked Electron fetch that the installer's self-heal repaired mid-run.
 * A second attempt then builds clean off the healed dist (the content-hash stamp
 * makes it a near-no-op when the first actually succeeded). Without the retry the
 * updater bails before the relaunch step — the app updates but doesn't restart.
 */

function shouldRetryRebuild(code) {
  return code !== 0
}

/** Return only bundles runnable by the current macOS architecture. */
function macRebuiltBundleCandidates(updateRoot, arch, pathModule = path) {
  const macArch = arch === 'x64' ? 'mac-x64' : 'mac-arm64'
  const releaseDir = pathModule.join(updateRoot, 'apps', 'desktop', 'release')

  return [
    pathModule.join(releaseDir, macArch, 'Hermes.app'),
    pathModule.join(releaseDir, 'mac', 'Hermes.app')
  ]
}

/**
 * Run `rebuild()` (async, resolves `{ code, ... }`), retrying once on failure.
 * Returns the final result.
 */
async function runRebuildWithRetry(rebuild) {
  let result = await rebuild(0)

  if (shouldRetryRebuild(result.code)) {
    result = await rebuild(1)
  }

  return result
}

export { macRebuiltBundleCandidates, runRebuildWithRetry, shouldRetryRebuild }
import path from 'node:path'
