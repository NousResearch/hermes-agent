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

export { runRebuildWithRetry, shouldRetryRebuild }
