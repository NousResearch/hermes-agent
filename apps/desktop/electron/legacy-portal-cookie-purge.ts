/**
 * One-shot purge of portal cookies an earlier build left in the shared OAuth
 * cookie partition (Hermes Cloud used to sign in there). The marker is
 * written only after a COMPLETE purge, so a failed or partial one retries on
 * the next launch; the purge itself is harmless to repeat.
 */
export interface LegacyPortalCookiePurgeDeps {
  markerExists: () => boolean
  /** Hydrate the persisted cookie jar before reading it (cold reads can be empty). */
  warm?: () => Promise<void>
  /** Resolves true only when every matching cookie was removed. */
  clearCookies: () => Promise<boolean>
  writeMarker: () => void
}

/** Resolves true when the purge is (now or already) recorded as done. */
export async function purgeLegacyPortalCookiesOnce(deps: LegacyPortalCookiePurgeDeps): Promise<boolean> {
  if (deps.markerExists()) {
    return true
  }

  await deps.warm?.()

  let cleared = false

  try {
    cleared = await deps.clearCookies()
  } catch {
    cleared = false
  }

  if (!cleared) {
    return false
  }

  try {
    deps.writeMarker()
  } catch {
    // Best effort: a repeat purge next launch is harmless.
  }

  return true
}
