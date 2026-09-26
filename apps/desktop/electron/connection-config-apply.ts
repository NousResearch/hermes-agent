interface ApplyConnectionConfigAtomicallyOptions<TConfig, TRegistry> {
  apply: () => Promise<void>
  nextConfig: TConfig
  nextRegistry: TRegistry
  /**
   * Optional reachability check (authenticated REST + a real WebSocket leg).
   * Runs BEFORE either file is written, so a rejected OAuth session or a
   * blocked /api/ws leaves the previous primary/current connection intact
   * rather than committing a gateway the app cannot actually reach.
   */
  preflight?: () => Promise<unknown>
  previousConfig: TConfig
  previousRegistry: TRegistry
  writeConfig: (config: TConfig) => void
  writeRegistry: (registry: TRegistry) => void
}

/**
 * Commit the legacy config and v2 registry as one recoverable Apply boundary.
 * File replacement itself is atomic per file; this wrapper restores both
 * previous snapshots when the second write or synchronous re-home fails.
 */
export async function applyConnectionConfigAtomically<TConfig, TRegistry>({
  apply,
  nextConfig,
  nextRegistry,
  preflight,
  previousConfig,
  previousRegistry,
  writeConfig,
  writeRegistry
}: ApplyConnectionConfigAtomicallyOptions<TConfig, TRegistry>): Promise<void> {
  // Outside the try: a preflight failure has written nothing, so there is
  // nothing to roll back and no reason to touch either store.
  await preflight?.()

  try {
    writeConfig(nextConfig)
    writeRegistry(nextRegistry)
  } catch (error) {
    try {
      writeConfig(previousConfig)
      writeRegistry(previousRegistry)
    } catch {
      // Preserve the original write failure. Both storage writers are atomic
      // replacements, so a rollback failure cannot be repaired by retrying
      // one side blindly here.
    }

    throw error
  }

  try {
    await apply()
  } catch (error) {
    // A completed preflight already exercised the authenticated REST + real
    // WebSocket leg against the config we just wrote, so it is proven
    // reachable. A later failure here is a live re-home/teardown hiccup
    // (tearing down the OLD primary, in-flight dial to a dead gateway, a
    // "not ready yet" race), not evidence the new connection is bad. Rolling
    // back would silently undo the one action (editing and saving a new URL)
    // meant to escape a dead gateway, so only roll back when nothing already
    // validated the config being applied.
    if (!preflight) {
      try {
        writeConfig(previousConfig)
        writeRegistry(previousRegistry)
      } catch {
        // Preserve the original activation failure, for the same reason as above.
      }
    }

    throw error
  }
}
