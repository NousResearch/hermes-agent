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
    await apply()
  } catch (error) {
    try {
      writeConfig(previousConfig)
      writeRegistry(previousRegistry)
    } catch {
      // Preserve the original activation/write failure. Both storage writers
      // are atomic replacements, so a rollback failure cannot be repaired by
      // retrying one side blindly here.
    }

    throw error
  }
}
