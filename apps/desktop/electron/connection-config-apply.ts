interface ApplyConnectionConfigAtomicallyOptions<TConfig, TRegistry> {
  apply: () => Promise<void>
  nextConfig: TConfig
  nextRegistry: TRegistry
  /**
   * Optional reachability check (authenticated REST + a real WebSocket leg).
   * Runs BEFORE either file is written, so a rejected OAuth session or a
   * blocked /api/ws leaves the previous primary/current connection intact
   * rather than committing a gateway the app cannot actually reach.
   *
   * Should resolve `{ wsVerified: boolean }` so a later `apply()` failure can
   * tell a fully-proven config (skip rollback) from one where the WS leg
   * itself was never exercised — e.g. a tokenless gateway, where the REST
   * probe passes but nothing tested the transport the app actually uses.
   */
  preflight?: () => Promise<{ wsVerified?: boolean }>
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
  const preflightResult = await preflight?.()

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
    // A preflight that actually verified the WS leg already exercised the
    // authenticated REST + real WebSocket transport against the config we
    // just wrote, so it is proven reachable. A failure here is a live
    // re-home/teardown hiccup (tearing down the OLD primary, an in-flight
    // dial to a dead gateway, a "not ready yet" race), not evidence the new
    // connection is bad — and rolling back would silently undo the one
    // action (editing and saving a new URL) meant to escape a dead gateway,
    // so the next launch dials the old, unreachable address again
    // (#123225). Only roll back when the WS leg was never proven: no
    // preflight ran, or the preflight could not exercise the WS transport
    // (e.g. a tokenless gateway with nothing to build a test URL from).
    if (!preflightResult || preflightResult.wsVerified !== true) {
      try {
        writeConfig(previousConfig)
        writeRegistry(previousRegistry)
      } catch {
        // Preserve the original activation failure, for the same reason as
        // the write rollback above.
      }
    }

    throw error
  }
}
