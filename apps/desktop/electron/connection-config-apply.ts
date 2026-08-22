interface ApplyConnectionConfigAtomicallyOptions<TConfig, TRegistry> {
  apply: () => Promise<void>
  nextConfig: TConfig
  nextRegistry: TRegistry
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
  previousConfig,
  previousRegistry,
  writeConfig,
  writeRegistry
}: ApplyConnectionConfigAtomicallyOptions<TConfig, TRegistry>): Promise<void> {
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
