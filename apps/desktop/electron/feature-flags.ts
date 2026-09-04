// feature-flags.ts — the desktop's feature-flag resolver.
//
// One place that decides which gated surfaces are on for this artifact.
// Flags resolve from two inputs:
//   - launch argv (e.g. `--local` on Hermes.exe, or `hermes desktop --local`)
//   - the release channel of the artifact (canary builds preview features)
// The verdict is static for the process lifetime; the renderer reads it
// once via the preload bridge (`hermes:feature-flags`).

export interface FeatureFlags {
  /** Local-models GUI surfaces (settings pane, pickers, statusbar, tips). */
  localModels: boolean
}

/** True when a baked install tag names a canary-channel build. */
export function isCanaryTag(tag: string | null | undefined): boolean {
  return /-canary\./.test(tag || '')
}

export interface FeatureFlagInput {
  /** The main process argv (launch flags like `--local`). */
  argv: readonly string[]
  /** Whether this artifact is a canary-channel build. */
  canary: boolean
}

/**
 * Resolve every feature flag for this process. A flag is on when the launch
 * argv opts in (`--local`) OR the artifact is a canary build — canary is the
 * preview channel, so gated features ride along by default there.
 */
export function resolveFeatureFlags({ argv, canary }: FeatureFlagInput): FeatureFlags {
  return {
    localModels: canary || argv.includes('--local')
  }
}
