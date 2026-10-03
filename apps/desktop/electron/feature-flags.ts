const PACKAGED_DESKTOP_PLATFORMS = ['win32', 'darwin', 'linux'] as const

const featureFlags = {
  /** Local-models GUI surfaces (settings pane, pickers, statusbar, tips). */
  localModels: ({ argv, platform }) =>
    (PACKAGED_DESKTOP_PLATFORMS as readonly string[]).includes(platform) || argv.includes('--local')
} satisfies Record<string, (args: FeatureFlagInput) => boolean>

export type FeatureFlags = { [K in keyof typeof featureFlags]: boolean }

export function isCanaryTag(tag: string | null | undefined): boolean {
  return /\+canary\.20\d{6}T\d{6}Z$/.test(tag || '')
}

export interface FeatureFlagInput {
  /** desktop launch flags */
  argv: readonly string[]
  /** Whether this artifact is a canary-channel build. */
  canary: boolean
  /** The platform this artifact was resolved for (`process.platform`). Data,
   * not the ambient global, so the flag table is unit-testable anywhere. */
  platform: NodeJS.Platform
}

export function resolveFeatureFlags(input: FeatureFlagInput): FeatureFlags {
  return Object.fromEntries(Object.entries(featureFlags).map(([k, v]) => [k, v(input)])) as FeatureFlags
}
