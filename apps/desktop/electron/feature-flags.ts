const featureFlags = {
  /** Local-models GUI surfaces (settings pane, pickers, statusbar, tips). */
  localModels: ({ argv }) =>
    process.platform === 'win32' ||
    process.platform === 'darwin' ||
    (process.platform === 'linux' && (process.arch === 'x64' || process.arch === 'arm64')) ||
    argv.includes('--local')
} satisfies Record<string, (args: FeatureFlagInput) => boolean>

export type FeatureFlags = { [K in keyof typeof featureFlags]: boolean }

/** A canary tag: canary.<UTC stamp>-vX.Y.Z (hermes_cli/update_channel.py _CANARY_TAG_RE). */
export function isCanaryTag(tag: string | null | undefined): boolean {
  return /^canary\.20\d{6}T\d{6}Z-v\d+\.\d+\.\d+$/.test(tag || '')
}

/** The version a release identity tag carries: X.Y.Z, or X.Y.Z+canary.<stamp> /
 * X.Y.Z+commit.<stamp> for a canary or commit build (hermes_cli/update_channel.py release_version).
 * Never strip a leading `v`: canary and commit tags start with their kind. */
export function releaseTagVersion(tag: string | null | undefined): string | null {
  const match = /^(?:(canary|commit)\.(20\d{6}T\d{6}Z)-)?v(\d+\.\d+\.\d+)$/.exec(tag || '')

  if (!match) {
    return null
  }

  return match[1] ? `${match[3]}+${match[1]}.${match[2]}` : match[3]
}

export interface FeatureFlagInput {
  /** desktop launch flags */
  argv: readonly string[]
  /** Whether this artifact is a canary-channel build. */
  canary: boolean
}

export function resolveFeatureFlags(input: FeatureFlagInput): FeatureFlags {
  return Object.fromEntries(Object.entries(featureFlags).map(([k, v]) => [k, v(input)])) as FeatureFlags
}
