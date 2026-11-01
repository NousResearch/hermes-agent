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
 * X.Y.Z+commit.<stamp> / X.Y.Z+channel.<stamp> for a canary, commit or channel claim
 * (hermes_cli/update_channel.py release_version). Never strip a leading `v`: prefix tags start with
 * their kind. */
export function releaseTagVersion(tag: string | null | undefined): string | null {
  const match = /^(?:(canary|commit|channel)\.(20\d{6}T\d{6}Z)-)?v(\d+\.\d+\.\d+)$/.exec(tag || '')

  if (!match) {
    return null
  }

  return match[1] ? `${match[3]}+${match[1]}.${match[2]}` : match[3]
}

/** The Windows package quad yy.mmdd.hh.mmss a stamped claim tag (canary, commit or channel) carries:
 * the same value as scripts/msix-shared.mjs canaryPackageVersionAt and
 * hermes_cli/update_channel.py stamped_windows_version. Null for any other tag. */
export function stampedWindowsVersion(tag: string | null | undefined): string | null {
  const match = /^(?:canary|commit|channel)\.(20\d{2})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z-v\d+\.\d+\.\d+$/.exec(tag || '')

  if (!match) {
    return null
  }

  const [year, month, day, hour, minute, second] = match.slice(1).map(Number)

  return `${year % 100}.${month * 100 + day}.${hour}.${minute * 100 + second}`
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
