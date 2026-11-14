import { randomUUID } from 'node:crypto'

const featureFlags = {
  /** Local-models GUI surfaces (settings pane, pickers, statusbar, tips). */
  localModels: ({ argv }) =>
    process.platform === 'win32' ||
    process.platform === 'darwin' ||
    (process.platform === 'linux' && (process.arch === 'x64' || process.arch === 'arm64')) ||
    argv.includes('--local')
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
}

export function resolveFeatureFlags(input: FeatureFlagInput): FeatureFlags {
  return Object.fromEntries(Object.entries(featureFlags).map(([k, v]) => [k, v(input)])) as FeatureFlags
}

/** This main process's identity: every window of one launch reads the same id, a relaunch mints a
 *  new one. The renderer keys per-launch choices on it (the redesign's rotating artwork). */
const LAUNCH_ID = randomUUID()

export interface LaunchFacts extends FeatureFlags {
  assumeFreshInstall: boolean
  launchId: string
}

/** What `hermes:feature-flags` answers each window's preload before its first paint. */
export function resolveLaunchFacts(input: FeatureFlagInput & { packaged: boolean }): LaunchFacts {
  return {
    ...resolveFeatureFlags(input),
    // Dev-only: `--fresh-install` makes onboarding treat this launch as a fresh install.
    assumeFreshInstall: !input.packaged && input.argv.includes('--fresh-install'),
    launchId: LAUNCH_ID
  }
}
