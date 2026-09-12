// install-stamp.ts — the typed build-time install stamp.
//
// scripts/write-build-stamp.mjs writes build/install-stamp.json during
// `npm run build`.
// bundle-electron-main.mjs bakes that file into the
// production bundle by defining the __HERMES_INSTALL_STAMP__ global as
// the stamp.  The stamp is a constant of the artifact.
// It cannot be missing, stale, or edited after signing.
//
// Dev bundles and test runs define nothing; the typeof guard makes the
// stamp null there (a dev run has no artifact to be truthful about).

/**
 * The desktop artifact kind — which runtime story this artifact tells:
 *  - 'bootstrap': no runtime in the artifact; first launch bootstraps a
 *    local install (the default; also what non-desktop stamps carry).
 *  - 'bundled': the agent runtime ships inside the artifact resources.
 *  - 'light': no runtime at all; remote connections only.
 * Selected at build time by HERMES_DESKTOP_VARIANT (unset = bootstrap).
 */
export type ArtifactKind = 'bootstrap' | 'bundled' | 'light'

/** Relative paths declared by the PM bundle builder, below agent-payload. */
export interface PayloadRuntime {
  repoDir: string
  toolsDir: string
  storePython: string
  sitePackages: string
  commands: Record<string, string>
}

/** Mirrors the build stamp with the PM builder's completed launch contract. */
export interface InstallStamp {
  schemaVersion: number
  commit: string | null
  commitDate: number | null
  branch: string | null
  builtAt: string | null
  dirty: boolean
  /** Build provenance: where the stamp's facts came from. */
  source: 'build' | 'commit-build' | 'ci' | 'docker' | 'fallback' | 'git' | 'local' | 'nix' | 'unknown' | null
  /** The steward of a sealed tree ('desktop-app' | 'docker' | 'nix'), when packaged. */
  distribution: string | null
  /** Who applies the next update. Required in every stamp. */
  updateMechanism: 'self' | 'app-installer' | 'electron-updater' | 'external' | 'microsoft-store'
  baseVersion: string | null
  displayVersion: string | null
  distance: number | null
  payload: ArtifactKind
  /** Present on bundled artifacts. Validated at build time, never discovered at boot. */
  runtime?: PayloadRuntime
  /** The pinned release tag. Always set for 'bundled' and 'light', never for 'bootstrap'. */
  tag: string | null
}

declare const __HERMES_INSTALL_STAMP__: InstallStamp

/** The baked stamp of this artifact, or null on dev bundles. */
export const INSTALL_STAMP: Readonly<InstallStamp> | null =
  typeof __HERMES_INSTALL_STAMP__ === 'undefined' ? null : Object.freeze(__HERMES_INSTALL_STAMP__)

/**
 * The install shape this process runs as — THE single split every
 * lifecycle decision gates on (mirror of Python's runtime_tree()):
 *  - 'bundled': the runtime ships inside the artifact. Venv machinery,
 *    installers, repair-reinstall escalation and update checkouts must
 *    never run; drift means rebuild, updates mean the steward.
 *  - 'checkout': a git tree with venv machinery, provisioner-on-demand
 *    and `hermes update`.
 *
 * Derived from the stamp CONSTANT, never from filesystem probes: a
 * payload/venv/marker probe answers "is this artifact intact?", not
 * "which shape am I?". PM and the bundle builder own payload integrity.
 * A backend launch failure must not quietly turn a bundle into a checkout. Dev runs
 * (null stamp) and bootstrap artifacts are 'checkout': their runtime
 * is a local install the app bootstraps and maintains.
 */
export function installShape(stamp: Readonly<InstallStamp> | null = INSTALL_STAMP): 'bundled' | 'checkout' {
  return stamp?.payload === 'bundled' ? 'bundled' : 'checkout'
}
