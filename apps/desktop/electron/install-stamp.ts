// install-stamp.ts — the typed build-time install stamp.
//
// scripts/write_install_stamp.py writes build/install-stamp.json during
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

/** Mirrors the dict scripts/write_install_stamp.py::build_stamp returns. */
export interface InstallStamp {
  schemaVersion: number
  commit: string | null
  commitDate: number | null
  branch: string | null
  builtAt: string | null
  dirty: boolean
  /** Build provenance: where the stamp's facts came from. */
  source: 'build' | 'ci' | 'docker' | 'fallback' | 'git' | 'local' | 'nix' | 'unknown' | null
  /** The steward of a sealed tree ('desktop-app' | 'docker' | 'nix'), when packaged. */
  distribution: string | null
  /** Who applies the next update. Required in every stamp. */
  updateMechanism: 'self' | 'electron-updater' | 'external'
  baseVersion: string | null
  displayVersion: string | null
  distance: number | null
  payload: ArtifactKind
  /** True for the Store-submission build (HERMES_DESKTOP_VARIANT=store). */
  store?: boolean
  /** The pinned release tag. Always set for 'bundled' and 'light', never for 'bootstrap'. */
  tag: string | null
}

declare const __HERMES_INSTALL_STAMP__: InstallStamp | undefined

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
 * "which shape am I?" — those probes remain only as integrity checks
 * inside an already-chosen shape (a bundled stamp with a damaged
 * payload throws; it must not quietly become a checkout). Dev runs
 * (null stamp) and bootstrap artifacts are 'checkout': their runtime
 * is a local install the app bootstraps and maintains.
 */
export function installShape(stamp: Readonly<InstallStamp> | null = INSTALL_STAMP): 'bundled' | 'checkout' {
  return stamp?.payload === 'bundled' ? 'bundled' : 'checkout'
}
