/**
 * Pure logic for the status-bar version item.
 *
 * In local mode the desktop app and its backend are the same install, so a
 * single version is correct and we keep today's behaviour byte-for-byte. In
 * remote (thin-client) mode the Electron client and the backend it connects to
 * are separate installs that drift independently — so we surface BOTH the
 * client version and the connected backend version, and flag skew.
 *
 * Kept as a pure helper (no React, no stores) so it's unit-testable in
 * isolation; the status-bar hook composes the label/click from this.
 */

export interface VersionBarCopy {
  clientLabel: (version: string) => string
  backendLabel: (version: string) => string
  unknown: string
}

export interface ResolveVersionBarInput {
  /** Desktop client version (from the Electron getVersion IPC). */
  appVersion?: string
  /** Short git sha fallback when no semantic version is available. */
  sha?: string | null
  /** Backend version reported by the connected gateway (StatusResponse.version). */
  backendVersion?: string
  /** Connection topology. Only 'remote' shows two versions. */
  mode?: 'local' | 'remote'
  copy: VersionBarCopy
}

export interface VersionBarResult {
  /** Composed status-bar label. */
  label: string
  /** True when the backend version is shown alongside the client version. */
  showsBackend: boolean
  /** The backend version actually shown (when showsBackend). */
  backendVersion?: string
  /** True when remote client and backend versions differ. */
  skew: boolean
}

/**
 * Compute the version-bar label and skew state.
 *
 * Local mode (or remote with no backend version yet) → single client version,
 * identical to the pre-existing behaviour. Remote mode with a backend version →
 * "client vX · backend vY".
 */
export function resolveVersionBar(input: ResolveVersionBarInput): VersionBarResult {
  const { appVersion, sha, backendVersion, mode, copy } = input

  const clientBase = appVersion ? `v${appVersion}` : (sha ?? copy.unknown)

  const isRemote = mode === 'remote'
  const showsBackend = isRemote && Boolean(backendVersion) && Boolean(appVersion)

  if (!showsBackend) {
    return {
      label: clientBase,
      showsBackend: false,
      skew: false
    }
  }

  const skew = backendVersion !== appVersion

  return {
    label: `${copy.clientLabel(appVersion as string)} · ${copy.backendLabel(backendVersion as string)}`,
    showsBackend: true,
    backendVersion,
    skew
  }
}
