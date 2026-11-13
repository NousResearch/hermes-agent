/** Artifact identity for recovery witnesses, adapted from kokhlo's #131067. */
import type { InstallStamp } from './install-stamp'

/** Version labels alone cannot distinguish source builds or same-version rebuilds. */
export function launchBuildIdentity(options: {
  appVersion?: string | null
  installStamp?: Readonly<Pick<InstallStamp, 'commit' | 'builtAt' | 'dirty'>> | null
}): string {
  const appVersion = String(options.appVersion ?? '').trim() || '0.0.0'
  const stamp = options.installStamp
  const commit = String(stamp?.commit ?? '').trim()
  const builtAt = String(stamp?.builtAt ?? '').trim()

  if (!commit && !builtAt) {
    return appVersion
  }

  return `${appVersion}+${commit ? `g${commit.slice(0, 12)}` : 'unknown'}${stamp?.dirty ? '-dirty' : ''}${builtAt ? `@${builtAt}` : ''}`
}

/** Legacy witnesses get one probe only when this build can record a real identity. */
export function launchMarkerNeedsReprobe(
  marker: { version?: string | null; build?: string | null } | null | undefined,
  options: { appVersion?: string | null; buildIdentity?: string | null }
): boolean {
  const appVersion = String(options.appVersion ?? '').trim()
  const buildIdentity = String(options.buildIdentity ?? '').trim()

  if (!marker || !appVersion) {
    return false
  }

  if (marker.version && marker.version !== appVersion) {
    return true
  }

  return Boolean(buildIdentity && buildIdentity !== appVersion) && marker.build !== buildIdentity
}
