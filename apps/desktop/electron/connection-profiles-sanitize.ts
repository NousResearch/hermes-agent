import { modeIsRemoteLike, normalizeRemoteHeaders, normalizeRemoteProfileName, normalizeSshConfig, normAuthMode } from './connection-config'
import { DESKTOP_PROFILE_NAME_RE as PROFILE_NAME_RE } from './desktop-profile'

// Validate + normalize the per-profile remote overrides map read from disk.
// Drops malformed names/entries and keeps only the recognized fields so a
// hand-edited or stale connection.json can't inject junk into resolution.
export function sanitizeConnectionProfiles(raw: Record<string, any>) {
  if (!raw || typeof raw !== 'object') {
    return {}
  }

  const out = {}

  for (const [name, entry] of Object.entries(raw)) {
    if (!entry || typeof entry !== 'object') {
      continue
    }

    if (name !== 'default' && !PROFILE_NAME_RE.test(name)) {
      continue
    }

    if (entry.mode === 'ssh') {
      const ssh = normalizeSshConfig(entry)

      if (ssh) {
        if (entry.token && typeof entry.token === 'object') {
          ssh.token = entry.token
        }

        out[name] = ssh
      }

      continue
    }

    const cleaned: {
      mode: 'remote' | 'local' | 'cloud'
      url?: string
      authMode?: string
      token?: object
      headers?: object
      org?: string
      name?: string
      remoteProfile?: string
      savedSsh?: object
    } = {
      mode: modeIsRemoteLike(entry.mode) ? entry.mode : 'local'
    }

    if (cleaned.mode === 'local') {
      const savedSsh = normalizeSshConfig(entry.savedSsh)

      if (savedSsh) {
        cleaned.savedSsh = savedSsh
      }
    }

    const url = String(entry.url || '').trim()

    if (url) {
      cleaned.url = url
    }

    cleaned.authMode = normAuthMode(entry.authMode)

    if ((entry as any).token && typeof entry.token === 'object') {
      cleaned.token = entry.token
    }

    // A URL-remote/cloud per-profile override can map its Desktop routing label
    // onto the backend's profile namespace (same contract as SSH remoteProfile).
    const remoteProfileName = normalizeRemoteProfileName((entry as any).remoteProfile)

    if (remoteProfileName) {
      cleaned.remoteProfile = remoteProfileName.remoteProfile
    }

    const headers = normalizeRemoteHeaders((entry as any).headers)

    if (Object.keys(headers).length > 0) {
      cleaned.headers = headers
    }

    // Preserve the Hermes Cloud org tag on cloud-mode entries so Settings can
    // reopen into the same org for a per-profile cloud connection.
    if (cleaned.mode === 'cloud') {
      const cloudName = String(entry.name || '').trim()

      if (cloudName) {
        cleaned.name = cloudName
      }

      const org = String(entry.org || '').trim()

      if (org) {
        cleaned.org = org
      }
    }

    out[name] = cleaned
  }

  return out
}
