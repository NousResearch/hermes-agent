import type {
  ProfileCreatePayload,
  ProfileDesktopOverlay,
  ProfileSetupCommand,
  ProfileSoul,
  ProfilesResponse
} from '@/types/hermes'

import { capabilityScoped, hermesApi, type ProfileScope, STARTUP_REQUEST_TIMEOUT_MS } from './client'

export function getProfiles(): Promise<ProfilesResponse> {
  return hermesApi<ProfilesResponse>({
    path: '/api/profiles',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function createProfile(body: ProfileCreatePayload): Promise<{ name: string; ok: boolean; path: string }> {
  return hermesApi<{ name: string; ok: boolean; path: string }>({
    path: '/api/profiles',
    method: 'POST',
    body
  })
}

// Explicit (connection, profile) pin for a profile that lives on a gateway
// other than the foreground one — the fleet profile rail edits a remote
// square's SOUL/name in place. capabilityScoped now forwards a `'local'` pin
// itself (it must, or a remote registry PRIMARY absorbs "This device" reads),
// so this is a plain alias kept for the call sites' self-documenting name.
function profileOwnerScoped(scope?: ProfileScope): { connectionId?: string; profile?: string } {
  return capabilityScoped(scope)
}

export function renameProfile(
  name: string,
  newName: string,
  scope?: ProfileScope
): Promise<{ name: string; ok: boolean; path: string }> {
  return hermesApi<{ name: string; ok: boolean; path: string }>({
    ...profileOwnerScoped(scope),
    path: `/api/profiles/${encodeURIComponent(name)}`,
    method: 'PATCH',
    body: { new_name: newName }
  })
}

export function deleteProfile(name: string, scope?: ProfileScope): Promise<{ ok: boolean; path: string }> {
  const normalized = name.trim()
  const scopedProfile = scope && typeof scope === 'object' ? scope.profile?.trim() : undefined

  if (!normalized) {
    return Promise.reject(new Error('Profile name required'))
  }

  if (normalized.toLowerCase() === 'default' || scopedProfile?.toLowerCase() === 'default') {
    return Promise.reject(new Error('The default profile cannot be deleted.'))
  }

  return hermesApi<{ ok: boolean; path: string }>({
    ...profileOwnerScoped(scope),
    path: `/api/profiles/${encodeURIComponent(normalized)}`,
    method: 'DELETE'
  })
}

export function getProfileSoul(name: string, scope?: ProfileScope): Promise<ProfileSoul> {
  return hermesApi<ProfileSoul>({
    ...profileOwnerScoped(scope),
    path: `/api/profiles/${encodeURIComponent(name)}/soul`
  })
}

export function updateProfileSoul(name: string, content: string, scope?: ProfileScope): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileOwnerScoped(scope),
    path: `/api/profiles/${encodeURIComponent(name)}/soul`,
    method: 'PUT',
    body: { content }
  })
}

export function getProfileSetupCommand(name: string): Promise<ProfileSetupCommand> {
  return hermesApi<ProfileSetupCommand>({
    path: `/api/profiles/${encodeURIComponent(name)}/setup-command`
  })
}

/** Export a profile to a shareable .tar.gz on the backend's filesystem.
 *  `extraFiles` stages extra root-level files (desktop.json — the appearance/
 *  interface overlay) into the archive alongside the profile's own artifacts. */
export function exportProfileArchive(
  name: string,
  opts: { extraFiles?: Record<string, string>; output?: string } = {}
): Promise<{ archive: string; ok: boolean }> {
  return hermesApi<{ archive: string; ok: boolean }>({
    path: `/api/profiles/${encodeURIComponent(name)}/export`,
    method: 'POST',
    body: { extra_files: opts.extraFiles ?? {}, output: opts.output ?? '' },
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

/** Import a profile .tar.gz as a new profile. Returns the bundled desktop
 *  appearance overlay too (when the archive carried one) so the caller can
 *  apply theme/layout without another round-trip. */
export function importProfileArchive(
  archive: string,
  name?: string
): Promise<{ desktop: null | ProfileDesktopOverlay; name: string; ok: boolean; path: string }> {
  return hermesApi<{ desktop: null | ProfileDesktopOverlay; name: string; ok: boolean; path: string }>({
    path: '/api/profiles/import',
    method: 'POST',
    body: { archive, name: name || null },
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}
