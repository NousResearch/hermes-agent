import type { StaleAuxAssignment } from '@/hermes'
import { getApiRequestProfile, profileScopeKey } from '@/hermes'
import { readKey, writeKey } from '@/lib/storage'

// Acknowledged stale-aux banner. The warning exists to catch a forgotten pin
// silently billing a dead provider; a user with a deliberate cross-provider
// pin (e.g. vision on the only provider that offers it, #66740) would
// otherwise see it on every settings visit with no way to acknowledge it.
// A dismissal is bound to a fingerprint of the exact configuration it
// acknowledged — main provider plus every slot's task/provider/model/endpoint
// — so any edit, a main switch, or a repointed base_url changes the
// fingerprint and re-arms the banner; the silent-credit-burn protection is
// never lost.
const DISMISSED_STALE_AUX_KEY_BASE = 'hermes.desktop.staleAuxDismissal.v1'

function dismissalKey(profile: null | string | undefined): string {
  const scope = profile ?? getApiRequestProfile() ?? undefined

  return `${DISMISSED_STALE_AUX_KEY_BASE}.profile.${encodeURIComponent(profileScopeKey(scope))}`
}

// Trailing-slash-insensitive so `http://x/v1` and `http://x/v1/` (the same
// billing surface) share a fingerprint; empty stays empty.
function normalizedEndpoint(url: null | string | undefined): string {
  const trimmed = (url ?? '').trim()

  return trimmed.replace(/\/+$/, '')
}

export function staleAuxFingerprint(mainProvider: string, slots: readonly StaleAuxAssignment[]): string {
  const pins = slots
    .map(slot => `${slot.task}:${slot.provider}:${slot.model}:${normalizedEndpoint(slot.base_url)}`)
    .sort()
    .join('|')

  return `${mainProvider.trim().toLowerCase()}#${pins}`
}

export function readStaleAuxDismissal(profile: null | string | undefined): null | string {
  return readKey(dismissalKey(profile))
}

export function dismissStaleAux(
  profile: null | string | undefined,
  mainProvider: string,
  slots: readonly StaleAuxAssignment[]
) {
  writeKey(dismissalKey(profile), staleAuxFingerprint(mainProvider, slots))
}
