import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

/**
 * The full-body Hermes character that replaces the legacy pixel pet in the
 * main Desktop window. Quick Entry runs in a separate renderer, so the
 * selection is persisted and mirrored through the browser `storage` event.
 */
export const DESKTOP_PET_AGENT_PROFILE_KEY = 'hermes.desktop.pet-agent-profile.v1'

export function normalizeDesktopPetAgentProfile(profile?: null | string): string {
  const normalized = profile?.trim().toLowerCase()

  return normalized && /^[a-z0-9_-]+$/.test(normalized) ? normalized : 'default'
}

function storedDesktopPetAgentProfile(): string {
  return normalizeDesktopPetAgentProfile(storedString(DESKTOP_PET_AGENT_PROFILE_KEY))
}

export const $desktopPetAgentProfile = atom(storedDesktopPetAgentProfile())

export function setDesktopPetAgentProfile(profile?: null | string): void {
  const normalized = normalizeDesktopPetAgentProfile(profile)

  $desktopPetAgentProfile.set(normalized)
  persistString(DESKTOP_PET_AGENT_PROFILE_KEY, normalized)
}

/** Keep the main renderer in sync when Quick Entry changes the character. */
export function listenForDesktopPetAgentProfile(): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === DESKTOP_PET_AGENT_PROFILE_KEY) {
      $desktopPetAgentProfile.set(normalizeDesktopPetAgentProfile(event.newValue))
    }
  }

  window.addEventListener('storage', onStorage)

  return () => window.removeEventListener('storage', onStorage)
}

