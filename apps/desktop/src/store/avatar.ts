import { atom, computed } from 'nanostores'

/**
 * Atom holding the current avatar data URL (or null if none is set).
 * Loaded once at app startup from disk (~/.hermes/avatar.png).
 * For per-profile support, this holds the IPC-loaded default; components should
 * read $profileAvatarDataUrl instead to get the profile-aware value.
 */
export const $avatarDataUrl = atom<string | null>(null)
export const $avatarLoading = atom<boolean>(false)

// ── Per-profile avatar support ─────────────────────────────────────────────
//
// Per-profile avatars are stored in localStorage under the key
// `hermes.avatar.<profile>` and fall back to the default IPC-loaded avatar
// when no per-profile override is set. The IPC avatar persists to
// ~/.hermes/avatar.png; per-profile avatars are renderer-only for now.
// A future main-process change could map these to
// ~/.hermes/profiles/<name>/avatar.png.
//
// $activeProfileName is wired by the app shell to the active gateway profile.
// Components import $profileAvatarDataUrl to get the right avatar per profile.

const AVATAR_PREFIX = 'hermes.avatar.'

/** The profile whose avatar we're currently displaying. */
export const $activeProfileName = atom<string>('default')

/**
 * Profile-aware avatar: checks localStorage for a per-profile override first,
 * then falls back to the default (IPC-loaded) avatar.
 */
export const $profileAvatarDataUrl = computed(
  [$avatarDataUrl, $activeProfileName],
  (defaultUrl, profile) => {
    try {
      const stored = localStorage.getItem(`${AVATAR_PREFIX}${profile}`)

      return stored ?? defaultUrl
    } catch {
      return defaultUrl
    }
  }
)

function readProfileAvatar(profile: string): string | null {
  try {
    return localStorage.getItem(`${AVATAR_PREFIX}${profile}`)
  } catch {
    return null
  }
}

function writeProfileAvatar(profile: string, dataUrl: string | null): void {
  try {
    if (dataUrl) {
      localStorage.setItem(`${AVATAR_PREFIX}${profile}`, dataUrl)
    } else {
      localStorage.removeItem(`${AVATAR_PREFIX}${profile}`)
    }
  } catch {
    // localStorage may be full or unavailable — ignore
  }
}

/**
 * Load the avatar from disk via the Electron IPC bridge.
 * Call once at app startup.
 */
export async function loadAvatar(): Promise<string | null> {
  $avatarLoading.set(true)

  try {
    const dataUrl = await window.hermesDesktop.avatar.get()
    $avatarDataUrl.set(dataUrl)

    return dataUrl
  } catch (error) {
    console.error('[avatar] Failed to load avatar:', error)
    $avatarDataUrl.set(null)

    return null
  } finally {
    $avatarLoading.set(false)
  }
}

/**
 * Set (upload) a new avatar from a data URL.
 * Persists to disk via IPC AND stores per-profile in localStorage.
 * When profile is provided, also saves the per-profile override.
 */
export async function setAvatar(
  dataUrl: string,
  profile?: string | null
): Promise<string | null> {
  try {
    const result = await window.hermesDesktop.avatar.set(dataUrl)
    $avatarDataUrl.set(result)

    // Also persist to localStorage for the specific profile
    const key = profile || $activeProfileName.get() || 'default'
    writeProfileAvatar(key, result)

    return result
  } catch (error) {
    console.error('[avatar] Failed to set avatar:', error)
    throw error
  }
}

/**
 * Remove the avatar.
 * - When profile is provided: clears ONLY that profile's localStorage override.
 *   Other profiles and the global IPC default are left untouched.
 * - When profile is omitted: clears the global IPC default AND ALL per-profile
 *   localStorage overrides, returning every profile to the gradient fallback.
 */
export async function resetAvatar(profile?: string | null): Promise<void> {
  try {
    if (profile) {
      // Only clear the per-profile override — fall back to default
      writeProfileAvatar(profile, null)
    } else {
      // Clear IPC default + all localStorage overrides for every profile
      await window.hermesDesktop.avatar.reset()
      $avatarDataUrl.set(null)

      const keysToRemove: string[] = []
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i)
        if (k?.startsWith(AVATAR_PREFIX)) {
          keysToRemove.push(k)
        }
      }
      for (const k of keysToRemove) {
        try {
          localStorage.removeItem(k)
        } catch {
          // ignore
        }
      }
    }
  } catch (error) {
    console.error('[avatar] Failed to reset avatar:', error)
    throw error
  }
}
