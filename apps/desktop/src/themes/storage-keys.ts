// Legacy global appearance (pre per-profile themes). Still the inheritance
// fallback for profiles without their own assignment.
export const SKIN_KEY = 'hermes-desktop-theme-v2'
export const MODE_KEY = 'hermes-desktop-mode-v1'

// Per-profile skin + light/dark mode assignments: { [profileKey]: value }.
export const PROFILE_SKINS_KEY = 'hermes-desktop-profile-themes-v1'
export const PROFILE_MODES_KEY = 'hermes-desktop-profile-modes-v1'

// User-installed themes shared by the boot-time paint and renderer registry.
export const USER_THEMES_KEY = 'hermes-desktop-user-themes-v1'

// Last active profile, used by the boot-time paint before gateway startup.
export const LAST_PROFILE_KEY = 'hermes-desktop-active-profile-v1'
