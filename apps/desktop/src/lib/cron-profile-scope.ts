import { ALL_PROFILES, normalizeProfileKey } from '@/store/profile'

// The desktop sidebar scopes cron jobs like sessions: concrete profile views show
// only that profile, while the explicit All Profiles view asks the backend for
// its aggregate list. Empty/null falls back to the primary/default profile.
export function cronJobsProfileForScope(profileScope: null | string | undefined): string {
  const normalized = normalizeProfileKey(profileScope)

  return normalized === ALL_PROFILES ? 'all' : normalized
}
