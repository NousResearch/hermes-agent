import type { ProfileInfo } from '@/types/hermes'

/** A profile remembered globally is not necessarily on the selected gateway. */
export function startupProfileForConnection(
  active: string,
  remembered: string | undefined,
  profiles: Pick<ProfileInfo, 'name'>[]
): string | null {
  const names = new Set(profiles.map(profile => profile.name))

  if (names.has(active)) {
    return null
  }

  if (remembered && names.has(remembered)) {
    return remembered
  }

  if (names.has('default')) {
    return 'default'
  }

  return profiles[0]?.name ?? null
}
