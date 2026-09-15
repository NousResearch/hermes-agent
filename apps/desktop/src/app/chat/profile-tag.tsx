import { useStore } from '@nanostores/react'

import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $profileColors, $profiles, normalizeProfileKey, profileLabel } from '@/store/profile'

/** Owning-profile chip: the shared {@link ProfileGlyph}, resolved against the
 *  live profile colors and labelled with the full name. Identity, not status —
 *  session state dots keep their own semantics (#66003). */
export function ProfileTag({ className, profile }: { className?: string; profile: null | string | undefined }) {
  const { t } = useI18n()
  const colors = useStore($profileColors)
  const profiles = useStore($profiles)
  const key = normalizeProfileKey(profile)
  const resolvedProfile = profiles.find(candidate => normalizeProfileKey(candidate.name) === key)
  const displayName = resolvedProfile ? profileLabel(resolvedProfile) : key
  const label = t.sidebar.row.ownedByProfile(displayName === key ? key : `${displayName} (${key})`)

  return (
    <Tip label={label}>
      <ProfileGlyph
        aria-label={label}
        className={className}
        color={resolveProfileColor(key, colors)}
        isDefault={key === 'default'}
        name={displayName}
        role="img"
      />
    </Tip>
  )
}
