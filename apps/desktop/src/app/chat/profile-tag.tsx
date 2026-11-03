import { useStore } from '@nanostores/react'

import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $profileColors, normalizeProfileKey } from '@/store/profile'

/** Owning-profile chip: the shared {@link ProfileGlyph}, resolved against the
 *  live profile colors and labelled with the full name. Identity, not status —
 *  session state dots keep their own semantics (#66003). */
export function ProfileTag({ className, profile }: { className?: string; profile: null | string | undefined }) {
  const { t } = useI18n()
  const colors = useStore($profileColors)
  const key = normalizeProfileKey(profile)
  const label = t.sidebar.row.ownedByProfile(key)

  return (
    <Tip label={label}>
      <ProfileGlyph
        aria-label={label}
        className={className}
        color={resolveProfileColor(key, colors)}
        isDefault={key === 'default'}
        name={key}
        role="img"
      />
    </Tip>
  )
}
