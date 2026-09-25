import { useStore } from '@nanostores/react'

import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $profileColors, normalizeProfileKey } from '@/store/profile'

/** Owning-profile chip: the shared {@link ProfileGlyph}, resolved against the
 *  live profile colors and labelled with the full name. Identity, not status —
 *  session state dots keep their own semantics (#66003).
 *
 *  `showName` is the session-tab lead: glyph + the profile name as secondary
 *  text set to the tab label's metrics, so who owns the session reads at the
 *  START of the tab without competing with its title. */
export function ProfileTag({
  className,
  profile,
  showName = false
}: {
  className?: string
  profile: null | string | undefined
  showName?: boolean
}) {
  const { t } = useI18n()
  const colors = useStore($profileColors)
  const key = normalizeProfileKey(profile)
  const label = t.sidebar.row.ownedByProfile(key)

  const glyph = (
    <ProfileGlyph
      aria-label={label}
      className={className}
      color={resolveProfileColor(key, colors)}
      isDefault={key === 'default'}
      name={key}
      role="img"
    />
  )

  return (
    <Tip label={label}>
      {showName ? (
        // Tab lead only: the status dot follows, so keep a hair of room after
        // the name, and cap its width — a long profile name must not eat the
        // tab's title (PaneTab caps the whole tab at max-w-48).
        <span className="mr-1 flex shrink-0 items-center gap-1">
          {glyph}
          <span className="block min-w-0 max-w-20 truncate text-[9px] font-medium tracking-wide uppercase text-(--ui-text-secondary)">
            {key}
          </span>
        </span>
      ) : (
        glyph
      )}
    </Tip>
  )
}
