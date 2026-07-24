import { useStore } from '@nanostores/react'

import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { displayEntityName } from '@/lib/display-name'
import { profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $profileColors, normalizeProfileKey } from '@/store/profile'

/** Owning-profile chip: soft profile-tint square with the initial, tooltip +
 *  accessible label carrying the full name. Same visual language as the
 *  profile rail; the default profile stays neutral. Identity, not status —
 *  session state dots keep their own semantics (#66003). */
export function ProfileTag({ className, profile }: { className?: string; profile: null | string | undefined }) {
  const { t } = useI18n()
  const colors = useStore($profileColors)
  const key = normalizeProfileKey(profile)
  const color = resolveProfileColor(key, colors)
  const hue = color ?? 'var(--ui-text-quaternary)'
  // Localized display for the reserved "default" profile; the glyph initial
  // below intentionally keeps the raw key (CJK display names have no a-z0-9
  // initial to extract).
  const label = t.sidebar.row.ownedByProfile(displayEntityName(key, t))

  return (
    <Tip label={label}>
      <span
        aria-label={label}
        className={cn(
          'grid size-4 shrink-0 place-items-center rounded-[3px] text-[0.5rem] font-semibold uppercase leading-none',
          className
        )}
        role="img"
        style={{ backgroundColor: profileColorSoft(hue, 22), color: color ?? undefined }}
      >
        {key.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
      </span>
    </Tip>
  )
}
