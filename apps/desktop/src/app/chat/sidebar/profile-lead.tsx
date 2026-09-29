import { useStore } from '@nanostores/react'
import type * as React from 'react'

import { resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $profileColors, normalizeProfileKey } from '@/store/profile'

/** Longest profile name the lead spells out. Past this it elides — a long
 *  name would otherwise take the room the title needs. */
const PROFILE_LEAD_MAX_CHARS = 14

/** The name as the lead prints it: whole when short, elided past the cap. */
export function profileLeadLabel(name: string): string {
  return name.length > PROFILE_LEAD_MAX_CHARS ? `${name.slice(0, PROFILE_LEAD_MAX_CHARS - 1)}…` : name
}

/** The owning profile's name ahead of a one-line row title — `inbox ›` — for
 *  lists that mix profiles (the All-profiles view). Quiet by default: the same
 *  grey as the row's age. While the row is hovered or selected it takes the
 *  profile's own colour, the colour of that profile's rail square, so the
 *  mark points at the rail exactly when you engage with a row and stays out
 *  of the way while you scan. Text rather than a swatch: it reads without
 *  colour, and it lines up under the status dot whatever the ages to the
 *  right are doing. Callers skip it for the default profile — a mark on every
 *  row that says "the normal one" is noise. */
export function ProfileLead({ profile, selected }: { profile: null | string | undefined; selected: boolean }) {
  const colors = useStore($profileColors)
  const key = normalizeProfileKey(profile)
  const color = resolveProfileColor(key, colors)

  return (
    <span
      className={cn(
        'transition-colors duration-100',
        selected
          ? 'font-medium text-(--profile-lead-color)'
          : 'text-(--ui-text-quaternary) group-hover:text-(--profile-lead-color)'
      )}
      data-profile-lead={key}
      style={{ '--profile-lead-color': color ?? 'var(--ui-text-tertiary)' } as React.CSSProperties}
    >
      {profileLeadLabel(key)}
      <span aria-hidden="true" className="mx-1 opacity-60">
        ›
      </span>
    </span>
  )
}
