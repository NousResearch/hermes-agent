import { useStore } from '@nanostores/react'

import { profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $profileAvatars, $profileColors, normalizeProfileKey } from '@/store/profile'

// A profile's face wherever it appears (rail squares, manage list/detail,
// all-profiles group headers). Renders the uploaded picture when one is cached
// in $profileAvatars, else the same colored-initial fallback the rail has always
// drawn — so profiles with no picture look exactly as before.
//
// Pass sizing/shape via className (e.g. "size-5 rounded-[3px]"). The color
// system is untouched: the initial fallback uses the profile's resolved hue.
export function ProfileAvatar({ className, name }: { className?: string; name: string }) {
  const avatars = useStore($profileAvatars)
  const colors = useStore($profileColors)
  const key = normalizeProfileKey(name)
  const dataUrl = avatars[key]

  const base = cn('inline-grid shrink-0 place-items-center overflow-hidden rounded-md', className)

  if (dataUrl) {
    return (
      <span aria-hidden="true" className={base}>
        <img alt="" className="size-full object-cover" draggable={false} src={dataUrl} />
      </span>
    )
  }

  const color = resolveProfileColor(name, colors)
  const hue = color ?? 'var(--ui-text-quaternary)'

  const initial =
    name
      .replace(/[^a-z0-9]/gi, '')
      .charAt(0)
      .toUpperCase() || '?'

  return (
    <span
      aria-hidden="true"
      className={cn(base, 'font-semibold uppercase leading-none')}
      style={{ backgroundColor: profileColorSoft(hue, 22), color: color ?? undefined }}
    >
      {initial}
    </span>
  )
}
