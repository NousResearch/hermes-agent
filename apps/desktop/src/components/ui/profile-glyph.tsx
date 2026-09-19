import { profileColorSoft } from '@/lib/profile-color'
import { cn } from '@/lib/utils'

import { Codicon } from './codicon'

/** A profile's mark, in one place: the default profile is the `home` icon (it
 *  has no color of its own and an initial would read as just another named
 *  profile); every other profile is a soft tint of its color carrying its
 *  initial. Presentational — callers resolve the color from `$profileColors`. */
export function ProfileGlyph({
  className,
  color,
  isDefault,
  name,
  ...props
}: Omit<React.ComponentProps<'span'>, 'color'> & {
  color: null | string
  isDefault: boolean
  name: string
}) {
  if (isDefault) {
    return (
      <span className={cn('grid size-4 shrink-0 place-items-center', className)} {...props}>
        <Codicon className="text-(--ui-text-quaternary)" name="home" size="0.75rem" />
      </span>
    )
  }

  const initial = name.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'

  return (
    <span
      className={cn(
        'grid size-4 shrink-0 place-items-center rounded-[3px] text-[0.5rem] font-semibold uppercase leading-none',
        className
      )}
      style={{ backgroundColor: profileColorSoft(color ?? 'var(--ui-text-quaternary)', 22), color: color ?? undefined }}
      {...props}
    >
      {initial}
    </span>
  )
}
