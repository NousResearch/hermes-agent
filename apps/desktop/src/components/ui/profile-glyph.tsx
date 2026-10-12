import { profileColorSoft } from '@/lib/profile-color'
import { profileShortLabel } from '@/lib/profile-short-label'
import { cn } from '@/lib/utils'

import { Codicon } from './codicon'

/** The default profile uses a home mark or an initial; named profiles use
 *  tinted initials. Callers resolve the color and visible profile name. */
export function ProfileGlyph({
  className,
  color,
  defaultStyle = 'home',
  isDefault,
  name,
  ...props
}: Omit<React.ComponentProps<'span'>, 'color'> & {
  color: null | string
  defaultStyle?: 'home' | 'initial'
  isDefault: boolean
  name: string
}) {
  if (isDefault && defaultStyle === 'home') {
    return (
      <span className={cn('grid size-4 shrink-0 place-items-center', className)} {...props}>
        <Codicon className="text-(--ui-text-quaternary)" name="home" size="0.75rem" />
      </span>
    )
  }

  const initial = profileShortLabel(name)

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
