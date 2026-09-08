import type { Icon } from '@tabler/icons-react'
import { DynamicIcon, type IconName, iconNames } from 'lucide-react/dynamic'
import type * as React from 'react'

import { Codicon } from '@/components/ui/codicon'

export const LUCIDE_PROJECT_ICON_PREFIX = 'lucide:'
export const LUCIDE_PROJECT_ICON_NAMES = iconNames as readonly IconName[]

const lucideIconNames = new Set<string>(LUCIDE_PROJECT_ICON_NAMES)

export function lucideProjectIconValue(name: IconName): string {
  return `${LUCIDE_PROJECT_ICON_PREFIX}${name}`
}

export function projectLucideIconName(value: null | string | undefined): IconName | null {
  if (!value?.startsWith(LUCIDE_PROJECT_ICON_PREFIX)) {
    return null
  }

  const name = value.slice(LUCIDE_PROJECT_ICON_PREFIX.length)

  return lucideIconNames.has(name) ? (name as IconName) : null
}

interface ProjectIconProps {
  name?: null | string
  fallback?: string
  size?: number | string
  className?: string
  style?: React.CSSProperties
  'aria-label'?: string
}

/** Render persisted project icons while keeping legacy Codicon values valid. */
export function ProjectIcon({
  name,
  fallback = 'folder-library',
  size,
  className,
  style,
  'aria-label': ariaLabel
}: ProjectIconProps) {
  const lucideName = projectLucideIconName(name)

  if (lucideName) {
    return (
      <DynamicIcon
        aria-hidden={ariaLabel ? undefined : true}
        aria-label={ariaLabel}
        className={className}
        data-project-icon={lucideProjectIconValue(lucideName)}
        name={lucideName}
        size={size}
        style={style}
      />
    )
  }

  const codiconName = name?.startsWith(LUCIDE_PROJECT_ICON_PREFIX) ? fallback : name || fallback

  return <Codicon aria-label={ariaLabel} className={className} name={codiconName} size={size} style={style} />
}

/** Adapt a persisted project glyph to palette rows that expect a Tabler-shaped component. */
export function projectIconComponent(name: null | string | undefined, fallback = 'folder-library'): Icon {
  function ProjectIconComponent({
    className,
    size,
    'aria-label': ariaLabel
  }: {
    className?: string
    size?: number | string
    'aria-label'?: string
  }) {
    return (
      <ProjectIcon aria-label={ariaLabel} className={className} fallback={fallback} name={name} size={size ?? '1em'} />
    )
  }

  ProjectIconComponent.displayName = `ProjectIcon(${name || fallback})`

  return ProjectIconComponent as Icon
}
