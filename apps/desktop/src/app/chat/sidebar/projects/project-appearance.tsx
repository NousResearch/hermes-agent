import { useMemo, useState } from 'react'

import { useDebounced } from '@/app/hooks/use-debounced'
import { ColorSwatches } from '@/components/ui/color-swatches'
import { SearchField } from '@/components/ui/search-field'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { PROFILE_SWATCHES } from '@/lib/profile-color'
import { cn } from '@/lib/utils'

import { LUCIDE_PROJECT_ICON_NAMES, lucideProjectIconValue, ProjectIcon, projectLucideIconName } from './project-icon'

const LUCIDE_RESULTS_LIMIT = 36

// Keep the familiar curated Codicons as the zero-query view. Searching switches
// to Lucide's complete dynamic catalogue; a selected Lucide icon is pinned here
// too, so clearing the search never makes the current choice disappear.
export const PROJECT_ICONS = [
  'folder-library',
  'repo',
  'rocket',
  'beaker',
  'flame',
  'star-full',
  'heart',
  'zap',
  'target',
  'lightbulb',
  'tools',
  'device-desktop',
  'device-mobile',
  'terminal',
  'dashboard',
  'globe',
  'broadcast',
  'cloud',
  'database',
  'package',
  'book',
  'organization',
  'bug',
  'shield',
  'key',
  'gift',
  'telescope',
  'home'
]

interface ProjectAppearancePickerProps {
  color: null | string
  icon: null | string
  noColorLabel: string
  onColor: (color: null | string) => void
  onIcon: (icon: null | string) => void
}

/** Color swatches + icon grid for a project's appearance — one component so the
 *  kebab popover and the right-click submenu render an identical picker. */
export function ProjectAppearancePicker({ color, icon, noColorLabel, onColor, onIcon }: ProjectAppearancePickerProps) {
  const { t } = useI18n()
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebounced(query, 120)
  const selectedLucideName = projectLucideIconName(icon)

  const displayedIcons = useMemo(() => {
    const terms = debouncedQuery
      .trim()
      .toLowerCase()
      .split(/[\s_-]+/)
      .filter(Boolean)

    if (terms.length) {
      return LUCIDE_PROJECT_ICON_NAMES.filter(name => terms.every(term => name.includes(term)))
        .slice(0, LUCIDE_RESULTS_LIMIT)
        .map(lucideProjectIconValue)
    }

    return selectedLucideName ? [lucideProjectIconValue(selectedLucideName), ...PROJECT_ICONS] : PROJECT_ICONS
  }, [debouncedQuery, selectedLucideName])

  return (
    <>
      <ColorSwatches
        clearIcon="circle-slash"
        clearLabel={noColorLabel}
        onChange={onColor}
        swatches={PROFILE_SWATCHES}
        value={color ?? null}
      />
      <SearchField
        aria-label={t.titlebar.search}
        containerClassName="mt-2 w-full"
        onChange={setQuery}
        placeholder={t.titlebar.search}
        value={query}
      />
      {/* Debouncing plus a six-row cap avoids an import storm while typing;
          every Lucide name remains reachable as soon as the query narrows. */}
      <div className="mt-1 grid max-h-48 grid-cols-6 gap-1.5 overflow-y-auto pr-0.5" data-slot="project-icon-results">
        {displayedIcons.map(value => {
          const name = projectLucideIconName(value) ?? value

          return (
            <Tip key={value} label={name}>
              <button
                aria-label={name}
                className={cn(
                  'grid aspect-square place-items-center rounded-md text-(--ui-text-tertiary) transition hover:bg-(--ui-control-hover-background)',
                  icon === value && 'bg-(--ui-control-active-background) text-foreground'
                )}
                onClick={() => onIcon(icon === value ? null : value)}
                style={icon === value && color ? { color } : undefined}
                type="button"
              >
                <ProjectIcon name={value} size="0.8125rem" />
              </button>
            </Tip>
          )
        })}
      </div>
    </>
  )
}
