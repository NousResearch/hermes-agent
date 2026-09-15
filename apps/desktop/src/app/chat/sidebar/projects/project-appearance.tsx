import { Codicon } from '@/components/ui/codicon'
import { ColorSwatches } from '@/components/ui/color-swatches'
import { Tip } from '@/components/ui/tooltip'
import { PROFILE_SWATCHES } from '@/lib/profile-color'
import { cn } from '@/lib/utils'

// Curated codicons for a project glyph (tinted by the chosen color). Shared by
// the kebab's Appearance popover and the right-click menu's Appearance submenu
// so both offer the same picker.
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
  return (
    <>
      <ColorSwatches
        clearIcon="circle-slash"
        clearLabel={noColorLabel}
        onChange={onColor}
        swatches={PROFILE_SWATCHES}
        value={color ?? null}
      />
      {/* Same 6 columns + gap as the swatch grid so the picker keeps the
          profile picker's width (icons flex to fill, not fixed-width). */}
      <div className="mt-2 grid grid-cols-6 gap-1.5">
        {PROJECT_ICONS.map(name => (
          <Tip key={name} label={name}>
            <button
              aria-label={name}
              className={cn(
                'grid aspect-square place-items-center rounded-md text-(--ui-text-tertiary) transition hover:bg-(--ui-control-hover-background)',
                icon === name && 'bg-(--ui-control-active-background) text-foreground'
              )}
              onClick={() => onIcon(icon === name ? null : name)}
              style={icon === name && color ? { color } : undefined}
              type="button"
            >
              <Codicon name={name} size="0.8125rem" />
            </button>
          </Tip>
        ))}
      </div>
    </>
  )
}
