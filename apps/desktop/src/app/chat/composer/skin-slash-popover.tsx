import { desktopSkinSlashCompletions } from '@/lib/desktop-slash-commands'
import { triggerHaptic } from '@/lib/haptics'
import { useTheme } from '@/themes/context'

import { COMPLETION_DRAWER_CLASS, COMPLETION_DRAWER_ROW_CLASS, CompletionDrawerEmpty } from './completion-drawer'

interface SkinSlashPopoverProps {
  draft: string
  onSelect: (command: string) => void
}

export function SkinSlashPopover({ draft, onSelect }: SkinSlashPopoverProps) {
  const { availableThemes, themeName } = useTheme()
  const match = draft.match(/^\/skin\s+(\S*)$/i)

  if (!match) {
    return null
  }

  const items = desktopSkinSlashCompletions(availableThemes, themeName, match[1] ?? '')

  return (
    <div
      aria-label="Desktop theme suggestions"
      className={COMPLETION_DRAWER_CLASS}
      data-slot="composer-skin-completion-drawer"
      data-state="open"
      role="listbox"
    >
      <div className="grid gap-0.5 pt-0.5">
        {items.length === 0 ? (
          <CompletionDrawerEmpty title="No matching themes.">
            Try <span className="font-mono text-foreground/80">/skin list</span>.
          </CompletionDrawerEmpty>
        ) : (
          items.map(item => (
            <button
              className={COMPLETION_DRAWER_ROW_CLASS}
              key={item.text}
              onClick={() => {
                triggerHaptic('selection')
                onSelect(item.text)
              }}
              onMouseDown={event => event.preventDefault()}
              role="option"
              type="button"
            >
              <span className="shrink-0 font-mono font-medium leading-5 text-foreground">{item.display}</span>
              <span className="min-w-0 truncate leading-5 text-muted-foreground/80">{item.meta}</span>
            </button>
          ))
        )}
      </div>
    </div>
  )
}
