import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { AppWindow } from '@/lib/icons'
import { $dockedWindow, undockWindow } from '@/store/dock'

/**
 * Shown while Hermes is docked beside an app (the app is tiled on the left,
 * Hermes snapped to the right). Names the controlled app and offers Undock,
 * which restores Hermes's previous geometry.
 */
export function DockBanner() {
  const { t } = useI18n()
  const c = t.composer
  const docked = useStore($dockedWindow)

  if (!docked) {
    return null
  }

  return (
    <div className="mx-1 mb-1 flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/10 px-2.5 py-1.5 text-[0.72rem]">
      <AppWindow className="size-3.5 shrink-0 text-primary" />
      <span className="min-w-0 flex-1 truncate font-medium text-foreground/90">{c.dockControlling(docked)}</span>
      <button
        className="shrink-0 rounded-md border border-border/60 px-2 py-0.5 text-[0.68rem] font-medium text-foreground/80 transition-colors hover:bg-accent hover:text-foreground"
        onClick={() => void undockWindow()}
        type="button"
      >
        {c.dockUndock}
      </button>
    </div>
  )
}
