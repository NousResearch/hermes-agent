import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

interface SessionStepNavProps {
  /** The currently active session, if the app is on a chat view. */
  activeId: null | string
  /** Ordered ids of the sessions the buttons step through (chronological). */
  ids: string[]
  /** Resume the stepped-to session (same as clicking its row). */
  onStep: (id: string) => void
}

/**
 * Up/down steppers for the flat sessions list. Move the active session one
 * row in either direction and resume it — a visible complement to the
 * Ctrl+Tab / Ctrl+PageUp-PageDown session switching keybinds (#53017), for
 * users who navigate the sidebar by mouse.
 *
 * Deliberately scoped to the flat Sessions view: in the Projects tree and the
 * Profiles browse view, "up/down" has no single honest axis, so the buttons
 * only render there if a caller passes ids for them.
 */
export function SessionStepNav({ activeId, ids, onStep }: SessionStepNavProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const index = activeId ? ids.indexOf(activeId) : -1
  const hasPrev = index > 0
  const hasNext = index >= 0 && index < ids.length - 1

  return (
    <div className="flex shrink-0 items-center gap-0.5">
      <Tip label={s.stepUp}>
        <Button
          aria-label={s.stepUp}
          className="text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-35"
          disabled={!hasPrev}
          onClick={() => onStep(ids[index - 1])}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Codicon name="chevron-up" size="0.75rem" />
        </Button>
      </Tip>
      <Tip label={s.stepDown}>
        <Button
          aria-label={s.stepDown}
          className="text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-35"
          disabled={!hasNext}
          onClick={() => onStep(ids[index + 1])}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Codicon name="chevron-down" size="0.75rem" />
        </Button>
      </Tip>
    </div>
  )
}
