import { memo, useState } from 'react'

import { composerFloatingPill } from '@/components/chat/composer-dock'
import { Codicon } from '@/components/ui/codicon'
import { useSessionSlice } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $composerActionsBySession, type ComposerAction } from '@/store/composer-actions'
import { notifyError } from '@/store/notifications'

/**
 * Floating pill — the shared treatment for a control that sits over the
 * composer (`composerFloatingPill`), plus this strip's own width cap and
 * disabled state.
 *
 * NEVER `pointer-events-none`, not even when disabled. The pop-out drag region
 * is an `absolute` sibling behind these pills, so a pill that stops taking
 * pointer events hands the hit test straight to it — and a dead-looking badge
 * becomes a grab handle that floats the composer.
 */
const PILL = cn(
  composerFloatingPill,
  'max-w-56',
  'disabled:cursor-default disabled:opacity-50 disabled:hover:bg-(--composer-fill)',
  'focus-visible:bg-(--chrome-action-hover) focus-visible:text-foreground focus-visible:outline-none focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50'
)

/**
 * The micro-action pills. Layout-free on purpose — the composer owns the strip
 * (`composerFloatingStrip`), this owns only the pills, so the strip above the
 * surface and the `composer.underside` strip below it can't drift apart.
 */
export const ActionBadges = memo(function ActionBadges({ sessionId }: { sessionId: null | string }) {
  const actions = useSessionSlice($composerActionsBySession, sessionId)
  // A pill can kick off async work (a gateway call, a submit). Track which one
  // is in flight so it can spin and lock instead of double-firing.
  const [runningId, setRunningId] = useState<null | string>(null)

  const run = async (action: ComposerAction) => {
    if (runningId || !sessionId) {
      return
    }

    setRunningId(action.id)

    try {
      await action.run(sessionId)
    } catch (error) {
      notifyError(error, action.label)
    } finally {
      setRunningId(null)
    }
  }

  return actions.map(action => {
    const running = runningId === action.id
    const glyph = running ? 'loading' : action.icon

    return (
      <button
        className={PILL}
        disabled={action.disabled || Boolean(runningId)}
        key={action.id}
        onClick={() => void run(action)}
        type="button"
      >
        {glyph && <Codicon className="shrink-0 opacity-70" name={glyph} size="0.75rem" spinning={running} />}
        <span className="truncate">{action.label}</span>
      </button>
    )
  })
})
