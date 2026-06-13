import { useStore } from '@nanostores/react'
import { useRef } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $threadJumpButtonVisible, requestScrollToBottom } from '@/store/thread-scroll'

/**
 * Floating "jump to bottom" control. Sits centered just above the composer,
 * clearing the out-of-flow status stack via the same measured-height CSS vars
 * the thread's bottom clearance uses (`--composer-measured-height` +
 * `--status-stack-measured-height`), so it never overlaps the queue / subagent
 * / background cards. Visible only while the user has scrolled meaningfully
 * away from the bottom; clicking re-arms sticky-bottom and pins the viewport.
 *
 * Enter/exit motion lives in styles.css under `.thread-jump-button` — a
 * directional scale (contract in from 1.1, contract out to 0.9) keyed off
 * `data-state`. `idle` (never-shown) stays silent so it can't flash on mount;
 * `in`/`out` only swap once it has actually appeared.
 */
export function ScrollToBottomButton() {
  const { t } = useI18n()
  const visible = useStore($threadJumpButtonVisible)
  const hasShownRef = useRef(false)

  if (visible) {
    hasShownRef.current = true
  }

  const state = visible ? 'in' : hasShownRef.current ? 'out' : 'idle'

  return (
    <button
      aria-hidden={!visible}
      aria-label={t.assistant.thread.scrollToBottom}
      className={cn(
        'thread-jump-button absolute left-1/2 z-20 grid size-8 place-items-center rounded-full',
        'border border-border/65 bg-(--composer-fill) text-muted-foreground hover:text-foreground',
        'backdrop-blur-[0.75rem] [-webkit-backdrop-filter:blur(0.75rem)]',
        !visible && 'pointer-events-none'
      )}
      data-state={state}
      onClick={() => {
        triggerHaptic('selection')
        requestScrollToBottom()
      }}
      style={{
        bottom: 'calc(var(--composer-measured-height) + var(--status-stack-measured-height) + 0.625rem)'
      }}
      tabIndex={visible ? 0 : -1}
      type="button"
    >
      <Codicon name="arrow-down" size="1rem" />
    </button>
  )
}
