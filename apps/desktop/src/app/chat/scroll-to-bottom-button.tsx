import { useStore } from '@nanostores/react'
import { useReducedMotion } from 'motion/react'
import { useRef } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { AnimatedInt } from '@/components/ui/diff-count'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $approvalRequest } from '@/store/prompts'
import { $threadJumpButtonVisible, $threadMessagesBelow, requestScrollToBottom } from '@/store/thread-scroll'

/**
 * Floating "jump to bottom" control. Sits centered just above the composer,
 * clearing the out-of-flow status stack via the same measured-height CSS vars
 * the thread's bottom clearance uses (`--composer-measured-height`, which
 * covers the whole dock), so it never overlaps the queue / subagent
 * / background cards. Visible only while the user has scrolled meaningfully
 * away from the bottom, with an animated count of messages below the viewport.
 * Clicking re-arms sticky-bottom and pins the viewport.
 *
 * When the turn is BLOCKED on an approval, this same control morphs into an
 * "Approval needed" pill — the only response surface is the inline Run/Reject
 * bar on the parked tool row, which is always the bottom-most content, so the
 * existing scroll-to-bottom action lands the user right on it. One control, no
 * collision, no second scroll path (native scrollIntoView would scroll
 * overflow:hidden ancestors that can't scroll back and wreck the layout).
 *
 * Enter/exit motion lives in styles.css under `.thread-jump-button` — a
 * directional scale (contract in from 1.1, contract out to 0.9) keyed off
 * `data-state`. `idle` (never-shown) stays silent so it can't flash on mount;
 * `in`/`out` only swap once it has actually appeared.
 */
export function ScrollToBottomButton({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const visible = useStore($threadJumpButtonVisible)
  const count = useStore($threadMessagesBelow)
  const reducedMotion = useReducedMotion()
  const request = useStore($approvalRequest)
  // Scrolled away while an approval is pending → the inline Run/Reject bar is
  // below the fold. Relabel so the user knows the session needs them, not just
  // that there's more to read.
  const approval = visible && Boolean(request)
  const hasShownRef = useRef(false)

  if (visible) {
    hasShownRef.current = true
  }

  const state = visible ? 'in' : hasShownRef.current ? 'out' : 'idle'
  const countLabel = t.sidebar.messageCount(count)
  const [beforeCount, afterCount] = countLabel.split(String(count))

  const label = approval ? t.assistant.approval.jumpToApproval : `${t.assistant.thread.scrollToBottom} · ${countLabel}`

  return (
    <button
      aria-hidden={!visible}
      aria-label={label}
      className={cn(
        'thread-jump-button absolute left-1/2 z-20 flex h-8 items-center gap-1.5 rounded-full border bg-(--composer-fill) px-3 text-xs font-medium backdrop-blur-[0.75rem] [-webkit-backdrop-filter:blur(0.75rem)]',
        approval
          ? 'border-primary/40 text-primary hover:bg-primary/10'
          : 'border-border/65 text-muted-foreground hover:text-foreground',
        !visible && 'pointer-events-none'
      )}
      data-state={state}
      onClick={() => {
        triggerHaptic('selection')
        requestScrollToBottom(sessionId)
      }}
      style={{
        bottom: 'calc(var(--composer-measured-height) + 1rem)'
      }}
      tabIndex={visible ? 0 : -1}
      type="button"
    >
      <Codicon name="arrow-down" size="0.875rem" />
      {approval ? (
        <span>{label}</span>
      ) : (
        <span aria-hidden className="whitespace-nowrap tabular-nums">
          {beforeCount}
          {!visible || reducedMotion ? count : <AnimatedInt key={sessionId} value={count} />}
          {afterCount}
        </span>
      )}
    </button>
  )
}
