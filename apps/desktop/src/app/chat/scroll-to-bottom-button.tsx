import { useStore } from '@nanostores/react'
import { useRef } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $approvalRequest } from '@/store/prompts'
import { $threadJumpState, $threadScrolledUp, requestThreadJump, type ThreadJumpTarget } from '@/store/thread-scroll'

/**
 * Floating thread navigation control. Sits centered just above the composer,
 * clearing the out-of-flow status stack via the same measured-height CSS vars
 * the thread's bottom clearance uses (`--composer-measured-height` +
 * `--status-stack-measured-height`), so it never overlaps the queue / subagent
 * / background cards.
 *
 * Default behavior still supports the old bottom jump. For long latest answers,
 * the thread viewport promotes the same pill to `Answer start` when the user is
 * below the start, then flips it to `Answer end` when the user is already near
 * the top of that answer. One control, no second sticky header.
 *
 * When the turn is BLOCKED on an approval, this same control morphs into an
 * "Approval needed" pill — the only response surface is the inline Run/Reject
 * bar on the parked tool row, which is always the bottom-most content, so the
 * bottom action lands the user right on it.
 *
 * Enter/exit motion lives in styles.css under `.thread-jump-button` — a
 * directional scale (contract in from 1.1, contract out to 0.9) keyed off
 * `data-state`. `idle` (never-shown) stays silent so it can't flash on mount;
 * `in`/`out` only swap once it has actually appeared.
 */
export function ScrollToBottomButton() {
  const { t } = useI18n()
  const jump = useStore($threadJumpState)
  const scrolledUp = useStore($threadScrolledUp)
  const request = useStore($approvalRequest)
  const answerJumpVisible = jump.visible
  const visible = scrolledUp || answerJumpVisible
  // Scrolled away while an approval is pending → the inline Run/Reject bar is
  // below the fold. Relabel so the user knows the session needs them, not just
  // that there's more to read.
  const approval = scrolledUp && Boolean(request)
  const target: ThreadJumpTarget = approval ? 'bottom' : answerJumpVisible ? jump.target : 'bottom'
  const hasShownRef = useRef(false)

  if (visible) {
    hasShownRef.current = true
  }

  const state = visible ? 'in' : hasShownRef.current ? 'out' : 'idle'

  const label = approval
    ? t.assistant.approval.jumpToApproval
    : target === 'answer-start'
      ? t.assistant.thread.answerStart
      : target === 'answer-end'
        ? t.assistant.thread.answerEnd
        : t.assistant.thread.scrollToBottom

  const showText = approval || target !== 'bottom'

  return (
    <button
      aria-hidden={!visible}
      aria-label={label}
      className={cn(
        'thread-jump-button absolute left-1/2 z-20 grid place-items-center backdrop-blur-[0.75rem] [-webkit-backdrop-filter:blur(0.75rem)]',
        showText
          ? cn(
              'h-8 grid-flow-col gap-1.5 rounded-full border bg-(--composer-fill) px-3 text-xs font-medium shadow-sm',
              approval
                ? 'border-primary/40 text-primary hover:bg-primary/10'
                : 'border-[color-mix(in_srgb,var(--dt-ring)_28%,var(--ui-stroke-secondary))] text-foreground/85 hover:bg-[color-mix(in_srgb,var(--dt-ring)_8%,var(--composer-fill))] hover:text-foreground'
            )
          : 'size-8 rounded-full border border-border/65 bg-(--composer-fill) text-muted-foreground hover:text-foreground',
        !visible && 'pointer-events-none'
      )}
      data-state={state}
      data-target={target}
      onClick={() => {
        triggerHaptic('selection')
        requestThreadJump(target)
      }}
      style={{
        bottom: 'calc(var(--composer-measured-height) + var(--status-stack-measured-height) + 0.625rem)'
      }}
      tabIndex={visible ? 0 : -1}
      type="button"
    >
      <Codicon name={target === 'answer-start' ? 'arrow-up' : 'arrow-down'} size={showText ? '0.875rem' : '1rem'} />
      {showText && <span>{label}</span>}
    </button>
  )
}
