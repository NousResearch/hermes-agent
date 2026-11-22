import { useStore } from '@nanostores/react'
import { useReducedMotion } from 'motion/react'
import { useMemo, useRef } from 'react'

import { usePendingTargetState } from '@/components/assistant-ui/thread/pending-input-navigation'
import { useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import { Codicon } from '@/components/ui/codicon'
import { AnimatedInt } from '@/components/ui/diff-count'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { sessionApprovalRequest } from '@/store/prompts'
import {
  $threadJumpButtonVisibleBySession,
  $threadMessagesBelowBySession,
  requestScrollToBottom
} from '@/store/thread-scroll'

import { useComposerSurfaceId } from './composer/scope'

// Attribute-safe selector fragment. jsdom (vitest) does not ship `CSS.escape`.
const cssEscape = (value: string): string => {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value)
  }

  return value.replace(/[^a-zA-Z0-9_:-]/g, ch => `\\${ch}`)
}

// The pending-approval stack renders once per pane, tagged with the owning
// session so a split view can't jump one pane's arrow into a sibling's
// approval. `nearest` keeps this a minimal scroll within the transcript's own
// scroll container instead of an unqualified scrollIntoView, which would also
// nudge any overflow-hidden ancestor's programmatic scroll offset.
function findSessionApprovalStack(sessionId: string | null, root: ParentNode): HTMLElement | null {
  if (!sessionId) {
    return null
  }

  return root.querySelector<HTMLElement>(`[data-approval-stack][data-session-id="${cssEscape(sessionId)}"]`)
}

/**
 * Floating "jump to bottom" control. Sits centered just above the composer,
 * clearing the out-of-flow status stack via the same measured-height CSS vars
 * the thread's bottom clearance uses (`--composer-measured-height`, which
 * covers the whole dock), so it never overlaps the queue / subagent
 * / background cards. Visible only while the user has scrolled meaningfully
 * away from the bottom, with an animated count of messages below the viewport.
 * Clicking re-arms sticky-bottom and pins the viewport.
 *
 * A pending heading outside the viewport also lights the pill at the true
 * bottom. The transcript resolves the jump within its own scroll owner.
 *
 * Enter/exit motion lives in styles.css under `.thread-jump-button` — a
 * directional scale (contract in from 1.1, contract out to 0.9) keyed off
 * `data-state`. `idle` (never-shown) stays silent so it can't flash on mount;
 * `in`/`out` only swap once it has actually appeared.
 */
export function ScrollToBottomButton({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const surfaceId = useComposerSurfaceId()
  const scrollSessionId = sessionId ?? surfaceId
  const { isHistorical } = useTranscriptWindow()

  const scrollVisible = useStoreSelector($threadJumpButtonVisibleBySession, map =>
    Boolean(scrollSessionId && map[scrollSessionId])
  )

  const count = useStoreSelector($threadMessagesBelowBySession, map =>
    scrollSessionId ? (map[scrollSessionId] ?? 0) : 0
  )

  const { target, offscreen, above, jump } = usePendingTargetState()
  const visible = isHistorical || (target && offscreen !== null ? offscreen : scrollVisible)
  const reducedMotion = useReducedMotion()
  const request = useStore(useMemo(() => sessionApprovalRequest(sessionId), [sessionId]))
  // Scrolled away while an approval is pending → the inline Run/Reject bar is
  // below the fold. Relabel so the user knows the session needs them, not just
  // that there's more to read.
  const visibleApproval = target ? target.kind === 'approval' : Boolean(request)
  const pending = Boolean(target || visibleApproval)
  const hasShownRef = useRef(false)

  if (visible) {
    hasShownRef.current = true
  }

  const state = visible ? 'in' : hasShownRef.current ? 'out' : 'idle'
  const countLabel = count > 0 ? t.sidebar.messageCount(count) : ''
  const [beforeCount, afterCount] = countLabel ? countLabel.split(String(count)) : ['', '']

  const label = visibleApproval
    ? t.assistant.approval.jumpToApproval
    : pending
      ? t.sidebar.row.needsInput
      : countLabel
        ? `${t.assistant.thread.scrollToBottom} · ${countLabel}`
        : t.assistant.thread.scrollToBottom

  return (
    <button
      aria-hidden={!visible}
      aria-label={label}
      className={cn(
        'thread-jump-button absolute left-1/2 z-20 flex h-8 items-center gap-1.5 rounded-full border bg-(--composer-fill) px-3 text-xs font-medium backdrop-blur-[0.75rem] [-webkit-backdrop-filter:blur(0.75rem)]',
        pending
          ? 'border-primary/40 text-primary hover:bg-primary/10'
          : 'border-border/65 text-muted-foreground hover:text-foreground',
        !visible && 'pointer-events-none'
      )}
      data-state={state}
      onClick={event => {
        triggerHaptic('selection')

        if (target && jump()) {
          return
        }

        const root = event.currentTarget.closest('[data-chat-surface]') ?? document
        const approvalStack = visibleApproval ? findSessionApprovalStack(request?.sessionId ?? null, root) : null

        if (approvalStack) {
          approvalStack.scrollIntoView({ block: 'nearest' })

          return
        }

        requestScrollToBottom(scrollSessionId)
      }}
      style={{
        bottom: 'calc(var(--composer-measured-height) + 1rem)'
      }}
      tabIndex={visible ? 0 : -1}
      type="button"
    >
      <Codicon name={target && above ? 'arrow-up' : 'arrow-down'} size="0.875rem" />
      {pending || count <= 0 ? (
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
