import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { sessionDotClassName } from '@/app/chat/session-status-dot'
import { ClarifyPendingForm } from '@/components/assistant-ui/clarify-tool'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { registerClarifyDock, sessionClarifyRequest } from '@/store/clarify'

/**
 * The "Needs you" section of the composer status stack: the session's pending
 * clarify question, docked directly above the input.
 *
 * Why here and not (only) inline in the transcript: the agent asks at the point
 * in the turn where it got stuck, which after a long tool run is many screens
 * above the fold. The transcript keeps a one-line marker at that point (so the
 * read-back order still makes sense); the live form with the choices, the
 * shortcuts, and Continue lives next to where the user types — the one spot
 * every session guarantees is on screen. Answering here resolves the same
 * request the inline card would (same store, same `clarify.respond`).
 *
 * The section registers itself as this session's dock, which is what tells the
 * inline card to collapse to its marker. Mount count, not a flag: a split
 * layout can host two composers for one session.
 */
export function NeedsYouSection({ sessionId }: { sessionId: string }) {
  const { t } = useI18n()
  const $request = useMemo(() => sessionClarifyRequest(sessionId), [sessionId])
  const request = useStore($request)

  useEffect(() => registerClarifyDock(sessionId), [sessionId])

  if (!request) {
    return null
  }

  const count = request.questions?.length ?? 1

  return (
    <section aria-label={t.statusStack.needsYou} data-slot="composer-needs-you">
      <div className="flex items-center gap-1.5 px-2 pt-1 pb-0.5 text-xs text-muted-foreground/92">
        <span aria-hidden className={cn(sessionDotClassName('needs-input'), 'shrink-0')} />
        <span className="min-w-0 truncate">
          {count > 1 ? t.statusStack.needsYouCount(count) : t.statusStack.needsYou}
        </span>
      </div>
      <div className="px-2.5 pt-1 pb-2">
        {/* Keyed by request so a follow-up question starts from a clean form —
            staged picks from the previous question must not carry over. */}
        <ClarifyPendingForm key={request.requestId} request={request} />
      </div>
    </section>
  )
}
