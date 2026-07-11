import type { MutableRefObject } from 'react'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import type { ClarifyRequest } from '@/store/clarify'
import { appendSessionDraft, takeSessionDraft } from '@/store/composer'

function formatRecoveredClarifyDraft(request: ClarifyRequest): string {
  const answer = (request.answerDraft || request.selectedChoice || '').trim()

  if (!answer) {
    return ''
  }

  const question = request.question.trim()

  return question ? `Unsent answer to Hermes question:\n${question}\n\n${answer}` : answer
}

function ensureSessionDraftContains(sessionId: string, text: string) {
  const current = takeSessionDraft(sessionId).text

  if (current !== text && !current.includes(`\n\n${text}`)) {
    appendSessionDraft(sessionId, text)
  }
}

export function recoverClarifyDrafts(
  requests: ClarifyRequest[],
  activeSessionIdRef: MutableRefObject<string | null>
) {
  const visibleDrafts: { sessionId: string; text: string }[] = []

  for (const request of requests) {
    const text = formatRecoveredClarifyDraft(request)

    if (!text) {
      continue
    }

    appendSessionDraft(request.sessionId, text)

    if (request.sessionId && request.sessionId === activeSessionIdRef.current) {
      visibleDrafts.push({ sessionId: request.sessionId, text })
    }
  }

  if (visibleDrafts.length === 0 || typeof window === 'undefined') {
    return
  }

  window.setTimeout(() => {
    const visibleText: string[] = []

    for (const draft of visibleDrafts) {
      // A session switch can flush the old live composer over the stash after
      // recovery starts. Re-assert the origin-scoped copy before considering
      // delivery to whichever composer is visible now.
      ensureSessionDraftContains(draft.sessionId, draft.text)

      if (activeSessionIdRef.current === draft.sessionId) {
        visibleText.push(draft.text)
      }
    }

    if (visibleText.length > 0) {
      requestComposerInsert(visibleText.join('\n\n'), { mode: 'block', target: 'main' })
    }
  }, 100)
}
