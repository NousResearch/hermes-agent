/**
 * Recovery for clarify answers the user typed but never sent (#58783).
 *
 * A clarify card parks its in-progress answers in the store (keyed by the
 * request), so remounts and reconnects no longer lose them. But when the turn
 * ends WITHOUT a confirmed submit — expiry, Stop, an "Operation interrupted"
 * unwind, or an error frame — the parked request is cleared, and with it the
 * answer the user spent real time typing. That destruction is silent and
 * irreversible; this module's single consumer contract: clear the request,
 * salvage whatever staged answer existed into the session's composer draft
 * (appended — never replacing text the user may have started typing), and
 * surface it in the ACTIVE composer only when the cleared request's session
 * is the one on screen. Answers are NEVER auto-sent: the agent's question may
 * be stale by the time the user looks, so the draft is an offer, not a send.
 */
import type { MutableRefObject } from 'react'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import type { ClarifyRequest } from '@/store/clarify'
import { appendSessionDraft } from '@/store/composer'

const formatStagedAnswer = (request: ClarifyRequest): string => {
  const answers: string[] = []

  for (const question of request.questions) {
    const stage = request.stagedAnswers?.[question.qid]
    const answer = (stage?.draft ?? '').trim() || (stage?.choices.length ? stage.choices.join(', ') : '')

    if (!answer) {
      continue
    }

    answers.push(question.question ? `${question.question}: ${answer}` : answer)
  }

  return answers.join('\n')
}

export function recoverClarifyDrafts(
  requests: ClarifyRequest[],
  activeSessionIdRef: MutableRefObject<string | null>
): void {
  const activeTexts: string[] = []

  for (const request of requests) {
    const text = formatStagedAnswer(request)

    if (!text) {
      continue
    }

    const formatted = `Unsent answer to Hermes question:\n${text}`

    // The session-scoped stash is the durable copy: recoverable even after a
    // restart, and never clobbers existing draft text (append, not replace).
    if (request.sessionId) {
      appendSessionDraft(request.sessionId, formatted)
    }

    if (request.sessionId === activeSessionIdRef.current) {
      activeTexts.push(formatted)
    }
  }

  if (activeTexts.length > 0 && typeof window !== 'undefined') {
    // Deferred like the plugin-SDK insert path: the transcript is mid-update
    // when the clear fires, and the composer's insert handler must not run
    // inside that reconciliation.
    window.setTimeout(() => {
      if (activeTexts.length > 0) {
        requestComposerInsert(activeTexts.join('\n\n'), { mode: 'block', target: 'main' })
      }
    }, 100)
  }
}
