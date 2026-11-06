/**
 * The `composer.middleware` handler that refines an eligible typed send.
 * Approval-First: the review drawer opens and nothing is sent until the user approves,
 * edits, sends the original or cancels. Automatic: the refined prompt is sent
 * on its own, stopping for the user only when refinement fails or the
 * preservation or intent check flags anything (the review store decides).
 * The sending composer is held read-only meanwhile, so nothing typed can be
 * overwritten if the draft is restored. Contract with the composer (contrib.ts):
 * return the draft = send it, return a rewritten draft = send that, return
 * null = cancel (the composer restores the draft and its attachments to the
 * chat it came from). Registered `failClosed`, and it also catches its own
 * failures, so nothing unexpected can let a send through unreviewed.
 */

import type { ComposerDraft, ComposerMiddleware } from '@hermes/plugin-sdk'

import type { ReviewRequest } from './review-store'
import { requestFor } from './settings'
import type { RefinementSettings, RefineRequest, RefineResult, ReviewDecision } from './types'

export interface ScopeGuard {
  /** True while the chat this send belongs to is still the one on screen. */
  stillCurrent: () => boolean
  /** Call when the guard should stop listening. */
  release: () => void
}

export interface MiddlewareDeps {
  settings: () => RefinementSettings
  review: (request: ReviewRequest) => Promise<ReviewDecision>
  refine: (request: RefineRequest, draft: ComposerDraft, signal: AbortSignal, attempt: number) => Promise<RefineResult>
  /** Watch the send's chat; `onDrift` fires if the user switches away mid-review. */
  guardScope: (draft: ComposerDraft, onDrift: () => void) => ScopeGuard
  /** Make the sending composer read-only until the returned release is called. */
  holdComposer: (target: string) => () => void
  /** Recent conversation of that composer's chat (used only when the user opted in). */
  contextFor: (target: string) => string
  isSlashCommand: (text: string) => boolean
  cancelActiveReview: () => void
  notify: (event: 'scope-changed' | 'no-host' | 'busy' | 'error') => void
}

const wordCount = (text: string) => text.split(/\s+/).filter(Boolean).length

/** Only the user's own typed, non-command, non-trivial draft is reviewed. */
export function isEligibleSend(draft: ComposerDraft, settings: RefinementSettings, isSlash: (t: string) => boolean) {
  const text = draft.text.trim()

  return (
    settings.mode !== 'off' &&
    draft.origin?.kind === 'typed' &&
    text.length > 0 &&
    !isSlash(text) &&
    wordCount(text) >= settings.minWords
  )
}

export function attachmentLabels(draft: ComposerDraft): string[] {
  return (draft.attachments ?? [])
    .filter(Boolean)
    .map(attachment => attachment.label || '')
    .filter(Boolean)
}

export function createRefinementMiddleware(deps: MiddlewareDeps): ComposerMiddleware {
  const handler = async (draft: ComposerDraft): Promise<ComposerDraft | null> => {
    const settings = deps.settings()

    if (!isEligibleSend(draft, settings, deps.isSlashCommand)) {
      return draft
    }

    let guard: ScopeGuard | null = null
    let releaseHold: (() => void) | null = null

    try {
      const target = draft.origin?.target ?? 'main'
      releaseHold = deps.holdComposer(target)
      guard = deps.guardScope(draft, () => deps.cancelActiveReview())

      const labels = attachmentLabels(draft)

      const request = requestFor(settings, draft.text, labels, settings.useChatContext ? deps.contextFor(target) : '')

      const decision = await deps.review({
        target,
        // Automatic sends a clean refinement on its own; Approval-First always waits.
        autoApprove: settings.mode === 'auto',
        original: draft.text,
        context: request.context ?? '',
        attachmentLabels: labels,
        run: (signal, attempt) => deps.refine(request, draft, signal, attempt)
      })

      if (decision.kind === 'cancel') {
        if (decision.reason !== 'user') {
          deps.notify(decision.reason === 'unloaded' ? 'error' : decision.reason)
        }

        return null
      }

      // Whoever decided (the user, or the store on a clean refinement), the chat
      // must still be the one they pressed Enter in. Otherwise cancel: the composer restores the draft to
      // its own chat (core's drift check would also refuse the send).
      if (!guard.stillCurrent()) {
        deps.notify('scope-changed')

        return null
      }

      if (decision.kind === 'original') {
        return draft
      }

      const text = decision.text.trim()

      // Same attachments array, untouched: only the words change.
      return text ? { ...draft, text } : null
    } catch {
      deps.notify('error')

      return null
    } finally {
      guard?.release()
      releaseHold?.()
    }
  }

  return { failClosed: true, handler }
}
