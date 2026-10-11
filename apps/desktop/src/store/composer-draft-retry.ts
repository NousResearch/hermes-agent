import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'

import { type ComposerAttachment, freshDraftScope } from './composer'

/** The identity of an uncertain send that was restored into a composer draft.
 *  Sending that exact draft again is the explicit retry (same `submission_id`,
 *  so the owner replays instead of running it twice); editing it, or typing
 *  equal text anew in any composer, is a new message with a new identity. */
export interface ComposerDraftRetry {
  id: string
  text: string
  attachmentIds: string[]
}

// One localStorage key per draft scope: every window of the origin shares this
// storage with last-writer-wins per KEY, so one window never drops another's.
const RETRY_PREFIX = 'hermes:composer-draft-retry:v1:'

const attachmentIds = (attachments: ComposerAttachment[]) => attachments.map(a => a.occurrenceId ?? a.id)

export function draftRetry(id: string, text: string, attachments: ComposerAttachment[]): ComposerDraftRetry {
  return { id, text, attachmentIds: attachmentIds(attachments) }
}

export function draftRetryTextMatches(retry: ComposerDraftRetry | undefined, text: string): boolean {
  return Boolean(retry && sanitizeComposerInput(retry.text).trim() === sanitizeComposerInput(text).trim())
}

/** `retry` while the draft is still exactly the restored send; otherwise undefined. */
export function matchingDraftRetry(
  retry: ComposerDraftRetry | undefined,
  text: string,
  attachments: ComposerAttachment[]
): ComposerDraftRetry | undefined {
  return retry &&
    draftRetryTextMatches(retry, text) &&
    JSON.stringify(retry.attachmentIds) === JSON.stringify(attachmentIds(attachments))
    ? retry
    : undefined
}

/** The storage key of a draft scope, the same resolution the draft stash uses. */
const scopeKey = (scope: string | null | undefined) => scope?.trim() || freshDraftScope()

export function readDraftRetry(scope: string | null | undefined): ComposerDraftRetry | undefined {
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(RETRY_PREFIX + scopeKey(scope)) || 'null')
    const retry = parsed as ComposerDraftRetry | null

    return retry && typeof retry.id === 'string' && typeof retry.text === 'string' && Array.isArray(retry.attachmentIds)
      ? retry
      : undefined
  } catch {
    return undefined
  }
}

export function writeDraftRetry(scope: string | null | undefined, retry: ComposerDraftRetry | undefined): void {
  try {
    if (retry) {
      window.localStorage.setItem(RETRY_PREFIX + scopeKey(scope), JSON.stringify(retry))
    } else {
      window.localStorage.removeItem(RETRY_PREFIX + scopeKey(scope))
    }
  } catch {
    // Best-effort like the draft text itself: without it a reload sends the words as new.
  }
}
