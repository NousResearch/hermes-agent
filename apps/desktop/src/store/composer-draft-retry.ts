import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'

import type { ComposerAttachment, SessionDraft } from './composer'

/** Provenance travels with the restored draft, never with equal text typed in
 * another composer. The prepared-submission journal owns the wire payload. */
export interface ComposerDraftRetry {
  id: string
  text: string
  attachmentIds: string[]
  fromQueue?: boolean
  /** Renderer-only: an empty editor during this send must not erase its draft. */
  pending?: boolean
}

export function draftRetryTextMatches(retry: ComposerDraftRetry | undefined, text: string): boolean {
  return Boolean(retry && sanitizeComposerInput(retry.text).trim() === sanitizeComposerInput(text).trim())
}

export function matchingDraftRetry(retry: ComposerDraftRetry | undefined, text: string, attachments: ComposerAttachment[]) {
  return retry && draftRetryTextMatches(retry, text) && JSON.stringify(retry.attachmentIds) ===
    JSON.stringify(attachments.map(attachment => attachment.occurrenceId ?? attachment.id)) ? retry : undefined
}

export function visibleSessionDraft(draft: SessionDraft): SessionDraft {
  return draft.retry?.pending ? { ...draft, text: '', attachments: [] } : draft
}

export function serializeSessionDraft(draft: SessionDraft): string | Record<string, unknown> {
  if (!draft.retry) { return draft.text }
  const { pending: _pending, ...retry } = draft.retry

  return { text: draft.text, retry, attachments: draft.attachments.map(attachment => {
    const { previewUrl: _preview, thumbnailUrl: _thumbnail, uploadState: _upload, ...retained } = attachment

    return retained
  }) }
}

export function deserializeSessionDraft(value: unknown): SessionDraft | undefined {
  if (typeof value === 'string') { return { text: value, attachments: [] } }

  if (!value || typeof value !== 'object') { return undefined }
  const draft = value as SessionDraft
  const retry = draft.retry

  if (typeof draft.text !== 'string' || !retry || typeof retry.id !== 'string' || !draftRetryTextMatches(retry, draft.text) ||
      !Array.isArray(retry.attachmentIds) || !retry.attachmentIds.every(id => typeof id === 'string') ||
      !Array.isArray(draft.attachments) || !draft.attachments.every(attachment => typeof attachment?.id === 'string')) {
    return undefined
  }

  return { text: draft.text, attachments: draft.attachments, retry: { ...retry, text: draft.text, pending: false } }
}
