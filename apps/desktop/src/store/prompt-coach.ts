import { atom } from 'nanostores'

import type { PromptCoachAnalysis } from '@/lib/prompt-coach'
import { enhancePromptCoachWithAI } from '@/lib/prompt-coach-ai'

export interface PromptCoachPreview extends PromptCoachAnalysis {
  original: string
}

export type PromptCoachAction = 'dismissed' | 'edited' | 'replaced' | 'sent-original'

export const $promptCoachPreviewBySession = atom<Record<string, PromptCoachPreview>>({})

const keyFor = (sessionId: null | string | undefined): string => sessionId ?? ''

const sendOriginalBySession = new Map<string, string>()

/** Allow exactly one explicit Send original action through the submit guard. */
export function allowPromptCoachOriginal(sessionId: null | string | undefined, original: string): void {
  sendOriginalBySession.set(keyFor(sessionId), original)
}

/** Consume only a matching one-shot bypass; changed drafts are coached again. */
export function consumePromptCoachOriginal(sessionId: null | string | undefined, draft: string): boolean {
  const key = keyFor(sessionId)

  if (sendOriginalBySession.get(key) !== draft) {
    return false
  }

  sendOriginalBySession.delete(key)

  return true
}

export function openPromptCoachPreview(
  sessionId: null | string | undefined,
  original: string,
  analysis: PromptCoachAnalysis
): void {
  $promptCoachPreviewBySession.set({
    ...$promptCoachPreviewBySession.get(),
    [keyFor(sessionId)]: { ...analysis, original }
  })
}

/**
 * Open immediately with the deterministic safe copy, then replace it only if
 * the AI response still belongs to the same visible draft. Generation failure
 * resolves to the local fallback and never blocks the composer.
 */
export function openPromptCoachWithAI(
  sessionId: null | string | undefined,
  original: string,
  analysis: PromptCoachAnalysis
): void {
  const key = keyFor(sessionId)

  openPromptCoachPreview(sessionId, original, { ...analysis, generatedBy: 'pending' })

  void enhancePromptCoachWithAI(original, analysis, sessionId).then(enhanced => {
    const current = $promptCoachPreviewBySession.get()[key]

    if (!current || current.original !== original) {
      return
    }

    openPromptCoachPreview(sessionId, original, enhanced)
  })
}

export function closePromptCoachPreview(sessionId: null | string | undefined): void {
  const key = keyFor(sessionId)
  const current = $promptCoachPreviewBySession.get()

  if (!(key in current)) {
    return
  }

  const next = { ...current }
  delete next[key]
  $promptCoachPreviewBySession.set(next)
}

/** Close a preview as soon as the sampled draft no longer matches it. */
export function reconcilePromptCoachPreview(sessionId: null | string | undefined, draft: string): void {
  const preview = $promptCoachPreviewBySession.get()[keyFor(sessionId)]

  if (preview && preview.original !== draft) {
    closePromptCoachPreview(sessionId)
  }
}

const HISTORY_KEY = 'hermes.promptCoach.history.v1'

/** Device-local counters only. Prompt contents are never persisted. */
export function recordPromptCoachAction(action: PromptCoachAction): void {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY)
    const parsed = raw ? (JSON.parse(raw) as Partial<Record<PromptCoachAction, number>>) : {}
    const next = { ...parsed, [action]: (parsed[action] ?? 0) + 1 }
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
  } catch {
    // History is an optional local convenience. Storage failure never blocks chat.
  }
}
