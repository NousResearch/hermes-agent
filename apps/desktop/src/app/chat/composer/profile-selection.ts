import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useState } from 'react'

import { $profiles } from '@/store/profile'

const MAX_STARTED_PROFILE_DRAFTS = 64
const $startedProfileDraftKeys = atom<ReadonlySet<string>>(new Set())

export type ProfileSelectionMode = 'draft' | 'started' | 'submitting'

export function defaultNewChatProfile(): string {
  return $profiles.get().find(profile => profile.is_default)?.name ?? 'default'
}

function markProfileDraftStarted(draftKey: string): void {
  const current = $startedProfileDraftKeys.get()

  if (current.has(draftKey)) {
    return
  }

  const next = new Set(current)
  next.add(draftKey)

  while (next.size > MAX_STARTED_PROFILE_DRAFTS) {
    const oldest = next.values().next().value

    if (oldest === undefined) {
      break
    }

    next.delete(oldest)
  }

  $startedProfileDraftKeys.set(next)
}

export function deriveProfileSessionStarted({
  hasMessages,
  hasPersistedSession,
  isSessionTile
}: {
  hasMessages: boolean
  hasPersistedSession: boolean
  isSessionTile: boolean
}): boolean {
  // A runtime id or an unsaved preview id is not proof that the user started a
  // session. Draft sessions are persisted lazily on the first prompt.
  return isSessionTile || hasPersistedSession || hasMessages
}

export function useDraftProfileSelection({ draftKey, sessionStarted }: { draftKey: string; sessionStarted: boolean }): {
  beginSubmission: () => void
  canSelectProfile: boolean
  cancelBeforeStart: () => void
  markStarted: () => void
  mode: ProfileSelectionMode
} {
  const startedDraftKeys = useStore($startedProfileDraftKeys)
  const [pendingDraftKey, setPendingDraftKey] = useState<null | string>(null)
  const started = sessionStarted || startedDraftKeys.has(draftKey)
  const submitting = !started && pendingDraftKey === draftKey
  const mode = started ? 'started' : submitting ? 'submitting' : 'draft'

  const beginSubmission = useCallback(() => {
    if (sessionStarted || $startedProfileDraftKeys.get().has(draftKey)) {
      return
    }

    setPendingDraftKey(draftKey)
  }, [draftKey, sessionStarted])

  const markStarted = useCallback(() => {
    markProfileDraftStarted(draftKey)
    setPendingDraftKey(current => (current === draftKey ? null : current))
  }, [draftKey])

  const cancelBeforeStart = useCallback(() => {
    setPendingDraftKey(current => (current === draftKey ? null : current))
  }, [draftKey])

  return {
    beginSubmission,
    canSelectProfile: mode === 'draft',
    cancelBeforeStart,
    markStarted,
    mode
  }
}
