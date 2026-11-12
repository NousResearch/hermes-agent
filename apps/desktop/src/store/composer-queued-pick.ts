import { persistString, storedString } from '@/lib/storage'

import { composerSelectionKey, releaseComposerModelSelection } from './session'

const QUEUED_PICK_KEY = 'hermes.desktop.composer.queued-pick'

function storedKey(): null | string {
  const key = composerSelectionKey(QUEUED_PICK_KEY)

  return key === null ? null : storedString(key)
}

interface QueuedLaunchPick {
  model: string
  provider: string
  sessionId: string
}

function storedQueuedLaunchPick(): null | QueuedLaunchPick {
  try {
    const pick = JSON.parse(storedKey() ?? 'null') as null | QueuedLaunchPick

    return typeof pick?.sessionId === 'string' ? pick : null
  } catch {
    return null
  }
}

/** A primary pick the gateway queues for its session's next turn is saved as the launch pick at once. Record which
 *  session queued which route, next to that pick (owner-scoped, across reloads), so only that session's refusal
 *  or decline can release it; any later pick on the session retires the record. */
export const noteQueuedLaunchPick = (sessionId: string, route: { model: string; provider: string } | null) => {
  const key = composerSelectionKey(QUEUED_PICK_KEY)

  if (key !== null && (route || storedQueuedLaunchPick()?.sessionId === sessionId)) {
    persistString(key, route ? JSON.stringify({ model: route.model, provider: route.provider, sessionId }) : null)
  }
}

/** The session's queued pick of `model` was refused or declined: release the route it saved. */
export const releaseQueuedLaunchPick = (sessionId: string, model: unknown, scope: null | string): void => {
  const queued = storedQueuedLaunchPick()

  if (queued?.sessionId === sessionId && queued.model === model) {
    noteQueuedLaunchPick(sessionId, null)
    releaseComposerModelSelection(queued, scope)
  }
}
