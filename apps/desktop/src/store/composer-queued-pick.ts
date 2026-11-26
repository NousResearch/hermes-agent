import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import { connectionScopeSuffix } from '@/lib/connection-scoped'
import { persistString, storedString } from '@/lib/storage'

import { $connection, composerScopeForOwner, releaseComposerModelSelection } from './session'

const QUEUED_PICK_KEY = 'hermes.desktop.composer.queued-pick'

interface QueuedLaunchPick {
  model: string
  provider: string
  sessionId: string
}

// Every read and write names the owner scope captured when the pick started: a result can outlive a switch.
function storedQueuedLaunchPick(scope: string): null | QueuedLaunchPick {
  try {
    const pick = JSON.parse(storedString(`${QUEUED_PICK_KEY}${scope}`) ?? 'null') as null | QueuedLaunchPick

    return typeof pick?.sessionId === 'string' ? pick : null
  } catch {
    return null
  }
}

// The newest pick (ledger order) that answered for each owner's session in this run: replies can arrive out of
// order, and another owner can hold a session with the same id (imports).
const newestAnswerBySession = new Map<string, number>()

/** A primary pick the gateway queues for its session's next turn is saved as the launch pick at once. Record which
 *  session queued which route, next to that pick in its owner's storage (across reloads), so only that session's
 *  refusal or decline can release it; any later pick on the session retires the record, an older one's late reply
 *  never does. */
export const noteQueuedLaunchPick = (
  sessionId: string,
  route: { model: string; provider: string } | null,
  scope: null | string,
  pickOrder?: number
) => {
  if (scope === null) {
    return
  }

  if (pickOrder !== undefined) {
    const answered = `${scope}\n${sessionId}`

    if (pickOrder < (newestAnswerBySession.get(answered) ?? 0)) {
      return
    }

    newestAnswerBySession.set(answered, pickOrder)
  }

  if (route || storedQueuedLaunchPick(scope)?.sessionId === sessionId) {
    persistString(
      `${QUEUED_PICK_KEY}${scope}`,
      route ? JSON.stringify({ model: route.model, provider: route.provider, sessionId }) : null
    )
  }
}

/** The session's queued pick of `model` was refused or declined: release the route it saved for owner `scope`. */
export const releaseQueuedLaunchPick = (sessionId: string, model: unknown, scope: null | string): void => {
  const queued = scope === null ? null : storedQueuedLaunchPick(scope)

  if (queued?.sessionId === sessionId && queued.model === model) {
    noteQueuedLaunchPick(sessionId, null, scope)
    releaseComposerModelSelection(queued, scope)
  }
}

/** Rename counterpart (local connection only, like the tiles it joins). A rename never lands on a live profile's
 *  name, so whatever the new name holds was left by a deleted one and is replaced. */
export function migrateQueuedLaunchPicksForProfile(from: string, to: string): void {
  const key = (profile: string) => `${QUEUED_PICK_KEY}${composerScopeForOwner(LOCAL_CONNECTION_ID, profile)}`

  persistString(key(to), storedString(key(from)))
  persistString(key(from), null)
}

/** The deleted profile's record on the connection it was deleted from: `connectionId` names it ('' is local);
 *  without one, the active connection (the bare local scope names no profile). */
export function dropQueuedLaunchPicksForProfile(profile: string, connectionId?: string): void {
  const connection = $connection.get()

  if (connectionId === undefined && connection?.mode === 'remote' && !connection.connectionId) {
    persistString(`${QUEUED_PICK_KEY}${connectionScopeSuffix({ ...connection, profile })}`, null)

    return
  }

  const owner = (connectionId ?? connection?.connectionId ?? '').trim() || LOCAL_CONNECTION_ID
  persistString(`${QUEUED_PICK_KEY}${composerScopeForOwner(owner, profile)}`, null)
}
