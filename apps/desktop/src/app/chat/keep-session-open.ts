import { translateNow } from '@/i18n'
import { requestComposerDraftSync } from '@/store/composer'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  $sessions,
  getSessionOwnerHint,
  knownSessionOwner
} from '@/store/session'
import {
  $botChatScopes,
  focusOpenSession,
  holdSessionOwnerUntilForeground,
  openSessionTile,
  patchSessionTile
} from '@/store/session-states'

/** Hand the primary chat to the existing tile/cache path before browsing another chat. */
export function keepWorkspaceSessionOpen(startFreshSessionDraft: () => unknown): boolean {
  const storedSessionId = $selectedStoredSessionId.get()
  const runtimeId = $activeSessionId.get()

  const owner =
    (storedSessionId && getSessionOwnerHint(storedSessionId)) || knownSessionOwner($sessions.get(), storedSessionId)

  if (!storedSessionId || !runtimeId || !owner) {
    return false
  }

  // The reset detaches main before the tile guard can accept it. Hold the exact
  // owner through that gap so a running chat never loses its socket.
  const releaseOwner = holdSessionOwnerUntilForeground(storedSessionId, owner)

  try {
    requestComposerDraftSync('flush')
    startFreshSessionDraft()
    openSessionTile(storedSessionId, 'center', 'workspace', undefined, {
      ...($botChatScopes.get()[storedSessionId] ?? { workspaceMode: 'sessions' }),
      ...(typeof owner === 'string' ? { ownerProfile: owner } : { ownerRoute: owner })
    })
    patchSessionTile(storedSessionId, { runtimeId })
    focusOpenSession(storedSessionId)
  } finally {
    releaseOwner()
  }

  return true
}

/** Explicit tab opens report a declined or failed handoff through the shared menu path. */
export function openWorkspaceSessionInNewTab(startFreshSessionDraft: () => unknown): boolean {
  try {
    if (keepWorkspaceSessionOpen(startFreshSessionDraft)) {
      return true
    }

    notify({
      kind: 'error',
      title: translateNow('sidebar.row.openInNewTab'),
      message: translateNow('simpleWorkspace.openInNewTabFailed')
    })
  } catch (error) {
    notifyError(error, translateNow('sidebar.row.openInNewTab'))
  }

  return false
}
