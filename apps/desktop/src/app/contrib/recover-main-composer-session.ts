import { $composerAttachments, $freshDraftKey } from '@/store/composer'
import { $activeConnectionId } from '@/store/connections'
import {
  $activeGatewayProfile,
  $freshSessionRequest,
  currentNewChatIntent,
  isExplicitNewChatIntent
} from '@/store/profile'
import { type ProfileSessionRecoveryScope, recoverMostRecentProfileSession } from '@/store/profile-session-recovery'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from '@/store/session'
import { currentUserSessionOpen } from '@/store/session-open-intent'
import type { SessionInfo } from '@/types/hermes'

import { requestComposerGetDraft } from '../chat/composer/focus'

export function mainComposerRecoveryIntent(routeToken: string): string {
  return JSON.stringify([
    routeToken,
    currentUserSessionOpen(),
    currentNewChatIntent(),
    $freshSessionRequest.get(),
    $freshDraftKey.get()
  ])
}

/** Consult the main editor, not a tile/focused composer or the legacy draft atom. */
export async function recoverMainComposerSession(params: {
  isCurrent: () => boolean
  onResume: (session: SessionInfo) => void
  scope: ProfileSessionRecoveryScope
}): Promise<boolean> {
  const canRecover = () =>
    params.isCurrent() &&
    !isExplicitNewChatIntent() &&
    ($activeConnectionId.get() ?? null) === (params.scope.connectionId ?? null) &&
    $activeGatewayProfile.get() === params.scope.profile &&
    !$activeSessionId.get() &&
    !$selectedStoredSessionId.get()

  if (!canRecover()) {
    return false
  }

  const draft = await requestComposerGetDraft([], { target: 'main' })

  // No mounted editor is not proof of an empty draft. Intent may change during the read.
  if (!draft || draft.text === null || !canRecover()) {
    return false
  }

  return recoverMostRecentProfileSession({
    hasNonEmptyDraft: draft.text.trim().length > 0 || $composerAttachments.get().length > 0,
    onResume: params.onResume,
    scope: params.scope,
    sessions: $sessions.get()
  })
}
