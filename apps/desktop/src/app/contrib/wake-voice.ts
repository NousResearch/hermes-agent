import { activateWakeIndicator } from '@/lib/wake-indicator'
import { playWakeSound } from '@/lib/wake-sound'
import { requestVoiceConversationStart } from '@/store/composer'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, ensureGatewayProfile, newSessionInProfile, normalizeProfileKey } from '@/store/profile'
import {
  $focusedRuntimeId,
  $focusedSessionIsTile,
  $focusedStoredSessionId,
  $sessionTiles
} from '@/store/session-states'
import { stopClientCapture } from '@/store/wake-word'

import type { useSessionActions } from '../session/hooks/use-session-actions'

type WakeSessionActions = Pick<ReturnType<typeof useSessionActions>, 'openNewSessionTile' | 'startFreshSessionDraft'>

/** Production gateway-event boundary; explicit profile routing retains precedence. */
export function handleWakeVoiceEvent(event: { type: string; payload?: unknown }, actions: WakeSessionActions): boolean {
  if (event.type !== 'wake.detected') {
    return false
  }

  const payload = event.payload as { profile?: null | string; start_new_session?: boolean } | undefined
  stopClientCapture()
  playWakeSound()
  activateWakeIndicator()
  const targetProfile = payload?.profile?.trim()

  if (targetProfile && normalizeProfileKey(targetProfile) !== normalizeProfileKey($activeGatewayProfile.get())) {
    if (payload?.start_new_session !== false) {
      newSessionInProfile(targetProfile)
    } else {
      void ensureGatewayProfile(normalizeProfileKey(targetProfile)).catch((error: unknown) => {
        notifyError(error, `Failed to switch to profile "${normalizeProfileKey(targetProfile)}"`)
      })
    }

    requestVoiceConversationStart()
  } else {
    void startFocusedWakeVoice(payload?.start_new_session !== false, actions).catch((error: unknown) => {
      notifyError(error, 'Failed to start wake voice conversation')
    })
  }

  return true
}

/** Resolve the default wake target at dispatch, not from primary navigation's durable id. */
export async function startFocusedWakeVoice(startNewSession: boolean, actions: WakeSessionActions): Promise<void> {
  if (!startNewSession) {
    const runtimeId = $focusedRuntimeId.get()

    // An unresolved tile is not permission to start in an unrelated main chat.
    if (runtimeId || !$focusedSessionIsTile.get()) {
      requestVoiceConversationStart(runtimeId)
    }

    return
  }

  if ($focusedSessionIsTile.get()) {
    const storedId = $focusedStoredSessionId.get()
    const tile = $sessionTiles.get().find(candidate => candidate.storedSessionId === storedId)

    // Keep the main conversation intact. A fresh unlisted tab in this tile's
    // group owns the request only after its backend runtime has been created.
    const runtimeId = await actions.openNewSessionTile('center', {
      anchor: storedId ? `session-tile:${storedId}` : undefined,
      listed: false,
      ...(tile
        ? {
            workspaceScope: {
              workspaceMode: tile.workspaceMode ?? 'sessions',
              workspaceOwnerKey: tile.workspaceOwnerKey,
              ownerRoute: tile.ownerRoute
            },
            profile: tile.ownerProfile,
            route: tile.ownerRoute
          }
        : {})
    })

    if (runtimeId) {
      requestVoiceConversationStart(runtimeId)
    }

    return
  }

  actions.startFreshSessionDraft()
  requestVoiceConversationStart()
}
