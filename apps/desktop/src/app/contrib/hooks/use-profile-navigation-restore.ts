import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $diskPluginsScanPending } from '@/contrib/runtime-loader'
import { activeConnectionScopeSuffix } from '@/lib/connection-scoped'
import { gatewayActivationEpoch } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $profileNavigationRequest, dropProfileNavigationRestore } from '@/store/profile-navigation'
import {
  $connection,
  forgetSessionOwnerHintsForSession,
  lineageAliases,
  sessionMatchesStoredId,
  sessionOwnerRouteFromRow,
  setSessionOwnerHint
} from '@/store/session'
import { $sessionListAuthority } from '@/store/session-list-authority'
import { $sessionTiles } from '@/store/session-states'
import { isBrowserWindow, isHudWindow, isPeerInstanceWindow } from '@/store/windows'

import { NEW_CHAT_ROUTE } from '../../routes'

import { isMainOnNewChatRoute, type RememberedSession, restoreRememberedNavigation } from './remembered-navigation'

interface ProfileNavigationRestoreOptions {
  activeProfile: string
  locationPathname: string
  profileReady: boolean
  navigate: (to: string, options?: { replace?: boolean }) => void
  sessions: readonly RememberedSession[]
  resumeExhaustedSessionId: null | string
}

/** A rail pick may replace only its own untouched draft, never a later user choice. */
export function useProfileNavigationRestore({
  activeProfile,
  locationPathname,
  profileReady,
  navigate,
  sessions,
  resumeExhaustedSessionId
}: ProfileNavigationRestoreOptions): void {
  const request = useStore($profileNavigationRequest)
  const authority = useStore($sessionListAuthority)
  const diskPluginsScanPending = useStore($diskPluginsScanPending)
  const connection = useStore($connection)
  useEffect(() => {
    if (!request) {
      return
    }

    const drop = () => dropProfileNavigationRestore(request.sequence)

    if (isHudWindow() || isBrowserWindow() || isPeerInstanceWindow()) {
      drop()

      return
    }

    // useFreshSessionRequest runs first. HashRouter useTransitions={false}
    // commits its hash synchronously, but this effect can still have the old
    // render pathname. Wait for that render rather than rejecting our reset.
    if (locationPathname !== NEW_CHAT_ROUTE && isMainOnNewChatRoute(locationPathname, 'barrier')) {
      return
    }

    const ownsDraft = () =>
      $profileNavigationRequest.get()?.sequence === request.sequence &&
      isMainOnNewChatRoute(locationPathname, 'restore')

    if (
      !ownsDraft() ||
      request.activation === 'failed' ||
      (request.scope !== undefined && request.scope !== activeConnectionScopeSuffix())
    ) {
      drop()

      return
    }

    if (
      !profileReady ||
      request.activation !== 'ready' ||
      activeProfile !== request.profile ||
      $activeGatewayProfile.get() !== request.profile ||
      authority?.profile !== request.profile ||
      authority.scope !== request.scope ||
      authority.epoch !== gatewayActivationEpoch()
    ) {
      return
    }

    if (authority.status === 'failed') {
      drop()

      return
    }

    restoreRememberedNavigation({
      profile: request.profile,
      sessions,
      diskPluginsScanPending,
      resumeExhaustedSessionId,
      onDestination: (route, sessionId) => {
        drop()
        const aliases = sessionId ? lineageAliases(sessionId, sessions) : []

        if (route && !$sessionTiles.get().some(tile => aliases.includes(tile.storedSessionId))) {
          // Repair hints only for the session we actually open, never a tile
          // that leaves main on its draft.
          const row = sessionId ? sessions.find(row => sessionMatchesStoredId(row, sessionId)) : undefined

          if (sessionId && row) {
            const owner = sessionOwnerRouteFromRow(row)

            if (owner) {
              setSessionOwnerHint(sessionId, owner)
            } else {
              forgetSessionOwnerHintsForSession(sessionId)
            }
          }

          navigate(route, { replace: true })
        }
      }
    })
  }, [
    activeProfile,
    authority?.profile,
    authority?.scope,
    authority?.epoch,
    authority?.status,
    connection,
    diskPluginsScanPending,
    locationPathname,
    navigate,
    profileReady,
    request,
    resumeExhaustedSessionId,
    sessions
  ])
}
