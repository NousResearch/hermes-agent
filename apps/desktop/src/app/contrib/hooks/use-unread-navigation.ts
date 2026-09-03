import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { openSession, type OpenSessionNavigate } from '@/app/open-session'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { findGroup, findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $narrowOverlayChrome } from '@/components/pane-shell/tree/renderer/narrow-overlay-state'
import { $activeTreeGroup, $layoutTree, $narrowViewport, revealTreePane } from '@/components/pane-shell/tree/store'
import { $workspaceMode, $workspaceOwnerKey } from '@/components/pane-shell/workspace-scope'
import { getSession } from '@/hermes'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile, $freshSessionRequest, $profileScope } from '@/store/profile'
import {
  $selectedStoredSessionId,
  $sessionResumeRequest,
  $sessions,
  forgetSessionOwnerHintsForSession,
  lineageAliases,
  requestSessionResume
} from '@/store/session'
import { $unreadSessionTargets, type UnreadSessionTarget } from '@/store/session-dot-state'
import type { SessionProfileRoute } from '@/store/session-request-router'
import { $focusedStoredSessionId, $sessionTiles } from '@/store/session-states'
import {
  $openNextUnreadRequest,
  openNextValidUnread,
  ownerRouteForUnreadTarget,
  preflightCandidatesForUnreadTarget
} from '@/store/session-unread-navigation'

function isStillUnread(target: UnreadSessionTarget): boolean {
  return $unreadSessionTargets
    .get()
    .some(
      current =>
        current.id === target.id &&
        current.kind === target.kind &&
        current.connectionId === target.connectionId &&
        current.profile === target.profile
    )
}

function activePaneId(): string | undefined {
  const tree = $layoutTree.get()
  const groupId = $activeTreeGroup.get()

  return tree ? (groupId ? findGroup(tree, groupId) : findGroupOfPane(tree, 'workspace'))?.active : undefined
}

/** openSession's focus fast path is ID/lineage-only. Until that shared store
 * can represent source twins, skip an open surface whose owner is unproven.
 * Do this BEFORE publishing resume intent or acknowledging any unread state. */
function canOpenUnread(target: UnreadSessionTarget, owner: SessionProfileRoute | null): boolean {
  const aliases = lineageAliases(target.id, $sessions.get())
  const tiles = $sessionTiles.get().filter(tile => aliases.includes(tile.storedSessionId))

  if (tiles.length) {
    return tiles.every(
      ({ ownerRoute }) =>
        owner &&
        ownerRoute &&
        owner.connectionId === ownerRoute.connectionId &&
        owner.profile === ownerRoute.profile &&
        (owner.targetProfile || owner.profile) === (ownerRoute.targetProfile || ownerRoute.profile)
    )
  }

  // Main's selected ID alone cannot prove which source is already hydrated.
  return !aliases.includes($selectedStoredSessionId.get() ?? '')
}

export function useUnreadNavigation(navigate: OpenSessionNavigate, routeToken: string): void {
  const request = useStore($openNextUnreadRequest)
  // A genuine remount consumes no historical intent; StrictMode can re-run
  // effects without replaying an already seen counter.
  const seen = useRef($openNextUnreadRequest.get())
  const pending = useRef<null | (() => void)>(null)
  const currentRoute = useRef(routeToken)
  currentRoute.current = routeToken

  useEffect(() => () => pending.current?.(), [routeToken])

  // eslint-disable-next-line no-restricted-syntax -- consume one-shot navigation intent, not an atom mirror
  useEffect(() => {
    if (request === seen.current) {
      return
    }

    seen.current = request
    revealTreePane('sessions')
    const narrow = $narrowViewport.get()

    if (narrow) {
      // Use the narrow reveal protocol without changing positional dock flags
      // (the left rail can belong to another pane in a flipped wide layout).
      window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: 'sessions', mode: 'open' } }))
    }

    if (pending.current) {
      return
    }

    const capturedRoute = currentRoute.current
    const activeConnectionId = $activeConnectionId.get()
    const cleanups: (() => void)[] = []

    const cancel = () => {
      if (pending.current === cancel) {
        pending.current = null
      }

      cleanups.splice(0).forEach(cleanup => cleanup())
    }

    pending.current = cancel
    const isCurrent = () => pending.current === cancel && currentRoute.current === capturedRoute

    // Capture AFTER our own reveal. A same-group Terminal tab changes neither
    // group nor focused session, whereas resizing the group changes neither
    // its active pane nor the user's intent.
    const capturedPaneId = activePaneId()
    cleanups.push(
      $layoutTree.listen(() => {
        if (activePaneId() !== capturedPaneId) {
          cancel()
        }
      })
    )

    cleanups.push(
      $narrowOverlayChrome.listen(chrome => {
        if (chrome) {
          if (chrome.paneId !== 'sessions') {
            cancel()
          }
        } else {
          // Overlay effects clear then republish chrome, including our own
          // Sessions reveal and cosmetic updates. Only a settled close wins;
          // leaving narrow mode alone is not a different pane selection.
          queueMicrotask(() => {
            if (isCurrent() && $narrowViewport.get() && $narrowOverlayChrome.get()?.paneId !== 'sessions') {
              cancel()
            }
          })
        }
      })
    )

    // Subscribe only during this operation. Synchronous invalidation catches
    // switch-away-and-back and focus-only changes even before React renders.
    for (const store of [
      $activeConnectionId,
      $activeGatewayProfile,
      $profileScope,
      $workspaceMode,
      $workspaceOwnerKey,
      $activeTreeGroup,
      $focusedStoredSessionId,
      $selectedStoredSessionId,
      $sessionResumeRequest,
      $freshSessionRequest
    ]) {
      cleanups.push(store.listen(cancel))
    }

    void openNextValidUnread(
      $unreadSessionTargets.get(),
      async target => {
        if (!isStillUnread(target)) {
          throw new Error('Unread target no longer eligible')
        }

        const baseOwner = ownerRouteForUnreadTarget(target, activeConnectionId)

        const routes = baseOwner ? await window.hermesDesktop.getProfileRoutes([baseOwner.profile]).catch(() => []) : []

        let lastError: unknown

        for (const candidate of preflightCandidatesForUnreadTarget(target, activeConnectionId, routes)) {
          if (!isCurrent() || !isStillUnread(target)) {
            throw new Error('Unread navigation superseded')
          }

          try {
            await getSession(target.id, candidate.scope)

            return candidate.ownerRoute
          } catch (error) {
            lastError = error
          }
        }

        throw lastError
      },
      (target, ownerRoute) => {
        if (!isStillUnread(target) || !canOpenUnread(target, ownerRoute)) {
          return false
        }

        cancel()

        // Preserve the ordinary row's ownerless recovery, including a fresh
        // request replacing a previously captured explicit owner.
        if (!ownerRoute) {
          forgetSessionOwnerHintsForSession(target.id)
        }

        requestSessionResume(target.id, ownerRoute ?? undefined)
        openSession(target.id, navigate)
      },
      isCurrent
    ).finally(cancel)
  }, [navigate, request])
}
