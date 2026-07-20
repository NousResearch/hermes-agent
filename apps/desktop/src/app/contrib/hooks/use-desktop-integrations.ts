import { useEffect, useRef } from 'react'

import { closeActiveTab } from '@/app/chat/close-tab'
import { storedSessionIdForNotification } from '@/lib/session-ids'
import { respondToApprovalAction } from '@/store/native-notifications'
import { normalizeProfileKey } from '@/store/profile'
import {
  getRememberedRoute,
  getRememberedSessionId,
  sessionMatchesStoredId,
  setRememberedRoute,
  setRememberedSessionId
} from '@/store/session'
import { onSessionsChanged } from '@/store/session-sync'
import { openUpdatesWindow, startUpdatePoller, stopUpdatePoller } from '@/store/updates'
import { isSecondaryWindow } from '@/store/windows'
import type { SessionInfo } from '@/types/hermes'

import { requestComposerFocus, requestComposerInsert } from '../../chat/composer/focus'
import { appViewForPath, isOverlayView, NEW_CHAT_ROUTE, routeSessionId, sessionRoute } from '../../routes'

type RememberedSession = Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>

function sessionBelongsToProfile(
  sessions: readonly RememberedSession[],
  storedSessionId: string,
  profile: string
): boolean {
  const profileKey = normalizeProfileKey(profile)

  return sessions.some(session => {
    const owner = session.profile?.trim()

    return Boolean(
      owner && sessionMatchesStoredId(session, storedSessionId) && normalizeProfileKey(owner) === profileKey
    )
  })
}

interface DesktopIntegrationsParams {
  activeProfile: string
  chatOpen: boolean
  hasPreview: boolean
  locationPathname: string
  navigate: (to: string, options?: { replace?: boolean }) => void
  profileReady: boolean
  refreshSessions: () => Promise<unknown> | unknown
  resumeExhaustedSessionId: null | string
  routedSessionId: null | string
  runtimeIdByStoredSessionId: { readonly current: Map<string, string> }
  sessions: readonly RememberedSession[]
}

/**
 * All the Electron-main / OS / cross-window integrations the shell listens for:
 * update polling, the ⌘W close shortcut, deep links, native-notification
 * navigation, preview-shortcut enablement, remembered-session restore, and
 * cross-window session-list sync. Kept out of the wiring controller so the
 * "talks to the desktop shell" surface reads as one unit.
 */
export function useDesktopIntegrations({
  activeProfile,
  locationPathname,
  navigate,
  profileReady,
  refreshSessions,
  resumeExhaustedSessionId,
  routedSessionId,
  runtimeIdByStoredSessionId,
  sessions
}: DesktopIntegrationsParams): void {
  // Update polling — populates $desktopVersion/$updateStatus, which feed the
  // statusbar version pill and the update toasts. Also honors the main
  // process's "open updates" menu request.
  useEffect(() => {
    startUpdatePoller()
    const unsubscribe = window.hermesDesktop?.onOpenUpdatesRequested?.(() => openUpdatesWindow())

    return () => {
      unsubscribe?.()
      stopUpdatePoller()
    }
  }, [])

  // The renderer OWNS ⌘W: on macOS the native menu accelerator would else
  // close the window, so claim it unconditionally — the menu then routes ⌘W
  // to us (close-preview-requested IPC) and we decide tab-vs-window.
  useEffect(() => {
    window.hermesDesktop?.setPreviewShortcutActive?.(true)
  }, [])

  const restoredRef = useRef(false)

  // Wait until boot has adopted the primary profile, then restore that profile's
  // navigation exactly once. The same effect owns subsequent writes so the
  // initial `/` cannot overwrite remembered history before it is read.
  useEffect(() => {
    if (!profileReady) {
      return
    }

    if (!restoredRef.current) {
      restoredRef.current = true

      // Only cold-start navigation at the default route is replaceable; a deep
      // link or hidden-then-shown window keeps its explicit destination.
      if (locationPathname === NEW_CHAT_ROUTE) {
        const route = getRememberedRoute(activeProfile)
        const routeSession = route ? routeSessionId(route) : null

        if (
          route &&
          route !== NEW_CHAT_ROUTE &&
          !isOverlayView(appViewForPath(route)) &&
          (!routeSession || sessionBelongsToProfile(sessions, routeSession, activeProfile))
        ) {
          navigate(route, { replace: true })

          return
        }

        if (routeSession) {
          setRememberedRoute(null, activeProfile)
        }

        const last = getRememberedSessionId(activeProfile)

        if (last && sessionBelongsToProfile(sessions, last, activeProfile)) {
          navigate(sessionRoute(last), { replace: true })

          return
        }

        if (last) {
          setRememberedSessionId(null, activeProfile)
        }
      }
    }

    // Remember the open chat (session id for notifications/resume) AND the last
    // non-overlay route (a page like /skills, or a session route) per profile.
    // Session-shaped routes require an explicit matching owner; unresolved and
    // wrong-profile rows must not replace known-safe navigation.
    if (routedSessionId && sessionBelongsToProfile(sessions, routedSessionId, activeProfile)) {
      setRememberedSessionId(routedSessionId, activeProfile)
      setRememberedRoute(locationPathname, activeProfile)
    } else if (!routedSessionId && !isOverlayView(appViewForPath(locationPathname))) {
      setRememberedRoute(locationPathname, activeProfile)
    }
  }, [activeProfile, locationPathname, navigate, profileReady, routedSessionId, sessions])

  useEffect(() => {
    if (!profileReady || !resumeExhaustedSessionId) {
      return
    }

    if (getRememberedSessionId(activeProfile) === resumeExhaustedSessionId) {
      setRememberedSessionId(null, activeProfile)
    }

    if (routeSessionId(getRememberedRoute(activeProfile) ?? '') === resumeExhaustedSessionId) {
      setRememberedRoute(null, activeProfile)
    }
  }, [activeProfile, profileReady, resumeExhaustedSessionId])

  // Native-notification click -> jump to the session (runtime id translated to
  // the stored id the chat route is keyed by); action buttons resolve in place.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onFocusSession?.(sessionId => {
      if (sessionId) {
        navigate(sessionRoute(storedSessionIdForNotification(sessionId, runtimeIdByStoredSessionId.current)))
      }
    })

    return () => unsubscribe?.()
  }, [navigate, runtimeIdByStoredSessionId])

  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onNotificationAction?.(({ actionId, sessionId }) => {
      void respondToApprovalAction(sessionId ?? null, actionId)
    })

    return () => unsubscribe?.()
  }, [])

  // hermes:// deep links -> a reviewable /blueprint command in the composer.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onDeepLink?.(payload => {
      if (!payload || payload.kind !== 'blueprint' || !payload.name) {
        return
      }

      const slots = Object.entries(payload.params || {})
        .map(([k, v]) => {
          const sval = /\s/.test(v) ? `"${v.replace(/"/g, '\\"')}"` : v

          return `${k}=${sval}`
        })
        .join(' ')

      const command = `/blueprint ${payload.name}${slots ? ' ' + slots : ''}`
      requestComposerInsert(command, { mode: 'block', target: 'main' })
      requestComposerFocus('main')
    })

    void window.hermesDesktop?.signalDeepLinkReady?.()

    return () => unsubscribe?.()
  }, [])

  // ⌘W via the macOS menu accelerator → close the focused tab; if nothing is
  // closeable, fall back to closing the window (so ⌘W still works as the
  // OS-standard window close, esp. secondary windows). The Win/Linux keyboard
  // path is the `view.closeTab` keybind (use-keybinds), sharing closeActiveTab.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onClosePreviewRequested?.(() => void closeActiveTab())

    return () => unsubscribe?.()
  }, [])

  // Another window mutated the shared session list -> re-pull the sidebar.
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    return onSessionsChanged(() => void refreshSessions())
  }, [refreshSessions])
}
