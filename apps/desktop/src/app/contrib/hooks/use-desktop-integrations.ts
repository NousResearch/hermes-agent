import { useEffect, useRef } from 'react'

import { closeActiveTab } from '@/app/chat/close-tab'
import { sessionIdentityForNotification } from '@/lib/session-ids'
import { respondToApprovalAction } from '@/store/native-notifications'
import { normalizeProfileKey } from '@/store/profile'
import {
  getRememberedRoute,
  getRememberedSessionIdentity,
  setRememberedRoute,
  setRememberedSessionIdentity
} from '@/store/session'
import { onSessionsChanged } from '@/store/session-sync'
import { openUpdatesWindow, startUpdatePoller, stopUpdatePoller } from '@/store/updates'
import { isSecondaryWindow } from '@/store/windows'

import { requestComposerFocus, requestComposerInsert } from '../../chat/composer/focus'
import { appViewForPath, isOverlayView, NEW_CHAT_ROUTE, routeSessionId, sessionRoute } from '../../routes'

interface DesktopIntegrationsParams {
  activeGatewayProfile: string
  chatOpen: boolean
  hasPreview: boolean
  locationPathname: string
  navigate: (to: string, options?: { replace?: boolean }) => void
  refreshSessions: () => Promise<unknown> | unknown
  resumeExhaustedSessionId: null | string
  routedSessionId: null | string
  routedSessionProfile: null | string
  runtimeIdByStoredSessionId: { readonly current: Map<string, string> }
}

/**
 * All the Electron-main / OS / cross-window integrations the shell listens for:
 * update polling, the ⌘W close shortcut, deep links, native-notification
 * navigation, preview-shortcut enablement, remembered-session restore, and
 * cross-window session-list sync. Kept out of the wiring controller so the
 * "talks to the desktop shell" surface reads as one unit.
 */
export function useDesktopIntegrations({
  activeGatewayProfile,
  locationPathname,
  navigate,
  refreshSessions,
  resumeExhaustedSessionId,
  routedSessionId,
  routedSessionProfile,
  runtimeIdByStoredSessionId
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

  // Remember the open chat (session id for notifications/resume) AND the last
  // non-overlay route (a page like /skills, or a session route) so a relaunch
  // lands where you were. Overlays (settings/command-center/…) aren't stored —
  // you don't want to boot into a modal.
  useEffect(() => {
    if (routedSessionId) {
      setRememberedSessionIdentity(routedSessionId, routedSessionProfile ?? activeGatewayProfile)
    }

    if (!isOverlayView(appViewForPath(locationPathname))) {
      setRememberedRoute(locationPathname)
    }
  }, [activeGatewayProfile, locationPathname, routedSessionId, routedSessionProfile])

  const restoredRef = useRef(false)

  // Restore once on cold start — only when the renderer booted at the default
  // route (a hidden-then-shown window keeps its own route). Prefer the full
  // remembered route (covers pages); fall back to the last session id.
  useEffect(() => {
    if (restoredRef.current || locationPathname !== NEW_CHAT_ROUTE) {
      restoredRef.current = true

      return
    }

    restoredRef.current = true
    const route = getRememberedRoute()
    const remembered = getRememberedSessionIdentity()

    if (route && route !== NEW_CHAT_ROUTE && !isOverlayView(appViewForPath(route))) {
      const rememberedRouteSessionId = routeSessionId(route)

      navigate(
        rememberedRouteSessionId && remembered?.sessionId === rememberedRouteSessionId
          ? sessionRoute(remembered.sessionId, remembered.profile)
          : route,
        { replace: true }
      )

      return
    }

    if (remembered) {
      navigate(sessionRoute(remembered.sessionId, remembered.profile), { replace: true })
    }
  }, [locationPathname, navigate])

  useEffect(() => {
    const remembered = getRememberedSessionIdentity()

    if (
      resumeExhaustedSessionId &&
      remembered?.sessionId === resumeExhaustedSessionId &&
      normalizeProfileKey(remembered.profile) === normalizeProfileKey(activeGatewayProfile)
    ) {
      setRememberedSessionIdentity(null)
    }
  }, [activeGatewayProfile, resumeExhaustedSessionId])

  // Native-notification click -> jump to the session (runtime id translated to
  // the stored id the chat route is keyed by); action buttons resolve in place.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onFocusSession?.(payload => {
      if (payload.sessionId) {
        const target = payload.profile
          ? { profile: payload.profile, sessionId: payload.sessionId }
          : sessionIdentityForNotification(payload.sessionId, runtimeIdByStoredSessionId.current)

        navigate(sessionRoute(target.sessionId, target.profile))
      }
    })

    return () => unsubscribe?.()
  }, [navigate, runtimeIdByStoredSessionId])

  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onNotificationAction?.(({ actionId, profile, runtimeSessionId }) => {
      void respondToApprovalAction(runtimeSessionId ?? null, actionId, profile)
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
