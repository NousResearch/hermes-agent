import { useEffect, useRef } from 'react'

import { closeActiveTab } from '@/app/chat/close-tab'
import {
  clearStickySessionId,
  cwdLooksSane,
  normalizeStickySlot,
  readStickySessionId,
  setStickyPending,
  takeStickyPending,
  writeStickySessionId
} from '@/lib/deeplink-chat-new'
import { storedSessionIdForNotification } from '@/lib/session-ids'
import { respondToApprovalAction } from '@/store/native-notifications'
import {
  $activeGatewayProfile,
  $newChatProfile,
  ensureGatewayProfile,
  newSessionInProfile,
  requestFreshSession
} from '@/store/profile'
import {
  $projectTree,
  $projects,
  enterProject,
  projectIdForCwd,
  requestStartWorkSession
} from '@/store/projects'
import {
  $sessions,
  getRememberedRoute,
  getRememberedSessionId,
  rememberedSessionProfile,
  setRememberedRoute,
  setRememberedSessionId
} from '@/store/session'
import { onSessionsChanged } from '@/store/session-sync'
import { openUpdatesWindow, startUpdatePoller, stopUpdatePoller } from '@/store/updates'
import { isSecondaryWindow } from '@/store/windows'

import { requestComposerFocus, requestComposerInsert } from '../../chat/composer/focus'
import { appViewForPath, isOverlayView, NEW_CHAT_ROUTE, sessionRoute } from '../../routes'

/** Resolve hermes://chat/new?project= id or slug/name → absolute cwd from live project caches. */
function resolveProjectCwd(projectKey: string): { cwd: string; projectId: null | string } | null {
  const key = projectKey.trim()
  if (!key) return null

  const lower = key.toLowerCase()
  for (const p of $projects.get()) {
    const id = String(p.id || '').trim()
    const name = String(p.name || '').trim()
    const slug = String(p.slug || name.toLowerCase().replace(/\s+/g, '-')).trim()
    if (id === key || name === key || slug === key || slug.toLowerCase() === lower || name.toLowerCase() === lower) {
      const folders = Array.isArray(p.folders) ? p.folders : []
      const folderPath = folders.map(f => String((f as { path?: string }).path || '').trim()).find(Boolean) || ''
      const cwd = String(p.primary_path || folderPath || '').trim()
      if (cwd) return { cwd, projectId: id || null }
    }
  }

  for (const node of $projectTree.get()) {
    const id = String(node.id || '').trim()
    const label = String(node.label || '').trim()
    const slug = label.toLowerCase().replace(/\s+/g, '-')
    if (id === key || label === key || slug === lower || label.toLowerCase() === lower) {
      const cwd = String(node.path || node.repos?.find(r => r.path)?.path || '').trim()
      if (cwd) return { cwd, projectId: id || null }
    }
  }

  return null
}

function sessionIdKnown(sessionId: string): boolean {
  const id = sessionId.trim()
  if (!id) return false
  return $sessions.get().some(s => String(s.id || '').trim() === id)
}

interface DesktopIntegrationsParams {
  chatOpen: boolean
  hasPreview: boolean
  locationPathname: string
  navigate: (to: string, options?: { replace?: boolean }) => void
  refreshSessions: () => Promise<unknown> | unknown
  resumeExhaustedSessionId: null | string
  routedSessionId: null | string
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
  locationPathname,
  navigate,
  refreshSessions,
  resumeExhaustedSessionId,
  routedSessionId,
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
  // Also binds deeplink sticky= slots once a real session mounts.
  useEffect(() => {
    if (routedSessionId) {
      setRememberedSessionId(
        routedSessionId,
        rememberedSessionProfile($sessions.get(), routedSessionId, $activeGatewayProfile.get())
      )
      const pendingSticky = takeStickyPending()
      if (pendingSticky) {
        writeStickySessionId(pendingSticky, routedSessionId)
      }
    }

    if (!isOverlayView(appViewForPath(locationPathname))) {
      setRememberedRoute(locationPathname)
    }
  }, [locationPathname, routedSessionId])

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

    if (route && route !== NEW_CHAT_ROUTE && !isOverlayView(appViewForPath(route))) {
      navigate(route, { replace: true })

      return
    }

    const last = getRememberedSessionId($activeGatewayProfile.get())

    if (last) {
      navigate(sessionRoute(last), { replace: true })
    }
  }, [locationPathname, navigate])

  useEffect(() => {
    if (!resumeExhaustedSessionId) {
      return
    }

    const owner = rememberedSessionProfile($sessions.get(), resumeExhaustedSessionId, $activeGatewayProfile.get())

    if (getRememberedSessionId(owner) === resumeExhaustedSessionId) {
      setRememberedSessionId(null, owner)
    }
  }, [resumeExhaustedSessionId])

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

  // hermes:// deep links:
  //  - blueprint/<name>?slots → reviewable /blueprint command in composer
  //  - chat/new?cwd=&profile=&project=&prompt=&sticky= → fresh (or sticky) session
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onDeepLink?.(payload => {
      if (!payload || !payload.kind) {
        return
      }

      // hermes://chat/new?cwd=…&profile=…&project=…&prompt=…&sticky=…
      // main delivers { kind: 'chat', name: 'new', params }
      if (payload.kind === 'chat' && payload.name === 'new') {
        const params = payload.params || {}
        const profile = (params.profile || '').trim()
        const prompt = (params.prompt || '').trim()
        const projectKey = (params.project || '').trim()
        const sticky = normalizeStickySlot(params.sticky)
        let cwd = (params.cwd || '').trim()
        let projectId: null | string = null

        if (!cwd && projectKey) {
          const hit = resolveProjectCwd(projectKey)
          if (hit) {
            cwd = hit.cwd
            projectId = hit.projectId
          }
        }

        if (cwd && !cwdLooksSane(cwd)) {
          console.warn('[deeplink] chat/new refused unsafe cwd', cwd)
          return
        }

        // sticky=<slot>: resume the same session instead of minting a new one.
        if (sticky) {
          const existing = readStickySessionId(sticky)
          if (existing) {
            const known = sessionIdKnown(existing)
            // Empty session cache (boot race) → trust id once; loaded+missing → clear.
            if (known || $sessions.get().length === 0) {
              navigate(sessionRoute(existing))
              requestComposerFocus('main')
              if (prompt) {
                requestComposerInsert(prompt, { mode: 'block', target: 'main' })
              }
              return
            }
            clearStickySessionId(sticky)
          }
          // First open (or cleared): create below and bind when routedSessionId lands.
          setStickyPending(sticky)
        }

        if (profile) {
          $newChatProfile.set(profile)
          void ensureGatewayProfile(profile)
          // Prefer newSessionInProfile when we do not yet have a cwd path — it
          // forces fresh. When cwd is set, requestStartWorkSession owns the
          // fresh draft with workspace target (avoids double fresh race).
          if (!cwd) {
            newSessionInProfile(profile)
          } else {
            requestFreshSession()
          }
        } else if (!cwd) {
          requestFreshSession()
        }

        if (cwd) {
          const matched = projectId || projectIdForCwd(cwd)
          if (matched) {
            enterProject(matched)
          }
          // Same path as sidebar "new session in worktree/project"
          requestStartWorkSession(cwd, prompt || undefined)
        } else {
          navigate(NEW_CHAT_ROUTE)
          if (prompt) {
            requestComposerInsert(prompt, { mode: 'block', target: 'main' })
          }
        }

        requestComposerFocus('main')
        return
      }

      if (payload.kind !== 'blueprint' || !payload.name) {
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
  }, [navigate])

  // ⌘W via the macOS menu accelerator → close the focused tab; if nothing is
  // closeable, fall back to closing the window (so ⌘W still works as the
  // OS-standard window close, esp. secondary windows). The Win/Linux keyboard
  // path is the `view.closeTab` keybind (use-keybinds), sharing closeActiveTab.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onClosePreviewRequested?.(
      () => void closeActiveTab(id => navigate(sessionRoute(id)))
    )

    return () => unsubscribe?.()
  }, [navigate])

  // Another window mutated the shared session list -> re-pull the sidebar.
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    return onSessionsChanged(() => void refreshSessions())
  }, [refreshSessions])
}
