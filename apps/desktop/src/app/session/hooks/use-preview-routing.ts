import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { desktopFsCacheKey } from '@/lib/desktop-fs'
import { gatewayEventCompletedFileDiff } from '@/lib/gateway-events'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import {
  $previewTarget,
  $sessionPreviewRegistry,
  $webPreviewTabs,
  beginPreviewServerRestart,
  completePreviewServerRestart,
  getSessionPreviewRecord,
  previewWorkspaceScopeId,
  progressPreviewServerRestart,
  requestPreviewReload,
  setPreviewTarget,
  setPreviewWorkspaceScope,
  setSessionPreviewTarget
} from '@/store/preview'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $currentCwd } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

type EventHandler = (event: RpcEvent) => void

const MAX_ROUTED_PREVIEW_TOOL_IDS = 512

function rememberRoutedPreviewToolId(ids: Set<string>, toolId: string) {
  ids.add(toolId)

  while (ids.size > MAX_ROUTED_PREVIEW_TOOL_IDS) {
    const oldest = ids.values().next().value

    if (typeof oldest !== 'string') {
      break
    }

    ids.delete(oldest)
  }
}

interface PreviewRoutingOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  baseHandleGatewayEvent: EventHandler
  currentCwd: string
  currentView: string
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  routedSessionId: string | null
  selectedStoredSessionId: string | null
}

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

function activePreviewSessionId(routedSessionId: string | null, selectedStoredSessionId: string | null): string {
  return selectedStoredSessionId || routedSessionId || ''
}

export function usePreviewRouting({
  activeSessionIdRef,
  baseHandleGatewayEvent,
  currentCwd,
  currentView,
  requestGateway,
  routedSessionId,
  selectedStoredSessionId
}: PreviewRoutingOptions) {
  const routedPreviewToolIdsRef = useRef(new Set<string>())
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const connection = useStore($connection)
  const previewRegistry = useStore($sessionPreviewRegistry)
  const webPreviewTabs = useStore($webPreviewTabs)
  const previewSessionId = activePreviewSessionId(routedSessionId, selectedStoredSessionId)
  const previewConnectionKey = desktopFsCacheKey(connection)
  const previewScopeId = previewWorkspaceScopeId(previewSessionId, previewConnectionKey, activeGatewayProfile)
  const previewRouteRef = useRef({ currentView, scopeId: previewScopeId, sessionId: previewSessionId })
  previewRouteRef.current = { currentView, scopeId: previewScopeId, sessionId: previewSessionId }

  useEffect(() => {
    setPreviewWorkspaceScope(previewSessionId, previewConnectionKey, activeGatewayProfile)
    routedPreviewToolIdsRef.current.clear()
  }, [activeGatewayProfile, previewConnectionKey, previewSessionId])

  // Restore a *user-opened* preview when its session becomes active. Tool
  // results no longer auto-register/open a preview — the inline preview card in
  // the tool row is the only entry point, so HTML artifacts never pop the rail
  // open on their own.
  useEffect(() => {
    if (currentView !== 'chat' || !previewSessionId) {
      setPreviewTarget(null)

      return
    }

    const record = getSessionPreviewRecord(previewSessionId)

    const representedByWebTab = record ? webPreviewTabs.some(tab => tab.target.url === record.normalized.url) : false

    setPreviewTarget(record && !representedByWebTab ? record.normalized : null)
  }, [currentView, previewRegistry, previewSessionId, webPreviewTabs])

  const restartPreviewServer = useCallback(
    async (url: string, context?: string) => {
      const sessionId = activeSessionIdRef.current

      if (!sessionId) {
        throw new Error('No active session for background restart')
      }

      const cwd = $currentCwd.get() || currentCwd || ''

      const result = await requestGateway<{ task_id?: string }>('preview.restart', {
        context: context || undefined,
        cwd: cwd || undefined,
        session_id: sessionId,
        url
      })

      const taskId = result.task_id || ''

      if (!taskId) {
        throw new Error('Background restart did not return a task id')
      }

      beginPreviewServerRestart(taskId, url)

      return taskId
    },
    [activeSessionIdRef, currentCwd, requestGateway]
  )

  const handleDesktopGatewayEvent = useCallback<EventHandler>(
    event => {
      const payload = asRecord(event.payload)
      const args = asRecord(payload.args)

      const routingOnlyPreviewCompletion =
        event.type === 'tool.complete' &&
        payload.name === 'read_file' &&
        args.preview === true &&
        payload.preview_success === true &&
        !Object.prototype.hasOwnProperty.call(payload, 'result')

      if (!routingOnlyPreviewCompletion) {
        baseHandleGatewayEvent(event)
      }

      if (event.type === 'preview.restart.complete') {
        const { task_id, text } = asRecord(event.payload)

        if (typeof task_id === 'string' && task_id) {
          completePreviewServerRestart(task_id, typeof text === 'string' ? text : '')
        }
      } else if (event.type === 'preview.restart.progress') {
        const { task_id, text } = asRecord(event.payload)

        if (typeof task_id === 'string' && task_id) {
          progressPreviewServerRestart(task_id, typeof text === 'string' ? text : '')
        }
      }

      const route = previewRouteRef.current

      if (event.session_id && event.session_id !== route.sessionId) {
        return
      }

      const toolId = typeof payload.tool_id === 'string' ? payload.tool_id : ''
      const path = typeof args.path === 'string' ? args.path.trim() : ''

      if (
        route.currentView === 'chat' &&
        route.sessionId &&
        event.type === 'tool.complete' &&
        event.session_id === route.sessionId &&
        payload.name === 'read_file' &&
        args.preview === true &&
        payload.preview_success === true &&
        path &&
        toolId &&
        !routedPreviewToolIdsRef.current.has(toolId)
      ) {
        rememberRoutedPreviewToolId(routedPreviewToolIdsRef.current, toolId)
        const cwd = $currentCwd.get() || currentCwd || undefined
        const requestedSessionId = route.sessionId
        const requestedScopeId = route.scopeId

        void normalizeOrLocalPreviewTarget(path, cwd)
          .then(target => {
            const currentRoute = previewRouteRef.current

            if (
              !target ||
              target.kind !== 'file' ||
              currentRoute.currentView !== 'chat' ||
              currentRoute.sessionId !== requestedSessionId ||
              currentRoute.scopeId !== requestedScopeId
            ) {
              return
            }

            setSessionPreviewTarget(
              requestedSessionId,
              { ...target, renderMode: 'source', source: path },
              'agent-request',
              path
            )
          })
          .catch(() => {
            // Preview requests are optional UI affordances; normalization can
            // fail on a disconnected remote backend without affecting the read.
          })
      }

      // Only refresh an already-open live preview when a file changes; never
      // open one unprompted. (Preview links are surfaced from the tool row into
      // the status stack — see tool-fallback.tsx.)
      if ($previewTarget.get()?.kind === 'url' && gatewayEventCompletedFileDiff(event)) {
        requestPreviewReload()
      }
    },
    [baseHandleGatewayEvent, currentCwd]
  )

  return { handleDesktopGatewayEvent, restartPreviewServer }
}
