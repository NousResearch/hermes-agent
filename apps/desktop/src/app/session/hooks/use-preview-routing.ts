import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect } from 'react'

import { gatewayEventCompletedFileDiff } from '@/lib/gateway-events'
import { extractGeneratedArtifactTargetsFromToolPayload } from '@/lib/generated-artifacts'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import {
  $previewTarget,
  $sessionPreviewRegistry,
  beginPreviewServerRestart,
  completePreviewServerRestart,
  getSessionPreviewRecord,
  progressPreviewServerRestart,
  requestPreviewReload,
  setCurrentSessionPreviewTarget,
  setPreviewTarget
} from '@/store/preview'
import { recordPreviewArtifact } from '@/store/preview-status'
import { $currentCwd } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

type EventHandler = (event: RpcEvent) => void

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

function generatedToolTargets(payload: Record<string, unknown>): string[] {
  const resultTargets = extractGeneratedArtifactTargetsFromToolPayload(payload.result)

  if (resultTargets.length > 0) {
    return resultTargets
  }

  const topLevelTargets = extractGeneratedArtifactTargetsFromToolPayload({
    file: payload.file,
    filepath: payload.filepath,
    path: payload.path,
    preview: payload.preview,
    result_text: payload.result_text,
    summary: payload.summary,
    target: payload.target,
    url: payload.url
  })

  if (topLevelTargets.length > 0) {
    return topLevelTargets
  }

  const name = typeof payload.name === 'string' ? payload.name : ''

  return name === 'write_file' || name === 'edit_file' || name === 'patch'
    ? extractGeneratedArtifactTargetsFromToolPayload(payload.args)
    : []
}

function activePreviewSessionId(
  activeSessionIdRef: MutableRefObject<string | null>,
  routedSessionId: string | null,
  selectedStoredSessionId: string | null
): string {
  return selectedStoredSessionId || routedSessionId || activeSessionIdRef.current || ''
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
  const previewRegistry = useStore($sessionPreviewRegistry)
  const previewSessionId = activePreviewSessionId(activeSessionIdRef, routedSessionId, selectedStoredSessionId)

  // Restore the session preview when its session becomes active.
  useEffect(() => {
    if (currentView !== 'chat' || !previewSessionId) {
      setPreviewTarget(null)

      return
    }

    const record = getSessionPreviewRecord(previewSessionId)

    setPreviewTarget(record?.normalized ?? null)
  }, [currentView, previewRegistry, previewSessionId])

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
      baseHandleGatewayEvent(event)

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

      if (event.session_id && event.session_id !== activeSessionIdRef.current) {
        return
      }

      // Refresh an already-open live preview when a file changes; generated
      // artifacts detected below may also open the preview rail for the turn.
      if ($previewTarget.get()?.kind === 'url' && gatewayEventCompletedFileDiff(event)) {
        requestPreviewReload()
      }

      if (event.type === 'tool.complete') {
        const targets = generatedToolTargets(asRecord(event.payload))
        const target = targets[0]

        if (target) {
          const sessionId = activePreviewSessionId(activeSessionIdRef, routedSessionId, selectedStoredSessionId)
          const cwd = $currentCwd.get() || currentCwd || ''

          recordPreviewArtifact(sessionId, target, cwd)

          void normalizeOrLocalPreviewTarget(target, cwd || undefined).then(preview => {
            if (preview && (!event.session_id || event.session_id === activeSessionIdRef.current)) {
              setCurrentSessionPreviewTarget(preview, 'tool-result', target)
            }
          })
        }
      }
    },
    [activeSessionIdRef, baseHandleGatewayEvent, currentCwd, routedSessionId, selectedStoredSessionId]
  )

  return { handleDesktopGatewayEvent, restartPreviewServer }
}
