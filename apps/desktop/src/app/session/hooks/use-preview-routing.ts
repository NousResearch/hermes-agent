import { useCallback } from 'react'

import { gatewayEventCompletedFileDiff } from '@/lib/gateway-events'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import {
  $previewTabs,
  beginPreviewServerRestart,
  completePreviewServerRestart,
  openPreview,
  progressPreviewServerRestart,
  requestPreviewReload
} from '@/store/preview'
import { $currentCwd } from '@/store/session'
import { $focusedRuntimeId } from '@/store/session-states'
import type { RpcEvent } from '@/types/hermes'

type EventHandler = (event: RpcEvent) => void

interface PreviewRoutingOptions {
  baseHandleGatewayEvent: EventHandler
  currentCwd: string
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

export function usePreviewRouting({ baseHandleGatewayEvent, currentCwd, requestGateway }: PreviewRoutingOptions) {
  const restartPreviewServer = useCallback(
    async (url: string, context?: string) => {
      const sessionId = $focusedRuntimeId.get()

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
    [currentCwd, requestGateway]
  )

  const handleDesktopGatewayEvent = useCallback<EventHandler>(
    event => {
      baseHandleGatewayEvent(event)

      if (event.type === 'preview.open') {
        // Agent-driven open in response to an explicit user request ("show
        // cnn.com in the preview pane"). Honor it only for the session the user
        // is actually looking at — a background turn must not yank the pane open
        // (see desktop AGENTS.md: offer, don't hijack). That session is the
        // focused one, which is a TILE's runtime whenever a tile is fronted, not
        // the primary chat's. Routes through the same normalizer as the file
        // browser so URLs, localhost, and file paths all resolve correctly.
        const { url, label } = asRecord(event.payload)
        const target = typeof url === 'string' ? url.trim() : ''

        if (target && (!event.session_id || event.session_id === $focusedRuntimeId.get())) {
          void normalizeOrLocalPreviewTarget(target, $currentCwd.get() || currentCwd || undefined).then(resolved => {
            if (resolved) {
              const trimmedLabel = typeof label === 'string' ? label.trim() : ''
              openPreview(trimmedLabel ? { ...resolved, label: trimmedLabel } : resolved, 'tool-result')
            }
          })
        }

        return
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

      if (event.session_id && event.session_id !== $focusedRuntimeId.get()) {
        return
      }

      // Only refresh an already-open live preview when a file changes; never
      // open one unprompted. (Preview links are surfaced from the tool row into
      // the status stack — see tool-fallback.tsx.)
      if ($previewTabs.get().some(tab => tab.target.kind === 'url') && gatewayEventCompletedFileDiff(event)) {
        requestPreviewReload()
      }
    },
    [baseHandleGatewayEvent, currentCwd]
  )

  return { handleDesktopGatewayEvent, restartPreviewServer }
}
