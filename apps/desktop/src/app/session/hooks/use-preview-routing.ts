import type { GatewayEvent } from '@hermes/shared'
import { useCallback } from 'react'

import { gatewayEventCompletedFileDiff } from '@/lib/gateway-events'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { reachablePreviewUrl } from '@/lib/preview-reach'
import {
  $previewTabs,
  beginPreviewServerRestart,
  closeAgentPreview,
  completePreviewServerRestart,
  openPreview,
  progressPreviewServerRestart,
  renderedHtmlTarget,
  requestPreviewReload
} from '@/store/preview'
import type { PreviewOwner } from '@/store/preview-ownership'
import { $activeSessionId, $currentCwd } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import {
  $focusedRuntimeId,
  $sessionTiles,
  previewScopeForRuntime,
  storedSessionIdForRuntimeId
} from '@/store/session-states'

type EventHandler = (event: GatewayEvent) => void

interface PreviewRoutingOptions {
  baseHandleGatewayEvent: EventHandler
  currentCwd: string
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

function sessionIsOnScreen(sessionId: string): boolean {
  return (
    sessionId === $focusedRuntimeId.get() ||
    sessionId === $activeSessionId.get() ||
    $sessionTiles.get().some(tile => tile.runtimeId === sessionId)
  )
}

/** The stored id whose drawer an agent's preview event belongs to: the session
 *  that ran the tool, not whichever one holds focus (#73890). A runtime with no
 *  stored id yet is a fresh draft, whose tabs are ownerless until adopted. */
function previewOwnerForEvent(sessionId: string | undefined): null | string {
  return sessionId ? storedSessionIdForRuntimeId(sessionId) : $focusedStoredSessionId.get()
}

/** The full identity an agent's close acts for: its stored id, the runtime
 *  (its pending tabs) and its profile (the only pins it may close). */
function previewCloserForEvent(sessionId: string | undefined): PreviewOwner {
  return sessionId
    ? { profile: previewScopeForRuntime(sessionId), runtimeId: sessionId, sessionId: previewOwnerForEvent(sessionId) }
    : previewOwnerForEvent(sessionId)
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
        // cnn.com in the preview pane"). Honor it for any session that's ON
        // SCREEN — the primary chat or an open tile — not only the focused
        // one: the turn's window routing already scoped the event to this
        // window, and gating on focus made the open silently vanish whenever
        // the user's click had moved focus to a different zone by the time
        // the tool ran (an "open reddit" they explicitly asked for). A
        // session that is NOT visible anywhere still can't yank the pane
        // open (offer, don't hijack). Routes through the same normalizer as
        // the file browser so URLs, localhost, and file paths all resolve.
        const { url, label } = asRecord(event.payload)
        const target = typeof url === 'string' ? url.trim() : ''

        if (target && (!event.session_id || sessionIsOnScreen(event.session_id))) {
          void normalizeOrLocalPreviewTarget(target, $currentCwd.get() || currentCwd || undefined).then(
            async resolved => {
              if (!resolved) {
                return
              }

              const trimmedLabel = typeof label === 'string' ? label.trim() : ''
              // The agent's loopback is the GATEWAY's loopback. Give the pane a
              // URL this machine can load, keeping the original as the label so
              // the user still sees the address the agent named.
              const url = resolved.kind === 'url' ? await reachablePreviewUrl(resolved.url) : resolved.url
              const reached = url === resolved.url ? resolved : { ...resolved, label: resolved.label || target, url }

              openPreview(
                renderedHtmlTarget(trimmedLabel ? { ...reached, label: trimmedLabel } : reached),
                previewOwnerForEvent(event.session_id),
                // The runtime that ran the tool, should its stored id lag.
                event.session_id || undefined,
                // Its profile: another profile's pinned Browser is not its to navigate.
                event.session_id ? previewScopeForRuntime(event.session_id) : undefined
              )
            }
          )
        }

        return
      }

      if (event.type === 'preview.close') {
        // Agent-driven close via close_preview. Same on-screen gate as open:
        // a session the user can see may tidy the pane it opened; a hidden
        // background turn must not dismiss the user's preview.
        const { url } = asRecord(event.payload)
        const target = typeof url === 'string' ? url.trim() : ''

        if (event.session_id && !sessionIsOnScreen(event.session_id)) {
          return
        }

        const owner = previewCloserForEvent(event.session_id)

        if (!target) {
          closeAgentPreview(owner, [])

          return
        }

        void normalizeOrLocalPreviewTarget(target, $currentCwd.get() || currentCwd || undefined).then(
          async resolved => {
            const candidates = [target]

            if (resolved) {
              candidates.push(resolved.source, resolved.url)

              if (resolved.kind === 'url') {
                candidates.push(await reachablePreviewUrl(resolved.url))
              }
            }

            closeAgentPreview(owner, candidates)
          }
        )

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
