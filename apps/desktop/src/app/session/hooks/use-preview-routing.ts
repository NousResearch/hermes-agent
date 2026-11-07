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
import { $activeSessionId, $currentCwd } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { visibleSessionIdentity } from '@/store/session-identity'
import { $focusedRuntimeId } from '@/store/session-states'

type EventHandler = (event: GatewayEvent) => void

interface PreviewRoutingOptions {
  baseHandleGatewayEvent: EventHandler
  currentCwd: string
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

function trimmedString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function previewIdentityForEvent(event: GatewayEvent) {
  return event.session_id
    ? visibleSessionIdentity({
        activeSessionId: $activeSessionId.get(),
        sessionId: event.session_id,
        source: event
      })
    : undefined
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
        const target = trimmedString(url)

        const owner = previewIdentityForEvent(event)

        if (target && (!event.session_id || owner)) {
          void normalizeOrLocalPreviewTarget(target, $currentCwd.get() || currentCwd || undefined).then(
            async resolved => {
              if (!resolved) {
                return
              }

              const trimmedLabel = trimmedString(label)
              // The agent's loopback is the GATEWAY's loopback. Give the pane a
              // URL this machine can load, keeping the original as the label so
              // the user still sees the address the agent named.
              const url = resolved.kind === 'url' ? await reachablePreviewUrl(resolved.url) : resolved.url
              const reached = url === resolved.url ? resolved : { ...resolved, label: resolved.label || target, url }

              openPreview(
                renderedHtmlTarget(trimmedLabel ? { ...reached, label: trimmedLabel } : reached),
                event.session_id ? owner!.sessionId : $focusedStoredSessionId.get(),
                // The runtime that ran the tool, should its stored id lag.
                owner?.runtimeId || event.session_id || undefined,
                // Its profile: another profile's pinned Browser is not its to navigate.
                owner?.profile
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
        const target = trimmedString(url)

        const identity = previewIdentityForEvent(event)

        if (event.session_id && !identity) {
          return
        }

        const owner = event.session_id ? identity! : $focusedStoredSessionId.get()

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
