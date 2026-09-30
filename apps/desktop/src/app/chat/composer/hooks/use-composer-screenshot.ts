import { useLayoutEffect, useRef } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { useI18n } from '@/i18n'
import { $activeGatewayRoute } from '@/store/gateway'
import { notify } from '@/store/notifications'
import { requestFreshSession } from '@/store/profile'

import { getActiveComposer, getVisibleComposerSurfaceId } from '../focus'
import { useComposerScope, useComposerSurfaceId } from '../scope'
import type { ChatBarProps } from '../types'

interface ScreenshotComposerOptions {
  sessionKey: string | null
  focusKey?: string | null
  onAttachImageBlob: ChatBarProps['onAttachImageBlob']
}

/** A `new-session` capture waiting for its fresh draft to mount the composer.
 *
 * The gesture delivers its request once, to the subscribers live at that
 * moment — but a `new-session` destination first resets the view to a fresh
 * draft, which re-runs this hook on the new sessionKey before the capture can
 * run. The outgoing instance parks the request here; the freshly mounted
 * instance (same window, same composer scope) claims it on mount. The main
 * process's single-use capture token is unchanged — only the window the
 * gesture selected may take it. */
interface PendingScreenshotRequest {
  requestId: string
  surfaceId: string
}

let pendingNewSessionRequest: PendingScreenshotRequest | null = null

function claimPendingNewSessionRequest(surfaceId: string): string | null {
  const pending = pendingNewSessionRequest

  if (pending && pending.surfaceId === surfaceId) {
    pendingNewSessionRequest = null

    return pending.requestId
  }

  return null
}

/** Capture into the exact draft that owned the gesture, even when the OS focus is elsewhere. */
export function useComposerScreenshot({ sessionKey, focusKey, onAttachImageBlob }: ScreenshotComposerOptions) {
  const scope = useComposerScope()
  const surfaceId = useComposerSurfaceId()
  const visible = usePaneVisible()
  const { t } = useI18n()
  const latest = useRef({ onAttachImageBlob, copy: t.settings.screenshot })
  latest.current = { onAttachImageBlob, copy: t.settings.screenshot }

  useLayoutEffect(() => {
    const api = window.hermesDesktop?.screenshot

    if (!api || !surfaceId || !visible) {
      return
    }

    let generation = 0
    let mounted = true
    let busy = false

    const offRoute = $activeGatewayRoute.listen(() => {
      generation += 1
    })

    const offStatus = api.onStatus(status => {
      if (!status.enabled) {
        generation += 1
      }
    })

    const captureIntoDraft = (
      requestId: string,
      isCurrent: () => boolean,
      attach: NonNullable<ChatBarProps['onAttachImageBlob']>
    ) => {
      const report = (message: string) => notify({ kind: 'error', title: latest.current.copy.enabledTitle, message })
      busy = true

      void (async () => {
        try {
          const result = await api.capture(requestId)

          if (!isCurrent()) {
            report(latest.current.copy.contextChanged)

            return
          }

          if (!result.ok) {
            report(latest.current.copy.captureFailed)

            return
          }

          // Reuse paste/image ingestion. Its final guard runs AFTER saving the
          // native bytes, before adding a chip, so a session swap cannot leak it.
          const blob = new Blob([new Uint8Array(result.png)], { type: 'image/png' })
          await attach(blob, isCurrent)

          if (!isCurrent()) {
            report(latest.current.copy.contextChanged)
          }
        } catch {
          report(latest.current.copy.captureFailed)
        } finally {
          busy = false
        }
      })()
    }

    const offRequest = api.onRequest((requestId, destination = 'current-draft') => {
      if (busy || getActiveComposer() !== scope.target || getVisibleComposerSurfaceId(scope.target) !== surfaceId) {
        return
      }

      const attach = latest.current.onAttachImageBlob

      if (!attach) {
        return
      }

      if (destination === 'new-session' && sessionKey !== null) {
        // Park the request: the fresh draft re-runs this hook with a new
        // sessionKey, and that instance claims it right after subscribing.
        pendingNewSessionRequest = { requestId, surfaceId }
        requestFreshSession()

        return
      }

      const capturedGeneration = generation
      captureIntoDraft(requestId, () => mounted && capturedGeneration === generation, attach)
    })

    // A new-session request parked by the pre-switch instance: this fresh
    // draft owns the gesture now. The busy guard is this instance's own —
    // it just mounted, so it cannot be mid-capture.
    const claimed = claimPendingNewSessionRequest(surfaceId)

    if (claimed && !busy && getActiveComposer() === scope.target) {
      const attach = latest.current.onAttachImageBlob

      if (attach) {
        captureIntoDraft(claimed, () => mounted, attach)
      }
    }

    return () => {
      mounted = false
      offRoute()
      offStatus()
      offRequest()
    }
  }, [sessionKey, focusKey, visible, scope.target, surfaceId])
}
