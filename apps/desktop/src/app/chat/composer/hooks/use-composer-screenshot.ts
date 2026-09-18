import { useLayoutEffect, useRef } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { useI18n } from '@/i18n'
import { $activeGatewayRoute } from '@/store/gateway'
import { notify } from '@/store/notifications'

import { getActiveComposer, getVisibleComposerSurfaceId } from '../focus'
import { useComposerScope, useComposerSurfaceId } from '../scope'
import type { ChatBarProps } from '../types'

interface ScreenshotComposerOptions {
  sessionKey: string | null
  focusKey?: string | null
  onAttachImageBlob: ChatBarProps['onAttachImageBlob']
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
    const offRequest = api.onRequest(requestId => {
      if (busy || getActiveComposer() !== scope.target || getVisibleComposerSurfaceId(scope.target) !== surfaceId) {
        return
      }
      const attach = latest.current.onAttachImageBlob
      if (!attach) {
        return
      }

      const capturedGeneration = generation
      const isCurrent = () => mounted && capturedGeneration === generation
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
    })

    return () => {
      mounted = false
      offRoute()
      offStatus()
      offRequest()
    }
  }, [sessionKey, focusKey, visible, scope.target, surfaceId])
}
