import { useEffect } from 'react'

import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'

// #40077: rendering is routed through ANGLE's SwiftShader backend on affected
// NVIDIA drivers to avoid a GPU-process crash — full GPU acceleration removed
// for the whole app. Same shape as RemoteDisplayBanner: surfaces as a
// persistent toast so the tradeoff isn't silent (previously only a
// console.log into desktop.log). The stable `id` collapses repeat mounts
// (HUD/popout/main renderers, a GPU-crash reload) into one toast instead of
// stacking undismissable duplicates.
export function NvidiaEglFallbackBanner() {
  useEffect(() => {
    void window.hermesDesktop?.getNvidiaEglFallbackReason?.().then(reason => {
      if (reason) {
        notify({
          id: 'nvidia-egl-fallback',
          durationMs: 0,
          kind: 'info',
          message: translateNow('nvidiaEglFallbackBanner.message', reason),
          placement: 'default'
        })
      }
    })
  }, [])

  return null
}
