import { useEffect } from 'react'

import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'

// #40077: rendering is routed through ANGLE's SwiftShader backend on affected
// NVIDIA drivers to avoid a GPU-process crash — full GPU acceleration removed
// for the whole app. Same shape as RemoteDisplayBanner: surfaces once per
// launch as a persistent toast so the tradeoff isn't silent (previously only
// a console.log into desktop.log).
export function NvidiaEglFallbackBanner() {
  useEffect(() => {
    void window.hermesDesktop?.getNvidiaEglFallbackReason?.().then(reason => {
      if (reason) {
        notify({
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
