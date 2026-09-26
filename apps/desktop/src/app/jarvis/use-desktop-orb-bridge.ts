import { useEffect, useRef } from 'react'

import { useI18n } from '@/i18n'
import { initPetOverlayBridge, restorePetOverlay } from '@/store/pet-overlay'

import { $desktopOrbConnection, $desktopOrbMode } from './desktop-orb-state'

interface OrbVoiceBridge {
  primary: boolean
  active: boolean
  connected: boolean
  start: () => void
  stop: () => void
}

export function useDesktopOrbBridge(options: OrbVoiceBridge): void {
  const { locale } = useI18n()
  const current = useRef(options)
  current.current = options
  useEffect(() => {
    if (!options.primary) {
      return
    }

    $desktopOrbConnection.set({ active: options.active, connected: options.connected, locale })
  }, [options.primary, options.active, options.connected, locale])
  useEffect(() => {
    if (!options.primary) {
      return
    }

    const offBridge = initPetOverlayBridge()

    if ($desktopOrbMode.get()) {
      restorePetOverlay()
    }

    const offControl = window.hermesDesktop?.petOverlay?.onControl(control => {
      if (control.type !== 'orb-toggle-voice') {
        return
      }

      const state = current.current

      if (state.active) {
        state.stop()
      } else if (state.connected) {
        state.start()
      }
    })

    return () => {
      offControl?.()
      offBridge()
    }
  }, [options.primary])
}
