import { useEffect, useState } from 'react'

import type { DesktopOrbState } from '@/app/jarvis/desktop-orb-state'
import { publishMicLevel } from '@/store/voice-level'

import { OrbOverlay } from './orb-overlay'
import { PetOverlayApp } from './pet-overlay-app'

export function OverlaySurface() {
  const [orb, setOrb] = useState<DesktopOrbState | null | undefined>(undefined)
  useEffect(() => {
    const api = window.hermesDesktop?.petOverlay

    const off = api?.onState(payload => {
      publishMicLevel(payload.audioLevel ?? 0)
      const next = payload.orb ?? null
      setOrb(previous => (JSON.stringify(previous) === JSON.stringify(next) ? previous : next))
    })

    api?.control({ type: 'ready' })

    return off
  }, [])

  if (orb === undefined) {return null}

  return orb ? <OrbOverlay state={orb} /> : <PetOverlayApp />
}
