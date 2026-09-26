import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Monitor } from '@/lib/icons'
import { $petOverlayActive, popInPet, popOutDesktopOrb } from '@/store/pet-overlay'

import { desktopOrbCopy } from './desktop-orb-copy'
import { $desktopOrbMode } from './desktop-orb-state'

export function DesktopOrbToggle() {
  const { locale } = useI18n()
  const active = useStore($petOverlayActive)
  const orb = useStore($desktopOrbMode)

  if (!window.hermesDesktop?.petOverlay) {return null}
  const shown = active && orb

  return (
    <Button aria-pressed={shown} onClick={() => (shown ? popInPet() : popOutDesktopOrb())} variant="secondary">
      <Monitor />
      {shown ? desktopOrbCopy[locale].hide : desktopOrbCopy[locale].show}
    </Button>
  )
}
