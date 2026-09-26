import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Monitor } from '@/lib/icons'
import { $petOverlayActive, popInPet, popOutDesktopOrb } from '@/store/pet-overlay'

import { desktopOrbCopy } from './desktop-orb-copy'
import { $desktopOrbMode } from './desktop-orb-state'

export function DesktopOrbToggle({ compact = false }: { compact?: boolean }) {
  const { locale } = useI18n()
  const active = useStore($petOverlayActive)
  const orb = useStore($desktopOrbMode)

  if (!window.hermesDesktop?.petOverlay) {
    return null
  }

  const shown = active && orb

  return (
    <Button
      aria-label={shown ? desktopOrbCopy[locale].hide : desktopOrbCopy[locale].show}
      aria-pressed={shown}
      className={compact ? 'size-10 min-h-10 min-w-10 rounded-full' : undefined}
      onClick={() => (shown ? popInPet() : popOutDesktopOrb())}
      size={compact ? 'icon' : 'default'}
      title={shown ? desktopOrbCopy[locale].hide : desktopOrbCopy[locale].show}
      variant="secondary"
    >
      <Monitor />
      {compact ? null : shown ? desktopOrbCopy[locale].hide : desktopOrbCopy[locale].show}
    </Button>
  )
}
