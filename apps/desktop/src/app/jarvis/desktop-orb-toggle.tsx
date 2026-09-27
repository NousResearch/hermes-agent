import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Monitor } from '@/lib/icons'
import { $petOverlayActive, popOutDesktopOrb } from '@/store/pet-overlay'

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
  const label = shown ? (locale === 'pl' ? 'Wróć do kuli' : 'Return to orb') : desktopOrbCopy[locale].show

  return (
    <Button
      aria-label={label}
      aria-pressed={shown}
      className={compact ? 'size-10 min-h-10 min-w-10 rounded-full' : 'min-h-11 rounded-full border border-(--ui-accent)/30 bg-(--ui-accent)/12 px-5 font-semibold text-(--ui-text-primary) shadow-sm hover:bg-(--ui-accent)/20'}
      onClick={() => shown
        ? window.hermesDesktop?.petOverlay?.control({ type: 'toggle-app' })
        : popOutDesktopOrb(() => window.hermesDesktop?.petOverlay?.control({ type: 'toggle-app' }))}
      size={compact ? 'icon' : 'default'}
      title={label}
      variant="secondary"
    >
      <Monitor />
      {compact ? null : label}
    </Button>
  )
}
