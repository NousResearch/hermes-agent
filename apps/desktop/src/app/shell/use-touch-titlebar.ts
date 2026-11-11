import { titlebarBandHeight } from '@/app/shell/titlebar'
import { useMediaQuery } from '@/hooks/use-media-query'
import { hostHas } from '@/lib/host-capabilities'
import { TOUCH_POINTER_QUERY } from '@/lib/touch-interaction'
import { $portalShell, usePortalShell } from '@/store/shell-variant'

/** Browser touch chrome needs a 44px band, not Electron's compact 34px band. */
export function useTouchTitlebar(): boolean {
  const touch = useMediaQuery(TOUCH_POINTER_QUERY)

  return hostHas('touchSizedControls') && touch
}

/** This window's titlebar band (see titlebarBandHeight). */
export function useTitlebarHeight(): number {
  const portal = usePortalShell()
  const touch = useTouchTitlebar()

  return titlebarBandHeight({ portal, touch })
}

/** The band outside React (geometry read at event time). */
export function titlebarHeightNow(): number {
  return titlebarBandHeight({ portal: $portalShell.get(), touch: false })
}
