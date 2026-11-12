import { computed } from 'nanostores'

import { titlebarBandHeight } from '@/app/shell/titlebar'
import { useMediaQuery } from '@/hooks/use-media-query'
import { hostHas } from '@/lib/host-capabilities'
import { $touchPointer, TOUCH_POINTER_QUERY } from '@/lib/touch-interaction'
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

/** This window's band including touch sizing (`$touchPointer` is already browser-host only, the
 *  same host `touchSizedControls` names), for the root custom property below. */
export const $titlebarBandHeight = computed([$portalShell, $touchPointer], (portal, touch) =>
  titlebarBandHeight({ portal, touch })
)

/** Publishes the band on <html> as `--titlebar-band-height`. The shell scopes `--titlebar-height`
 *  (zero inside content, the band on its own chrome), so body-portalled surfaces — find bar, toast
 *  stack, palette HUDs — read this root value instead of guessing a classic fallback. Called at boot
 *  beside `installShellAttribute`, so it follows runtime shell flips. */
export function installTitlebarBandProperty(root: HTMLElement = document.documentElement): () => void {
  return $titlebarBandHeight.subscribe(height => root.style.setProperty('--titlebar-band-height', `${height}px`))
}
