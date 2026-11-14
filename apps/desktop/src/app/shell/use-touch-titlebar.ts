import { computed } from 'nanostores'

import {
  $shortViewport,
  $widthClass,
  COMPACT_QUERY as COMPACT_BAND_QUERY,
  SHORT_QUERY
} from '@/app/redesign/breakpoints'
import { portalBandLeadingInset, titlebarBandHeight } from '@/app/shell/titlebar'
import { useMediaQuery } from '@/hooks/use-media-query'
import { hostHas } from '@/lib/host-capabilities'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { $touchPointer, TOUCH_POINTER_QUERY } from '@/lib/touch-interaction'
import { $connection } from '@/store/session'
import { $portalShell, usePortalShell } from '@/store/shell-variant'

/** Browser touch chrome needs a 44px band, not Electron's compact 34px band. */
export function useTouchTitlebar(): boolean {
  const touch = useMediaQuery(TOUCH_POINTER_QUERY)

  return hostHas('touchSizedControls') && touch
}

/** This window's titlebar band (see titlebarBandHeight): touch sizing, and on a compact browser
 *  window the redesign's shorter band. */
export function useTitlebarHeight(): number {
  const portal = usePortalShell()
  const touch = useTouchTitlebar()
  const narrow = useMediaQuery(COMPACT_BAND_QUERY)
  const short = useMediaQuery(SHORT_QUERY)

  return titlebarBandHeight({ portal, touch, compact: narrow || short, browserHost: isBrowserHostedDesktop() })
}

/** The band body-portalled and event-time readers place against. Classic's readers (find bar, toast
 *  stack, floating HUDs, pets, floating panes) have always used its compact band, touch or not, as on
 *  main; only the redesign's band follows touch sizing (`$touchPointer` is already browser-host only,
 *  the same host `touchSizedControls` names) and the compact class. */
export const $titlebarBandHeight = computed(
  [$portalShell, $touchPointer, $widthClass, $shortViewport],
  (portal, touch, width, short) =>
    titlebarBandHeight({
      portal,
      touch: portal && touch,
      compact: width === 'compact' || short,
      browserHost: isBrowserHostedDesktop()
    })
)

/** The band outside React (geometry read at event time). */
export function titlebarHeightNow(): number {
  return $titlebarBandHeight.get()
}

/** Where the redesigned band's leading cluster starts (portalBandLeadingInset), from the live
 *  window chrome facts: the titlebar clusters and every page's Back read it. */
export const $portalBandStart = computed([$connection, $widthClass, $portalShell], (connection, width, portal) =>
  portalBandLeadingInset(
    connection?.windowButtonPosition,
    Boolean(connection?.isFullscreen),
    undefined,
    portal && width === 'compact'
  )
)

/** Publishes the band on <html> as `--titlebar-band-height` and the redesign's leading cluster start
 *  as `--portal-band-start`. The shell scopes `--titlebar-height` (zero inside content, the band on its
 *  own chrome), so body-portalled surfaces — find bar, toast stack, palette HUDs, page toolbars — read these root
 *  values instead of guessing a classic fallback. Called at boot beside `installShellAttribute`, so
 *  they follow runtime shell flips and fullscreen changes. */
export function installTitlebarBandProperty(root: HTMLElement = document.documentElement): () => void {
  const stopHeight = $titlebarBandHeight.subscribe(height =>
    root.style.setProperty('--titlebar-band-height', `${height}px`)
  )

  const stopStart = $portalBandStart.subscribe(start => root.style.setProperty('--portal-band-start', `${start}px`))

  return () => {
    stopHeight()
    stopStart()
  }
}
