import { TITLEBAR_HEIGHT, TOUCH_TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { useMediaQuery } from '@/hooks/use-media-query'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { TOUCH_POINTER_QUERY } from '@/lib/touch-interaction'

/** Browser touch chrome needs a 44px band, not Electron's compact 34px band. */
export function useTouchTitlebar(): boolean {
  const touch = useMediaQuery(TOUCH_POINTER_QUERY)

  return isBrowserHostedDesktop() && touch
}

export function useTitlebarHeight(): number {
  return useTouchTitlebar() ? TOUCH_TITLEBAR_HEIGHT : TITLEBAR_HEIGHT
}
