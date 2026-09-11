import { useMediaQuery } from './use-media-query'

/** Narrower than this and the sidebar stops being a docked column: it renders
 *  as a floating Sheet over the content instead (components/ui/sidebar.tsx).
 *  Anything that sizes a window around a docked sidebar has to clear it. */
export const DOCKED_SIDEBAR_MIN_PX = 768

export const useIsMobile = () => useMediaQuery(`(max-width: ${(DOCKED_SIDEBAR_MIN_PX - 1) / 16}rem)`)
