import { useMediaQuery } from './use-media-query'

// Mobile-standalone (iOS WebView) always counts as mobile so the Sidebar
// renders as a Sheet drawer, not a docked column. The flag is set by mobile's
// main.tsx after this module loads, so read it dynamically per call.
export const useIsMobile = () => {
  const narrow = useMediaQuery(`(max-width: ${768 / 16 - 1 / 16}rem)`)
  return (
    narrow ||
    (typeof window !== 'undefined' &&
      Boolean((window as { __HERMES_MOBILE_STANDALONE__?: boolean }).__HERMES_MOBILE_STANDALONE__))
  )
}
