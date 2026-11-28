import { useSyncExternalStore } from 'react'

import { availableViewportHeight, subscribeAvailableViewportHeight } from './browser-viewport'

/** A short viewport: a phone with the on-screen keyboard up, or a landscape phone. Keep in step with the
 *  `max-height: 520px` block in styles.css. */
export const SHORT_VIEWPORT_QUERY = '(max-height: 520px)'
export const SHORT_VIEWPORT_MAX_HEIGHT = 520

function layoutIsShort(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(SHORT_VIEWPORT_QUERY).matches
}

/** The keyboard shrinks the LAYOUT height only where `interactive-widget=resizes-content` works (the Webapp sets
 *  it). Where it does not — iOS Safari without it — only `visualViewport` shrinks, so the same breakpoint is read
 *  off the visible height, pinch zoom masked out. Either signal saying "short" is enough. */
function isShort(): boolean {
  if (layoutIsShort()) {
    return true
  }

  const height = availableViewportHeight()

  return height > 0 && height <= SHORT_VIEWPORT_MAX_HEIGHT
}

function subscribe(onChange: () => void): () => void {
  if (typeof window === 'undefined') {
    return () => {}
  }

  const cleanups: Array<() => void> = []

  if (window.matchMedia) {
    const query = window.matchMedia(SHORT_VIEWPORT_QUERY)

    query.addEventListener('change', onChange)
    cleanups.push(() => query.removeEventListener('change', onChange))
  }

  cleanups.push(subscribeAvailableViewportHeight(onChange))

  return () => cleanups.forEach(cleanup => cleanup())
}

export function useShortViewport(): boolean {
  return useSyncExternalStore(subscribe, isShort, () => false)
}
