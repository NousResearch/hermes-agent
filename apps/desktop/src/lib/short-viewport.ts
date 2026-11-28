import { useSyncExternalStore } from 'react'

/** A short viewport: a phone with the on-screen keyboard up (the Webapp sets
 *  interactive-widget=resizes-content, so the layout height shrinks with it), or a
 *  landscape phone. Keep in step with the `max-height: 520px` block in styles.css,
 *  which unpins the sticky prompt and caps the composer input under the same condition. */
export const SHORT_VIEWPORT_QUERY = '(max-height: 520px)'

function subscribe(onChange: () => void): () => void {
  if (typeof window === 'undefined' || !window.matchMedia) {
    return () => {}
  }

  const query = window.matchMedia(SHORT_VIEWPORT_QUERY)

  query.addEventListener('change', onChange)

  return () => query.removeEventListener('change', onChange)
}

function snapshot(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(SHORT_VIEWPORT_QUERY).matches
}

export function useShortViewport(): boolean {
  return useSyncExternalStore(subscribe, snapshot, () => false)
}
