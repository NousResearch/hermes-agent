// `host.state.viewport`: window geometry + the app's responsive posture.

import { atom } from 'nanostores'

import { $widthClass } from '@/app/redesign/breakpoints'

/** Window geometry + the app's responsive posture, one readonly rect. */
export interface ViewportRect {
  width: number
  height: number
  /** Below the sidebar dock point (640px), in both shells. The redesign's own "below wide" layout
   *  (sheets from 640 to 1023px, the pane tree's `$narrowViewport`) is shell chrome, not this
   *  contract: a plugin that adapts its layout reads the same breakpoint whichever shell hosts it. */
  narrow: boolean
}

const readViewport = (): ViewportRect => ({
  width: typeof window === 'undefined' ? 0 : window.innerWidth,
  height: typeof window === 'undefined' ? 0 : window.innerHeight,
  narrow: $widthClass.get() === 'compact'
})

export const $viewport = atom<ViewportRect>(readViewport())

if (typeof window !== 'undefined') {
  const refresh = () => $viewport.set(readViewport())
  window.addEventListener('resize', refresh)
  $widthClass.listen(refresh)
}
