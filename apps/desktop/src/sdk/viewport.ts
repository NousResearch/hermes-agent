// `host.state.viewport`: window geometry + the app's responsive posture.

import { atom } from 'nanostores'

import { $narrowViewport } from '@/components/pane-shell/tree/store'

/** Window geometry + the app's responsive posture, one readonly rect. */
export interface ViewportRect {
  width: number
  height: number
  /** Below the app's sidebar-collapse breakpoint (rails become overlays). */
  narrow: boolean
}

const readViewport = (): ViewportRect => ({
  width: typeof window === 'undefined' ? 0 : window.innerWidth,
  height: typeof window === 'undefined' ? 0 : window.innerHeight,
  narrow: $narrowViewport.get()
})

export const $viewport = atom<ViewportRect>(readViewport())

if (typeof window !== 'undefined') {
  const refresh = () => $viewport.set(readViewport())
  window.addEventListener('resize', refresh)
  $narrowViewport.listen(refresh)
}
