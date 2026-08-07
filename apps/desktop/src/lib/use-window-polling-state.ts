import { useEffect, useState } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'

import { createRendererLoopPauseController } from './renderer-loop-pause'

/** Three-tier window state for polling decisions. */
export type WindowPollingState = 'active' | 'idle' | 'hidden'

/**
 * Derives the current polling tier from:
 * - document.visibilityState (tab hidden / minimized)
 * - window focus/blur (foreground vs background)
 * - Electron window state (minimized / occluded via hermesDesktop)
 * - Pane visibility (inactive tab in a tab stack: visibility:hidden)
 *
 * Priority: hidden > idle > active
 *   hidden: window minimized/occluded OR pane not visible (inactive tab)
 *   idle:   window visible but unfocused (user monitoring on side)
 *   active: window focused and pane visible
 */
export function useWindowPollingState(): WindowPollingState {
  const isPaneVisible = usePaneVisible()
  const [windowState, setWindowState] = useState<WindowPollingState>('active')

  useEffect(() => {
    // Initial synchronous classification
    const classify = () => {
      if (document.visibilityState === 'hidden' || !isPaneVisible) {
        return 'hidden'
      }

      if (document.hasFocus()) {
        return 'active'
      }

      return 'idle'
    }

    setWindowState(classify())

    // Controller notifies on any visibility/focus/window-state change
    const controller = createRendererLoopPauseController(() => {
      setWindowState(classify())
    })

    return () => controller.dispose()
  }, [isPaneVisible])

  return windowState
}