import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useLayoutEffect, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'
import { triggerHaptic } from '@/lib/haptics'
import {
  $composerPopout,
  $composerPopoutGesturesEnabled,
  clampPopoutPosition,
  type PopoutPosition,
  setComposerPoppedOut
} from '@/store/composer-popout'
import { isSecondaryWindow } from '@/store/windows'

import { claimFloatingComposer } from '../floating-target'
import { requestComposerFocus } from '../focus'
import { useComposerSurfaceId } from '../scope'
import { useComposerVisible } from '../visibility'

import { useComposerPopoutGestures } from './use-popout-drag'

interface UseComposerPopoutOptions {
  composerRef: RefObject<HTMLFormElement | null>
}

/** Only the visible recipient measures its box. Placement is viewport-wide;
 * hidden tabs must not overwrite the shared drag intent. */
function usePopoutPlacement(
  composerRef: RefObject<HTMLFormElement | null>,
  intent: PopoutPosition,
  dragging: boolean,
  poppedOut: boolean
): PopoutPosition {
  const [placement, setPlacement] = useState(intent)
  const visible = useComposerVisible()
  // Re-place while this surface is the visible tab and isn't itself dragging.
  const live = poppedOut && visible && !dragging

  const reclamp = useCallback(() => {
    const el = composerRef.current

    if (!el) {
      return
    }

    const size = { height: el.offsetHeight, width: el.offsetWidth }
    const next = clampPopoutPosition($composerPopout.get().position, size)

    // Preserve identity when a resize leaves the placement unchanged.
    setPlacement(prev => (prev.bottom === next.bottom && prev.right === next.right ? prev : next))
  }, [composerRef])

  // A growing draft must stay within the viewport too.
  useResizeObserver(
    useCallback(() => {
      if (live) {
        reclamp()
      }
    }, [live, reclamp]),
    composerRef
  )

  // useLayoutEffect, not useEffect: a tab revealed after the box was dragged in
  // another one must not paint a frame at its stale placement before catching
  // up. Runs before paint, and no-ops for hidden tabs (`live`).
  useLayoutEffect(() => {
    if (!live) {
      return undefined
    }

    reclamp()
    // A second pass after layout settles (sidebar widths, fonts): anyone
    // restored out of bounds is pulled back even if the first measure was
    // premature.
    const raf = requestAnimationFrame(reclamp)
    window.addEventListener('resize', reclamp)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', reclamp)
    }
  }, [intent, live, reclamp])

  return dragging ? intent : placement
}

/** Window-wide dock/float gestures. Secondary scratch/watch windows stay docked. */
export function useComposerPopout({ composerRef }: UseComposerPopoutOptions) {
  const surfaceId = useComposerSurfaceId()
  const gesturesEnabled = useStore($composerPopoutGesturesEnabled)
  const popoutAllowed = gesturesEnabled && !isSecondaryWindow()
  const state = useStore($composerPopout)
  const poppedOut = state.poppedOut && popoutAllowed

  const handleComposerPopOut = useCallback(() => {
    triggerHaptic('open')

    if (surfaceId) {
      claimFloatingComposer(surfaceId)
    }

    setComposerPoppedOut(true)
    requestComposerFocus()
  }, [surfaceId])

  const handleComposerDock = useCallback(() => {
    triggerHaptic('success')
    setComposerPoppedOut(false)
  }, [])

  // Double-click the grab area toggles dock/float. Undocking restores the last
  // position (docking never clears the shared placement).
  const handleComposerToggle = useCallback(() => {
    poppedOut ? handleComposerDock() : handleComposerPopOut()
  }, [handleComposerDock, handleComposerPopOut, poppedOut])

  const {
    dockProximity,
    dragging,
    onPointerDown: onComposerGesturePointerDown
  } = useComposerPopoutGestures({
    composerRef,
    onDock: handleComposerDock,
    onPopOut: handleComposerPopOut,
    poppedOut,
    position: state.position
  })

  const popoutPosition = usePopoutPlacement(composerRef, state.position, dragging, poppedOut)

  return {
    dockProximity,
    dragging,
    handleComposerToggle,
    onComposerGesturePointerDown,
    popoutAllowed,
    popoutPosition,
    poppedOut
  }
}
