import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { usePaneGroup, usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { triggerHaptic } from '@/lib/haptics'
import {
  $composerPopoutGesturesEnabled,
  $composerPopoutZone,
  clampPopoutPosition,
  getComposerPopoutZone,
  popoutBoundsElement,
  type PopoutPosition,
  readPopoutBounds,
  setComposerPoppedOut
} from '@/store/composer-popout'
import { isSecondaryWindow } from '@/store/windows'

import { useComposerPopoutGestures } from './use-popout-drag'

interface UseComposerPopoutOptions {
  composerRef: RefObject<HTMLFormElement | null>
}

/**
 * This surface's on-screen placement, derived from its zone's drag intent.
 *
 * A zone stores one intent for its whole tab stack — drag the box in any tab and
 * it moves in all of them — but each surface owns a different rect, so the
 * intent is clamped per surface. Clamping into the store instead would have
 * every keep-alive-mounted tab overwrite the others with a position bounded by
 * ITS geometry, last writer winning: that's how a drag in one tab used to get
 * lost in another.
 *
 * Re-placing is skipped while this surface drags (the gesture already clamped
 * against this rect) and while it's an inactive tab (still mounted, so a live
 * drag would otherwise force a reflow per background tab per frame).
 */
function usePopoutPlacement(
  composerRef: RefObject<HTMLFormElement | null>,
  groupId: string,
  intent: PopoutPosition,
  dragging: boolean,
  poppedOut: boolean
): PopoutPosition {
  const [placement, setPlacement] = useState(intent)
  const visible = usePaneVisible()
  // Re-place while this surface is the visible tab and isn't itself dragging.
  const live = poppedOut && visible && !dragging

  // Resolved before the shared ResizeObserver below registers (hook order puts
  // this layout effect first), so the observer always has this surface's own
  // bounds element rather than a document-wide first match.
  const boundsRef = useRef<Element | null>(null)

  useLayoutEffect(() => {
    boundsRef.current = popoutBoundsElement(composerRef.current)
  })

  const reclamp = useCallback(() => {
    const el = composerRef.current

    if (!el) {
      return
    }

    const size = { height: el.offsetHeight, width: el.offsetWidth }
    const next = clampPopoutPosition(getComposerPopoutZone(groupId).position, size, readPopoutBounds(el))

    // Bail on an unchanged placement: a sash drag resizes the surface every
    // frame, and a fresh object each time re-renders the whole composer.
    setPlacement(prev => (prev.bottom === next.bottom && prev.right === next.right ? prev : next))
  }, [composerRef, groupId])

  // The surface resizing (sash drag, sidebar open, tab split) re-places the box
  // against its new rect; the composer resizing (a growing draft) re-places it
  // against its new height.
  useResizeObserver(
    useCallback(() => {
      if (live) {
        reclamp()
      }
    }, [live, reclamp]),
    composerRef,
    boundsRef
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

/**
 * Pop-out engine: the docked↔floating state, the dock/float/toggle actions, the
 * drag gestures, and this surface's placement.
 *
 * State is scoped to the surface's layout ZONE (its tab stack): tabs in the same
 * zone share one float, so switching tabs keeps the box exactly where you put
 * it, while a split zone beside them keeps its own — popping out on the left
 * doesn't fling a composer out of the right.
 *
 * Secondary windows (the tiny Ctrl+Shift+N window, subagent watch windows) stay
 * docked: a floating composer makes no sense in a scratch window.
 */
export function useComposerPopout({ composerRef }: UseComposerPopoutOptions) {
  const groupId = usePaneGroup()
  const gesturesEnabled = useStore($composerPopoutGesturesEnabled)
  const popoutAllowed = gesturesEnabled && !isSecondaryWindow()
  const zone = useStore(useMemo(() => $composerPopoutZone(groupId), [groupId]))
  const poppedOut = zone.poppedOut && popoutAllowed

  const handleComposerPopOut = useCallback(() => {
    triggerHaptic('open')
    setComposerPoppedOut(groupId, true)
  }, [groupId])

  const handleComposerDock = useCallback(() => {
    triggerHaptic('success')
    setComposerPoppedOut(groupId, false)
  }, [groupId])

  // Double-click the grab area toggles dock/float. Undocking restores the last
  // position (a zone's stored position is never cleared on dock).
  const handleComposerToggle = useCallback(() => {
    poppedOut ? handleComposerDock() : handleComposerPopOut()
  }, [handleComposerDock, handleComposerPopOut, poppedOut])

  const {
    dockProximity,
    dragging,
    onPointerDown: onComposerGesturePointerDown
  } = useComposerPopoutGestures({
    composerRef,
    groupId,
    onDock: handleComposerDock,
    onPopOut: handleComposerPopOut,
    poppedOut,
    position: zone.position
  })

  const popoutPosition = usePopoutPlacement(composerRef, groupId, zone.position, dragging, poppedOut)

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
