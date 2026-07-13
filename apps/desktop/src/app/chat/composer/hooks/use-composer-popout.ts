import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useEffect } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import {
  $composerPopoutPosition,
  $composerPoppedOut,
  readPopoutBounds,
  setComposerPopoutPosition,
  setComposerPoppedOut
} from '@/store/composer-popout'
import { isSecondaryWindow } from '@/store/windows'

import { useComposerScope } from '../scope'

import { useComposerPopoutGestures } from './use-popout-drag'

interface UseComposerPopoutOptions {
  composerRef: RefObject<HTMLFormElement | null>
}

/**
 * Pop-out engine: the docked↔floating state (a shared, persisted atom), the
 * dock/float/toggle actions, the drag gestures, and the on-screen re-clamp.
 * Secondary windows (the tiny Ctrl+Shift+N window, subagent watch windows) can't
 * pop out — a floating composer makes no sense there and would yank the main
 * window's composer out via the shared atom.
 */
export function useComposerPopout({ composerRef }: UseComposerPopoutOptions) {
  // The floating composer is a window-level singleton: only the main scope
  // (not tiles) in a primary window may pop out.
  const scope = useComposerScope()
  const popoutAllowed = !isSecondaryWindow() && scope.popoutAllowed
  const poppedOut = useStore($composerPoppedOut) && popoutAllowed
  const popoutPosition = useStore($composerPopoutPosition)

  const handleComposerPopOut = useCallback(() => {
    triggerHaptic('open')
    setComposerPoppedOut(true)
  }, [])

  const handleComposerDock = useCallback(() => {
    triggerHaptic('success')
    setComposerPoppedOut(false)
  }, [])

  // Double-click the grab area toggles dock/float. Undocking restores the last
  // position (the persisted atom is never cleared on dock).
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
    position: popoutPosition
  })

  // Keep the floating box on-screen: re-clamp (with the real measured size +
  // thread bounds) when it pops out and on every window resize — so a position
  // persisted on a bigger/other monitor, a shrunk window, or now-wider sidebar
  // can never strand it. The rAF pass re-clamps after layout settles (sidebar
  // widths, fonts), so anyone loading in out of bounds is pulled back + saved
  // even if the first measure was premature.
  useEffect(() => {
    if (!poppedOut) {
      return undefined
    }

    const reclamp = (persist: boolean) => {
      const el = composerRef.current
      const size = el ? { height: el.offsetHeight, width: el.offsetWidth } : undefined
      setComposerPopoutPosition($composerPopoutPosition.get(), { area: readPopoutBounds(el), persist, size })
    }

    reclamp(true)
    const raf = requestAnimationFrame(() => reclamp(true))
    const onResize = () => reclamp(false)
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
    }
  }, [composerRef, poppedOut])

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
