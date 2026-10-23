import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Presentation-only element fullscreen for the Screen pane's viewport. Never owns
 * the stream, viewer identity or control lease — it only moves where the existing
 * noVNC surface is painted and hands keyboard ownership to the local Escape/F6
 * handler while fullscreen.
 */
export function useScreenFullscreen() {
  const [viewport, setViewport] = useState<HTMLDivElement | null>(null)
  const [active, setActive] = useState(false)
  const [failed, setFailed] = useState(false)
  const returnFocus = useRef<HTMLElement | null>(null)
  const controls = useRef<HTMLDivElement | null>(null)
  const trigger = useRef<HTMLButtonElement | null>(null)
  // Previous committed `active`, used by the focus-restore effect for transition detection.
  const activeRef = useRef(false)
  // The two halves (keydown + keyup) of a local key already reserved this stroke,
  // so the keyup after a mid-stroke exit never reaches the remote app either.
  const localKeys = useRef(new Set<string>())

  const changed = useCallback(() => {
    const owns = viewport !== null && document.fullscreenElement === viewport
    setActive(owns)
  }, [viewport])

  useEffect(() => {
    if (!viewport) {
      return
    }

    changed()
    document.addEventListener('fullscreenchange', changed)

    return () => {
      document.removeEventListener('fullscreenchange', changed)

      if (document.fullscreenElement === viewport) {
        void document.exitFullscreen().catch(() => {})
      }
    }
  }, [changed, viewport])

  const exit = useCallback(async () => {
    if (viewport && document.fullscreenElement === viewport) {
      try {
        await document.exitFullscreen()
      } catch {
        setFailed(true)
      }
    }
  }, [viewport])

  // Restore focus after React has swapped the toolbar between the chrome row and
  // the fullscreen overlay: at event time the trigger ref still points at the
  // overlay button that is about to be detached, so focusing there loses the
  // caret to <body> once React remounts. This effect runs after that commit.
  // eslint-disable-next-line no-restricted-syntax -- local-state transition flag and captured focus, never read by stale callbacks, so the mirror hazard this rule guards does not apply
  useEffect(() => {
    activeRef.current = active

    if (active) {
      return
    }

    const target = returnFocus.current?.isConnected ? returnFocus.current : trigger.current
    target?.focus()
    returnFocus.current = null
  }, [active])

  useEffect(() => {
    // Capture before noVNC's canvas listeners: Escape and F6 are the pane's own
    // controls, not keystrokes meant for the remote desktop.
    const key = (event: KeyboardEvent) => {
      const ownsFullscreen = viewport !== null && document.fullscreenElement === viewport
      const local = ownsFullscreen && (event.key === 'Escape' || event.key === 'F6')

      if (!local && !localKeys.current.has(event.code)) {
        return
      }

      event.preventDefault()
      event.stopImmediatePropagation()

      if (event.type === 'keyup') {
        localKeys.current.delete(event.code)
      } else {
        localKeys.current.add(event.code)

        if (!event.repeat && ownsFullscreen) {
          if (event.key === 'Escape') {
            void exit()
          } else {
            controls.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus()
          }
        }
      }
    }

    document.addEventListener('keydown', key, true)
    document.addEventListener('keyup', key, true)

    return () => {
      document.removeEventListener('keydown', key, true)
      document.removeEventListener('keyup', key, true)
    }
  }, [exit, viewport])

  const enter = useCallback(async () => {
    setFailed(false)

    if (!viewport?.requestFullscreen || document.fullscreenElement) {
      setFailed(true)

      return
    }

    returnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null

    try {
      // Must stay on the user-gesture stack, before any await.
      await viewport.requestFullscreen()

      // A grant that lands after this viewport went away (pane unmounted, bot tile
      // closed) must not strand the window in fullscreen with no exit listener.
      if (!viewport.isConnected && document.fullscreenElement === viewport) {
        await document.exitFullscreen()
      }
    } catch {
      returnFocus.current = null
      setFailed(true)
    }
  }, [viewport])

  return { viewportRef: setViewport, controls, trigger, active, failed, enter, exit }
}
