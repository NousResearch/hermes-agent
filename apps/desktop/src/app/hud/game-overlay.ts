import { useEffect, useState } from 'react'

/**
 * Whether a fullscreen app (a game) is under the HUD.
 *
 * Main owns the answer — it polls the OS window list while the HUD is open
 * (see startHudGameOverlayFeed / hud-game-overlay.ts) and pushes changes here.
 * The renderer cannot know this on its own: which OS window owns the screen is
 * not a fact a page can observe.
 *
 * The shell wears it as `data-hud-game`, and the stylesheet answers with the
 * in-game chat treatment: the idle bar steps back to a glanceable opacity so it
 * reads as part of the game's HUD rather than a desktop window parked over it,
 * and the transcript stays up instead of fading on a timer (see `useHudHeld`) —
 * a chat log you look back at during a lull is useless if it erases itself.
 */
export function useHudGameOverlay(): boolean {
  const [active, setActive] = useState(false)

  useEffect(() => {
    const off = window.hermesDesktop?.hud?.onGameOverlay?.(state => setActive(state.active))

    return () => off?.()
  }, [])

  return active
}
