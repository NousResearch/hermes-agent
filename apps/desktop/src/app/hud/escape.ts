import { type RefObject, useEffect } from 'react'

import { closeHud } from '@/store/hud'

/** What an Escape press inside the HUD window should do. */
export type HudEscapeAction = 'close' | 'ignore'

/**
 * Decide whether Escape dismisses the HUD.
 *
 * Escape is the gesture people reach for to put a floating thing away, and
 * the HUD had no answer to it — dismissal was ⌘W or the composer's exit
 * button. But Escape also belongs to whatever is stacked ON TOP of the shell:
 *
 * - Focus BESIDE the shell is a portalled dialog, popover, menu or the model
 *   picker (same test `hudIgnoresMouse` uses). That surface owns the press —
 *   Escape closes IT, not the HUD under it — so the HUD steps back.
 * - A press some handler already consumed (`defaultPrevented`) stays consumed;
 *   the composer's own editor may claim Escape to clear an autocomplete.
 *
 * Everything else — focus inside the shell, or nowhere — closes the HUD.
 */
export function hudEscapeAction(root: Element, active: Element | null, defaultPrevented: boolean): HudEscapeAction {
  if (defaultPrevented) {
    return 'ignore'
  }

  const overlayFocused = active !== null && !root.contains(active) && !active.contains(root)

  return overlayFocused ? 'ignore' : 'close'
}

/**
 * Escape dismisses the HUD.
 *
 * Listens on the window in the capture phase so a press reaches the decision
 * before any bubbling handler, but defers to anything that has already claimed
 * it (see `hudEscapeAction`). Closing goes through the store so main restores
 * the app window exactly as it does for ⌘W and the exit button.
 */
export function useHudEscape(rootRef: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      const root = rootRef.current

      if (!root) {
        return
      }

      if (hudEscapeAction(root, document.activeElement, event.defaultPrevented) !== 'close') {
        return
      }

      event.preventDefault()
      closeHud()
    }

    // Bubble phase, not capture: a portalled overlay's own Escape handler runs
    // first and can preventDefault, which `hudEscapeAction` then honours.
    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [rootRef])
}
