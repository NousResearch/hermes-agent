import { type RefObject, useEffect } from 'react'

/** The caret is in the composer — see the `:has()` rules in styles.css. */
const TYPING_SELECTOR = '[data-slot="composer-rich-input"]:focus'

/**
 * Native frost behind the band.
 *
 * macOS vibrancy, not CSS — `backdrop-filter` reaches nothing here, because a
 * transparent window's backdrop root is the document and the desktop was never
 * in it. Vibrancy is composited by WindowServer BELOW the web contents, which
 * is what lets it see the desktop and also what makes it untouchable from the
 * page: no mask, clip or stacking order can shape it.
 *
 * That is survivable because the band is a flat panel. It was NOT survivable
 * while the band carried a vertical gradient — the frost stayed a slab under a
 * fading tint and the top went pale exactly where it should have been
 * disappearing, which is what sent the whole thing round in circles. Uniform
 * panel, uniform frost.
 *
 * It still cannot animate, so it is switched while the tint is at full strength
 * and can hide the change: on the moment the band is engaged, off the moment the
 * hold ends and the opacity fade begins. Letting it outlive the fade leaves bare
 * untinted frost on screen — a grey blurred rectangle that pops out at the end
 * instead of a band fading away.
 *
 * Engaged means the caret is in the composer, matching the stylesheet. Merely
 * holding window focus does not count: activating a window restores focus to
 * whatever had it last, so grabbing the bar to drag the HUD would otherwise
 * read as sitting down to use it. Queried live rather than tracked from
 * document.activeElement, which stays put when the window is blurred and would
 * latch the frost on forever once the user had ever typed here.
 *
 * `backing` is the veto over both of those. Because the frost is the window and
 * not the sheet, it is only ever right when the sheet covers the window; short
 * of that the excess is frost over empty space. Gating the caller's `engaged`
 * alone would not do it — focus turns the frost on by itself, which is how a
 * brand new thread still frosted its whole empty window.
 */
export function useHudGlass(rootRef: RefObject<HTMLElement | null>, engaged: boolean, backing: boolean): void {
  useEffect(() => {
    const root = rootRef.current
    const setVibrancy = window.hermesDesktop?.hud?.setVibrancy

    if (!root || !setVibrancy) {
      return
    }

    let on: boolean | null = null

    const apply = () => {
      const next = backing && (engaged || root.querySelector(TYPING_SELECTOR) !== null)

      if (on !== next) {
        on = next
        void setVibrancy(next)
      }
    }

    apply()
    root.addEventListener('focusin', apply)
    root.addEventListener('focusout', apply)

    return () => {
      void setVibrancy(false)
      root.removeEventListener('focusin', apply)
      root.removeEventListener('focusout', apply)
    }
  }, [backing, engaged, rootRef])
}
