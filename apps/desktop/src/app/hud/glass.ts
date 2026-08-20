import { type RefObject, useEffect } from 'react'

/** The caret is in the composer — see the `:has()` rules in styles.css. */
const TYPING_SELECTOR = '[data-slot="composer-rich-input"]:focus'

/** An open completion list owns the surface; the band falls back behind it. */
const DRAWER_SELECTOR = '[data-slot="composer-completion-drawer"]'

/**
 * Native frost behind the band.
 *
 * A platform material, not CSS — `backdrop-filter` reaches nothing here,
 * because a transparent window's backdrop root is the document and the desktop
 * was never in it. The material is composited BELOW the web contents (macOS
 * vibrancy via WindowServer, Windows 11 via the DWM backdrop), which is what
 * lets it see the desktop and also what makes it untouchable from the page: no
 * mask, clip or stacking order can shape it.
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
 * Two vetoes sit over that:
 *
 * - `backing` — because the frost is the window and not the sheet, it is only
 *   ever right when the sheet covers the window; short of that the excess is
 *   frost over empty space. Gating the caller's `engaged` alone would not do
 *   it — focus turns the frost on by itself, which is how a brand new thread
 *   still frosted its whole empty window.
 * - An open completion drawer, which drops the band to 25% and blurs it
 *   (see the `composer-completion-drawer` rule in styles.css). Full-strength
 *   frost behind a band that has deliberately stepped back is the same bare
 *   slab in a different disguise. Observed rather than passed in: the drawer
 *   mounts inside the composer subtree from three different call sites, so a
 *   prop would need every one of them to remember.
 *
 * Whether the frost is wanted AT ALL is the user's translucency setting, and
 * that answer lives in main (`hudFrostFor`) next to the state it reads. This
 * hook reports what the band is doing; it does not decide the material.
 */
export function useHudGlass(rootRef: RefObject<HTMLElement | null>, engaged: boolean, backing: boolean): void {
  useEffect(() => {
    const root = rootRef.current
    const setFrost = window.hermesDesktop?.hud?.setFrost

    if (!root || !setFrost) {
      return
    }

    let on: boolean | null = null

    const apply = () => {
      const next =
        backing &&
        root.querySelector(DRAWER_SELECTOR) === null &&
        (engaged || root.querySelector(TYPING_SELECTOR) !== null)

      if (on !== next) {
        on = next
        void setFrost(next)
      }
    }

    // The drawer mounts and unmounts without any focus change, so neither
    // focusin/focusout nor a re-render is guaranteed to follow it. Coalesced
    // to a frame: this observes the whole shell, and a streaming reply mutates
    // the transcript tens of times a second — the drawer's state cannot change
    // more than once per paint, so re-deciding per mutation is pure churn.
    let frame: null | number = null

    const schedule = () => {
      if (frame === null) {
        frame = requestAnimationFrame(() => {
          frame = null
          apply()
        })
      }
    }

    const observer = new MutationObserver(schedule)

    observer.observe(root, { childList: true, subtree: true })

    apply()
    root.addEventListener('focusin', apply)
    root.addEventListener('focusout', apply)

    return () => {
      void setFrost(false)
      observer.disconnect()

      if (frame !== null) {
        cancelAnimationFrame(frame)
      }

      root.removeEventListener('focusin', apply)
      root.removeEventListener('focusout', apply)
    }
  }, [backing, engaged, rootRef])
}
