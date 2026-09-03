import { type RefObject, useEffect, useState } from 'react'

import { hudTranscriptHeight } from './layout'

/** Breathing room the sheet keeps above the first row, so the fade has
 *  somewhere to land. Folded into the measured height rather than added in CSS,
 *  so an empty transcript measures a true zero instead of a 12px strip. */
const HUD_SHEET_OVERHANG_PX = 12

/**
 * Measures the HUD's transcript band and publishes it as `--hud-band-height` /
 * `--hud-bar-height` on the root, returning whether the band + bar fill the
 * window (which gates the frost — see `useHudGlass`).
 *
 * The viewport mounts async (lazy chat surface); poll briefly until it exists,
 * then let the ResizeObserver own it. Window resize is separate: the
 * transcript's rows may not change size, but the available scrollback must, so
 * observing the rows alone cannot update the band.
 */
/** `topInset` is read on every measure: room reserved above the bar (the pixel
 *  pets' headroom) that the transcript band cannot use. */
export function useHudTranscriptBand(rootRef: RefObject<HTMLDivElement | null>, topInset: () => number = () => 0): boolean {
  const [filled, setFilled] = useState(false)

  useEffect(() => {
    const root = rootRef.current

    if (!root) {
      return
    }

    let viewport: HTMLElement | null = null
    const ro = new ResizeObserver(() => measure())

    const measure = () => {
      const el = viewport ?? root.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')

      if (el !== viewport) {
        viewport = el

        if (el) {
          ro.observe(el)

          if (el.firstElementChild) {
            ro.observe(el.firstElementChild)
          }
        }
      }

      // How tall the band actually needs to be — the tight bbox of the message
      // rows only. Measuring to the viewport edge counted the full-window scroll
      // container (min-height: 100%) as transcript and painted a empty slab almost
      // the size of the HUD.
      const rows = el?.querySelectorAll<HTMLElement>('[data-slot="aui_thread-content"] > *:not([data-slot])')

      // Zero-height rows are not a transcript. A fresh thread still renders
      // scaffolding inside the content box (clearance, empty state), so
      // counting rows alone paid the overhang for nothing and left a sliver of
      // sheet hanging under the bar with no text in it.
      const text = !rows?.length
        ? 0
        : Math.max(0, rows[rows.length - 1].getBoundingClientRect().bottom - rows[0].getBoundingClientRect().top)

      const contentSpan = text < 1 ? 0 : text + HUD_SHEET_OVERHANG_PX

      // Once the HUD has a transcript, a resize must buy readable scrollback.
      // The old glance-band ceiling froze this at 152px and turned every extra
      // pixel of native window height into empty transparent chrome.
      const visible = hudTranscriptHeight({
        barHeight: root.querySelector<HTMLElement>('[data-slot="composer-dock"]')?.getBoundingClientRect().height ?? 0,
        contentHeight: contentSpan,
        viewportHeight: window.innerHeight - topInset()
      })

      root.style.setProperty('--hud-band-height', `${visible}px`)

      // …and the bar's real height, which is what the thread has to clear.
      // --composer-measured-height would be the obvious source, but it is a
      // surface var that never lands here, so the clearance silently fell back
      // to the root estimate and reserved ~20px more than the bar occupies —
      // a visible hole under the last message.
      const bar = root.querySelector<HTMLElement>('[data-slot="composer-dock"]')
      const barHeight = bar?.getBoundingClientRect().height ?? 0

      if (bar) {
        ro.observe(bar)
        root.style.setProperty('--hud-bar-height', `${Math.round(barHeight)}px`)
      }

      setFilled(barHeight + visible >= window.innerHeight - 1)
    }

    measure()

    // Once the viewport has mounted, the ResizeObserver above owns every
    // future measurement — a probe that never stops re-runs this on every
    // tick forever, which is exactly the sustained idle CPU / re-render loop
    // the HUD must not have.
    const probe = window.setInterval(() => {
      if (viewport) {
        window.clearInterval(probe)

        return
      }

      measure()
    }, 500)

    window.addEventListener('resize', measure)

    return () => {
      window.clearInterval(probe)
      window.removeEventListener('resize', measure)
      ro.disconnect()
    }
  }, [rootRef])

  return filled
}
