import { type CSSProperties, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { closeHud } from '@/store/hud'
import { $busy, $messages } from '@/store/session'

import { WiredPane } from '../contrib/wiring'
import { titlebarButtonClass } from '../shell/titlebar'

import { useHudClickThrough } from './click-through'
import { useReportHudSession } from './handoff'

/** How long the thread stays visible after the last activity before it starts
 *  fading (WoW chat frame behavior). Focus holds it open past this. */
const HUD_RECENT_HOLD_MS = 6_000

/** Band visibility timings, published to CSS as custom properties so this
 *  module and the stylesheet cannot drift apart. Reveal is quick — it is an
 *  answer to the user; the fade lingers, then goes slowly. */
const HUD_REVEAL_MS = 150
const HUD_FADE_DELAY_MS = 3_000
const HUD_FADE_MS = 1_200

/**
 * True for a hold window after any conversation activity (a message landing,
 * a stream flushing, a turn starting or ending). The CSS uses it — alongside
 * :focus-within — to decide whether the thread is visible; idle HUD mode is
 * just the Spotlight bar.
 *
 * $messages replaces ~30×/s mid-stream, so activity RESTARTS the timer on
 * every flush — the thread stays up while a reply is writing and for the hold
 * window after it finishes, without a per-flush re-render (state only changes
 * on the false↔true edges).
 */
function useRecentActivity(): boolean {
  const [recent, setRecent] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // eslint-disable-next-line no-restricted-syntax -- timer handle, not an atom mirror
  useEffect(() => {
    const bump = () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }

      setRecent(true)
      timerRef.current = setTimeout(() => setRecent(false), HUD_RECENT_HOLD_MS)
    }

    // subscribe() fires immediately, so a HUD opened onto an existing
    // conversation starts with the thread showing, then fades.
    const offMessages = $messages.subscribe(bump)
    const offBusy = $busy.subscribe(busy => busy && bump())

    return () => {
      offMessages()
      offBusy()

      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [])

  return recent
}

/**
 * HUD mode's shell — the chrome-free floating chat.
 *
 * Deliberately almost nothing: it mounts the SAME wired chat surface the
 * workspace pane does, so the composer here IS the app's composer (slash
 * commands, `@` refs, attachments, queue, voice, model pill) and the transcript
 * is the app's transcript, rendered by the app's renderer. Only the frame
 * changes — no titlebar, no statusbar, no pane tree, no sidebars.
 *
 * The shape is macOS Spotlight: at rest, the centered composer bar is the
 * whole interface. The thread renders as bare text above it and is
 * visibility-gated like a game chat frame — shown while a turn is recent or the
 * composer has focus, faded out otherwise (see the `[data-hud-shell]` CSS and
 * `useRecentActivity`).
 */
export function HudShell() {
  const { t } = useI18n()
  const recent = useRecentActivity()

  // Main holds the session id on this window's behalf, so leaving HUD mode can
  // hand the app window back whatever conversation ended up here.
  useReportHudSession()

  // Which screen EDGE the window is parked against. Parked tight to the top,
  // the composer flips to the window's top edge and the thread grows DOWN
  // (data-hud-edge). Computed here from window.screenY — no IPC: the renderer
  // always knows where its window is. Polled because the DOM has no
  // window-move event; 300ms is imperceptible for a layout flip.
  //
  // EDGE-tight, not a midpoint rule: the first cut compared topGap<bottomGap,
  // which flips the layout the moment the window crosses the vertical center
  // of the screen — reported (correctly) as "flips way too early". Now it
  // flips to 'top' only when the window is actually parked against the top
  // (within FLIP_ON px of the usable area, i.e. below the menu bar), and back
  // once it clearly leaves (FLIP_OFF) — the gap between the two is hysteresis
  // so the layout can't flutter while the window is dragged along the line.
  const [edge, setEdge] = useState<'bottom' | 'top'>('bottom')

  useEffect(() => {
    // ZERO tolerance by explicit request: top-mode only when the window is
    // flush against the usable top (gap 0 — macOS won't let it overlap the
    // menu bar, so flush IS availTop). Tiny FLIP_OFF so the 300ms poll can't
    // flutter on sub-pixel jitter while parked.
    const FLIP_ON = 0
    const FLIP_OFF = 4

    const measure = () => {
      // availTop ≈ menu bar / notch inset on macOS; screenY is in full-screen
      // coordinates, so "parked at the top" means screenY ≈ availTop, not 0.
      const availTop = (window.screen as { availTop?: number }).availTop ?? 0
      const topGap = window.screenY - availTop

      setEdge(prev => (topGap <= FLIP_ON ? 'top' : topGap >= FLIP_OFF ? 'bottom' : prev))
    }

    measure()
    const timer = setInterval(measure, 300)
    window.addEventListener('resize', measure)

    return () => {
      clearInterval(timer)
      window.removeEventListener('resize', measure)
    }
  }, [])

  // Whether the thread actually overflows its band. Gates the band's no-drag
  // carve-out (styles.css): a band with nothing to scroll stays part of the
  // window's drag region, so a short conversation never blocks moving the HUD.
  const [scrollable, setScrollable] = useState(false)
  const rootRef = useRef<HTMLDivElement | null>(null)

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

      setScrollable(Boolean(el && el.scrollHeight > el.clientHeight + 4))

      // How tall the band actually needs to be. The transcript is packed to the
      // bottom, so this is the distance from the topmost visible row down to the
      // bar — which is 0 on a fresh session, and the glass then collapses behind
      // the bar instead of painting an empty slab over the whole window.
      //
      // Written straight to the element rather than through state: it changes on
      // every stream flush, and the sheet resizing must not re-render the tree.
      const rows = el?.querySelectorAll<HTMLElement>('[data-slot="aui_thread-content"] > *:not([data-slot])')
      const top = rows?.length ? rows[0].getBoundingClientRect().top : null
      const height = top === null || !el ? 0 : Math.max(0, el.getBoundingClientRect().bottom - top)
      root.style.setProperty('--hud-band-height', `${Math.round(height)}px`)
    }

    // The viewport mounts async (lazy chat surface); poll briefly until it
    // exists, then let the ResizeObserver own it.
    measure()
    const probe = setInterval(measure, 500)

    return () => {
      clearInterval(probe)
      ro.disconnect()
    }
  }, [])

  useHudClickThrough(rootRef)

  // Force the HOST layers transparent. index.html's pre-paint script writes an
  // opaque themed background onto <html> as an INLINE style (the anti-white-
  // flash trick), and an inline style beats any stylesheet rule — so without
  // this the window is a solid slab and every translucent panel below is just
  // glass over a white wall. A style tag with `!important` is what the pet
  // overlay and quick entry already do; they get it at mount because they are
  // bespoke roots, and the HUD needs the same because it is not.
  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = 'html,body,#root{background:transparent !important;}'
    document.head.appendChild(style)

    return () => style.remove()
  }, [])

  return (
    <div
      className="relative flex h-screen w-screen flex-col overflow-hidden"
      data-hud-edge={edge}
      data-hud-recent={recent ? '' : undefined}
      data-hud-scrollable={scrollable ? '' : undefined}
      data-hud-shell
      ref={rootRef}
      style={
        {
          '--hud-fade-delay': `${HUD_FADE_DELAY_MS}ms`,
          '--hud-fade': `${HUD_FADE_MS}ms`,
          '--hud-reveal': `${HUD_REVEAL_MS}ms`
        } as CSSProperties
      }
    >
      {/* The band's sheet, on a layer of its own so it can carry the fade
          without the app's chat surface having to know about it. FIRST child so
          it paints behind the transcript. */}
      <div aria-hidden data-hud-glass />

      <WiredPane part="chatRoutes" />

      {/* The top fade band, as a drag handle. Its text is masked to nothing up
          there, so handing the band's mouse input to the window manager costs
          no readable content — and it gives the HUD a grab area that isn't the
          composer.

          LAST child on purpose. Electron collects draggable regions by walking
          the layout tree in order, uniting `drag` rects and subtracting
          `no-drag` ones, so later elements win. Above `WiredPane` this strip
          was silently subtracted away by the scrollback's full-height `no-drag`
          rect (z-index does not enter into it — the region math is rect-based,
          not paint-order-based). */}
      <div aria-hidden data-hud-drag-strip />

      {/* The way back. HUD mode has no titlebar, so without this the only
          exits are ⌘⇧H and ⌘W — both invisible. Floats over the scrollback
          (which is short and top-fades, so it rarely collides with text) and
          carves itself out of the drag region so the click lands. */}
      <Tip label={t.titlebar.exitHud}>
        <Button
          aria-label={t.titlebar.exitHud}
          className={`${titlebarButtonClass} absolute right-1.5 top-1.5 z-20 bg-transparent [-webkit-app-region:no-drag]`}
          data-hud-exit=""
          onClick={closeHud}
          size="icon-titlebar"
          type="button"
          variant="ghost"
        >
          <Codicon name="screen-normal" />
        </Button>
      </Tip>
    </div>
  )
}
