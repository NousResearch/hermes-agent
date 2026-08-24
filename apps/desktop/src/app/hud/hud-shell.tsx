import { useStore } from '@nanostores/react'
import { type CSSProperties, useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { chatMessageText } from '@/lib/chat-messages'
import { $activeSessionAwaitingInput } from '@/store/prompts'
import { $busy, $messages } from '@/store/session'

import { RICH_INPUT_SLOT } from '../chat/composer/rich-editor'
import { WiredPane } from '../contrib/wiring'

import { useHudClickThrough } from './click-through'
import { useHudGameOverlay } from './game-overlay'
import { useHudGlass } from './glass'
import { useHudGoto, useReportHudSession } from './handoff'
import { hudTranscriptHeight } from './layout'
import { hudResizeDirections, useHudResizeHandle } from './resize-handle'
import { useHudThreadFocus } from './thread-focus'

/** How long the transcript lingers at its glanceable opacity — after a turn
 *  lands, or after you let go of the composer — before it goes. This is the ONLY
 *  hold: the CSS carries no transition-delay, because two stacked holds read as
 *  a third fade state that nobody asked for. Focus keeps it open past this. */
const HUD_RECENT_HOLD_MS = 1100

/** Band visibility timings, published to CSS as custom properties so this
 *  module and the stylesheet cannot drift apart. Reveal is quick — it is an
 *  answer to the user; the fade lingers, then goes slowly. */
const HUD_REVEAL_MS = 110
const HUD_FADE_MS = 180

/** The step DOWN to the glanceable opacity when you let go. Deliberately slower
 *  than the fade that follows it — easing off is a softer gesture than leaving,
 *  and matching them made the two read as one long dissolve. */
const HUD_DIM_MS = Math.round(HUD_FADE_MS * 1.5)

/** The sheet rolling shut. Shorter than the fade so the panel is already gone
 *  while the last of the text is still going — it reads as the transcript being
 *  drawn down into the bar rather than the two dissolving in lockstep. */
const HUD_COLLAPSE_MS = Math.round(HUD_FADE_MS * 0.66)

/** Breathing room the sheet keeps above the first row, so the fade has
 *  somewhere to land. Folded into the measured height rather than added in CSS,
 *  so an empty transcript measures a true zero instead of a 12px strip. */
const HUD_SHEET_OVERHANG_PX = 12

/** Composer on top, transcript always hanging below it — Spotlight's shape,
 *  rather than flipping to follow the screen edge the HUD is parked against. */
const HUD_THREAD_ALWAYS_BELOW = true

const composerHasFocus = () => document.activeElement?.closest(`[data-slot="${RICH_INPUT_SLOT}"]`) != null

/**
 * True for a hold window after any conversation activity (a message landing,
 * a stream flushing, a turn starting or ending). The CSS uses it — alongside
 * :focus-within — to decide whether the thread is visible; idle HUD mode is
 * just the Spotlight bar.
 *
 * $messages replaces ~30×/s mid-stream, so a FOCUSED composer restarts the
 * timer on every flush and the thread stays up for the whole reply, without a
 * per-flush re-render (state only changes on the false↔true edges).
 *
 * Unfocused, activity buys one hold window and no more. Re-arming there kept
 * the band up for the entire length of a reply on a HUD nobody was looking at
 * — the thing it exists to avoid. A turn landing still flashes it; watching
 * one write is what focus is for.
 */
function useRecentActivity(): [boolean, () => void] {
  const [recent, setRecent] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const bumpRef = useRef(() => {})

  // eslint-disable-next-line no-restricted-syntax -- timer handle, not an atom mirror
  useEffect(() => {
    let signature = ''

    // `letGo` is the deliberate gesture — clicking away from the composer —
    // and always buys the full window. Ambient activity does not, unless the
    // composer has focus: whatever is left of the current hold is what a HUD
    // nobody is looking at gets.
    const bump = (letGo = false) => {
      if (timerRef.current) {
        if (!letGo && !composerHasFocus()) {
          return
        }

        clearTimeout(timerRef.current)
      }

      setRecent(true)
      timerRef.current = setTimeout(() => setRecent(false), HUD_RECENT_HOLD_MS)
    }

    // Gated on the transcript actually CHANGING, not on the atom being written.
    // $messages is republished for plenty of reasons that aren't new content
    // (session sync, re-renders, relative timestamps), and re-arming the hold on
    // every one of those latched the band open permanently — the fade simply
    // never got to start.
    const onMessages = () => {
      const messages = $messages.get()
      const last = messages[messages.length - 1]
      const next = `${messages.length}:${last?.id ?? ''}:${last ? chatMessageText(last).length : 0}`

      if (next === signature) {
        return
      }

      signature = next
      bump()
    }

    bumpRef.current = () => bump(true)

    // subscribe() fires immediately, so a HUD opened onto an existing
    // conversation starts with the thread showing, then fades.
    const offMessages = $messages.subscribe(onMessages)
    const offBusy = $busy.subscribe(busy => busy && bump())

    return () => {
      offMessages()
      offBusy()

      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [])

  // Stable, so callers can hang listeners off it.
  const holdBand = useCallback(() => bumpRef.current(), [])

  return [recent, holdBand]
}

/**
 * True while the HUD must stay up regardless of the hold timer.
 *
 * Three things qualify:
 *
 * - A clarify/approval/sudo/secret prompt: something the agent cannot continue
 *   without, and letting it fade hands you a surface you cannot use — zero
 *   opacity, and the window goes mouse-transparent under it, so the prompt is
 *   neither readable nor clickable.
 * - The session being BUSY at all — thinking, calling a tool, streaming — plus
 *   a grace window after the turn ends, so the answer you were waiting for is
 *   still there when it arrives.
 * - GAME OVERLAY MODE, unconditionally. Over a fullscreen game the HUD is the
 *   chat frame, and a chat frame that erases itself a second after each line is
 *   useless: you look back at it when there is a lull in the game, not when the
 *   text happens to be fresh. This is what every in-game chat log does, and it
 *   costs nothing here because the band over a game is bare text with no panel.
 *
 * Pinning on busy was deliberately avoided while the band brought its tinted
 * sheet up with it — that put an opaque panel across the screen for as long as
 * the agent worked. The sheet is a translucent scrim now, so a held band is
 * legible text rather than a slab.
 */
function useHudHeld(gameUnder: boolean): boolean {
  const awaiting = useStore($activeSessionAwaitingInput)
  const busy = useStore($busy)

  // The reply landing must stay readable. Held for the whole turn AND for a
  // grace window after it ends, tracked here rather than by re-arming
  // `useRecentActivity`'s timer: that timer deliberately refuses to re-arm for
  // ambient activity on an unfocused HUD, so the turn's own end could not buy a
  // hold through it and the answer blinked out the instant it finished.
  const [grace, setGrace] = useState(false)

  useEffect(() => {
    if (busy) {
      setGrace(true)

      return
    }

    // Falling edge: keep it up a moment longer, then let the fade have it.
    const timer = setTimeout(() => setGrace(false), HUD_RECENT_HOLD_MS)

    return () => clearTimeout(timer)
  }, [busy])

  return gameUnder || awaiting || busy || grace
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
  const [recent, holdBand] = useRecentActivity()
  // A fullscreen app (a game) is under the HUD: wear `data-hud-game` so the
  // idle bar steps back to overlay opacity (see styles.css). Detection is
  // main's — the page cannot see other apps' windows.
  const gameUnder = useHudGameOverlay()
  const held = useHudHeld(gameUnder)

  // Clicking away to another APP is the most common way the HUD is let go of,
  // and it fires no focusout: the composer stays document.activeElement while
  // the window is inactive. Chrome stops matching `:focus` on an unfocused
  // window all the same, so the band lost its focus state with no hold running
  // and snapped shut instead of stepping down to the glanceable stage.
  useEffect(() => {
    window.addEventListener('blur', holdBand)

    return () => window.removeEventListener('blur', holdBand)
  }, [holdBand])

  // Main holds the session id on this window's behalf, so leaving HUD mode can
  // hand the app window back whatever conversation ended up here.
  useReportHudSession()
  useHudGoto(useNavigate())

  // Which screen EDGE the window is parked against. Parked tight to the top,
  // the composer flips to the window's top edge and the thread grows DOWN
  // (data-hud-edge). Computed here from window.screenY — no IPC: the renderer
  // always knows where its window is. Polled because the DOM has no
  // window-move event; 300ms is imperceptible for a layout flip.
  //
  // EDGE-tight, not a midpoint rule: the first cut compared topGap<bottomGap,
  // which flips the layout the moment the window crosses the vertical center
  // of the screen — reported (correctly) as "flips way too early". Now it
  // flips to 'top' only when the HUD is actually parked against the top, and
  // back once it clearly leaves — the gap between the two thresholds is
  // hysteresis so the layout can't flutter while it's dragged along the line.
  const [edge, setEdge] = useState<'bottom' | 'top'>('top')

  useEffect(() => {
    // Measured on the WINDOW, and flush-only. Flipping is what lets the bar
    // reach the top of the screen at all: the window's top edge can sit against
    // the menu bar, and the flip moves the composer to that edge. Keying it off
    // the visible panel instead meant it could never fire — the panel hugs the
    // bar at the bottom of the window — and took the top of the screen away
    // with it. FLIP_OFF is just enough hysteresis that the 300ms poll can't
    // flutter on sub-pixel jitter while parked.
    const FLIP_ON = 0
    const FLIP_OFF = 4

    const measure = () => {
      // TRYING IT: the bar stays on top and the transcript always hangs below,
      // wherever the HUD is parked. Flip the constant to re-enable the
      // edge-aware layout (the CSS for both orientations is still here).
      if (HUD_THREAD_ALWAYS_BELOW) {
        setEdge('top')

        return
      }

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

  // Whether bar + band actually cover the window. Gates the frost, which is
  // native vibrancy and therefore the WINDOW's content view — it fills the whole
  // rectangle and nothing in the page can clip it to the sheet. Whenever the
  // sheet is shorter than the window, the difference is frost over empty space:
  // a grey slab hanging under the bar with nothing in it. Now that the band is
  // capped it almost never covers the window, so this is almost always false —
  // which is correct, and asking anything looser paints the slab back.
  const [filled, setFilled] = useState(false)
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
        viewportHeight: window.innerHeight
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

    // The viewport mounts async (lazy chat surface); poll briefly until it
    // exists, then let the ResizeObserver own it. Window resize is separate:
    // the transcript's rows may not change size, but the available scrollback
    // must, so observing the rows alone cannot update the band.
    measure()
    const probe = setInterval(measure, 500)
    window.addEventListener('resize', measure)

    return () => {
      clearInterval(probe)
      window.removeEventListener('resize', measure)
      ro.disconnect()
    }
  }, [])

  useHudGlass(rootRef, recent || held, filled)
  useHudClickThrough(rootRef)
  useHudThreadFocus(rootRef)

  // Edge/corner resize frame. The window is created non-resizable so dragging can
  // never be misread as a resize gesture (the Windows transparent-frameless
  // growth bug); the handle is the one sanctioned way to change size, driving
  // the same flip-resizable-for-the-call pattern the pet overlay uses.
  const { resizing: hudResizing, onPointerDown: onHudResizePointerDown } = useHudResizeHandle()
  const resizeDirections = hudResizeDirections(window.hermesDesktop?.hud?.nativeDrag === true)

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
      data-hud-game={gameUnder ? '' : undefined}
      data-hud-recent={recent || held ? '' : undefined}
      data-hud-shell
      // Letting go of the composer re-arms the hold, so the transcript steps
      // down to its glanceable opacity and lingers there instead of jumping
      // straight from full to gone.
      onBlur={holdBand}
      ref={rootRef}
      style={
        {
          '--hud-fade': `${HUD_FADE_MS}ms`,
          '--hud-collapse': `${HUD_COLLAPSE_MS}ms`,
          '--hud-dim': `${HUD_DIM_MS}ms`,
          '--hud-reveal': `${HUD_REVEAL_MS}ms`
        } as CSSProperties
      }
    >
      {/* The band's sheet, on a layer of its own so it can carry the fade
          without the app's chat surface having to know about it. FIRST child so
          it paints behind the transcript. */}
      <div aria-hidden data-hud-glass />

      <WiredPane part="chatRoutes" />

      {/* CanvasTTY-style resize frame. Windows/macOS/X11 get every edge and
          corner; native Wayland gets the right/bottom handles it can honour
          without forbidden global positioning. The handles are invisible
          chrome with native cursors. `data-hud-grabbing` keeps click-through
          from handing the pointer away while the window changes under it. */}
      {resizeDirections.map(direction => (
        <div
          aria-hidden
          data-hud-grabbing={hudResizing ? '' : undefined}
          data-hud-resize={direction}
          key={direction}
          onPointerDown={event => onHudResizePointerDown(event, direction)}
        />
      ))}
    </div>
  )
}
