/**
 * Narrow-viewport edge overlays — the tree's take on the app's hover-reveal
 * collapse. Collapsible panes leave the grid below the sidebar-collapse
 * breakpoint; an edge strip (hover) or PANE_TOGGLE_REVEAL_EVENT (⌘B / ⌘G /
 * titlebar toggles route here on narrow) slides the pane OVER the layout
 * instead of squeezing it. Event reveals pin; hover reveals follow the mouse.
 */

import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useEffect, useMemo, useRef, useState } from 'react'

import { PaneTab, PaneTabLabel, PaneTabStrip } from '@/components/ui/pane-tab'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import type { Contribution } from '@/contrib/types'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { cn } from '@/lib/utils'

import { PANE_TOGGLE_REVEAL_EVENT } from '../..'
import { allPaneIds, findGroupOfPane } from '../model'
import { $hiddenTreePanes, $layoutTree, $narrowViewport } from '../store'

import { paneChrome } from './track-model'

const SWIPE_THRESHOLD_PX = 56
const SWIPE_AXIS_RATIO = 1.25

type SwipeSide = 'left' | 'right'
type SwipeMode = 'close' | 'open'
type SwipeState = { mode: SwipeMode; pointerId: number; side: SwipeSide; startX: number; startY: number }

const isSessionsPane = (pane: Contribution | undefined) =>
  Boolean(pane && (pane.id === 'sessions' || paneChrome(pane).revealAliases?.includes('sessions')))

const completedHorizontalSwipe = (swipe: SwipeState, x: number, y: number): boolean => {
  const dx = x - swipe.startX
  const dy = y - swipe.startY
  const inward = swipe.side === 'left' ? dx : -dx
  const distance = swipe.mode === 'open' ? inward : -inward

  return distance >= SWIPE_THRESHOLD_PX && Math.abs(dx) >= Math.abs(dy) * SWIPE_AXIS_RATIO
}

export function NarrowOverlays() {
  const narrow = useStore($narrowViewport)
  const tree = useStore($layoutTree)
  const panes = useContributions('panes')
  const hiddenPanes = useStore($hiddenTreePanes)
  const [reveal, setReveal] = useState<{ id: string; pinned: boolean } | null>(null)
  const overlayLayerRef = useRef<HTMLDivElement>(null)
  const swipeRef = useRef<SwipeState | null>(null)

  // Own an Escape layer only while something is revealed, so Escape closes the
  // overlay only when it's the top layer (never under a dialog / edit mode).
  const revealActive = reveal !== null
  useEffect(() => (revealActive ? pushEscapeLayer(ESCAPE_PRIORITY.narrowOverlay) : undefined), [revealActive])

  const inTree = useMemo(() => new Set(tree ? allPaneIds(tree) : []), [tree])

  const collapsibles = useMemo(
    () => panes.filter(p => paneChrome(p).collapsible && inTree.has(p.id) && !hiddenPanes.has(p.id)),
    [panes, inTree, hiddenPanes]
  )

  const collapsiblesRef = useRef(collapsibles)
  collapsiblesRef.current = collapsibles

  // Phone navigation owns horizontal swipes while preserving ordinary vertical
  // scroll. Restore the prior page policy exactly when narrow mode unmounts.
  useEffect(() => {
    if (!narrow) {
      return
    }

    const previousTouchAction = globalThis.document.body.style.touchAction
    globalThis.document.body.style.touchAction = 'pan-y'

    return () => {
      globalThis.document.body.style.touchAction = previousTouchAction
    }
  }, [narrow])

  // A revealed drawer is modal to the layout beneath it: pointer input lands
  // on the dismiss surface and keyboard focus stays inside the drawer. Preserve
  // every sibling's prior inert state and the user's prior focus exactly.
  useEffect(() => {
    if (!reveal?.pinned) {
      return
    }

    const layer = overlayLayerRef.current
    const parent = layer?.parentElement

    if (!layer || !parent) {
      return
    }

    const background = [...parent.children].filter(
      (element): element is HTMLElement => element !== layer && element instanceof HTMLElement
    )

    const previousInert = background.map(element => ({ element, inert: element.inert }))

    const previousFocus =
      globalThis.document.activeElement instanceof HTMLElement ? globalThis.document.activeElement : null

    previousInert.forEach(({ element }) => {
      element.inert = true
    })

    layer.querySelector<HTMLElement>('[data-narrow-overlay-pane]')?.focus({ preventScroll: true })

    return () => {
      previousInert.forEach(({ element, inert }) => {
        element.inert = inert
      })

      if (previousFocus?.isConnected) {
        previousFocus.focus({ preventScroll: true })
      }
    }
  }, [reveal])

  // On narrow screens, an inward horizontal gesture may start anywhere in the
  // left half of the app. Listen at window scope so chat content, blank space,
  // and titlebar chrome all share one route without a transparent hit layer
  // stealing taps.
  useEffect(() => {
    if (!narrow) {
      return
    }

    let openingSwipe: SwipeState | null = null

    const clearSwipe = (event: globalThis.PointerEvent) => {
      if (openingSwipe?.pointerId === event.pointerId) {
        openingSwipe = null
      }
    }

    const onPointerDown = (event: globalThis.PointerEvent) => {
      openingSwipe = null

      if (
        reveal ||
        (event.pointerType !== 'touch' && event.pointerType !== 'pen') ||
        event.button !== 0 ||
        event.clientX > globalThis.innerWidth / 2
      ) {
        return
      }

      const sessionsPane = collapsiblesRef.current.find(isSessionsPane)

      if (!sessionsPane) {
        return
      }

      const side: SwipeSide = paneChrome(sessionsPane).placement === 'left' ? 'left' : 'right'
      openingSwipe = {
        mode: 'open',
        pointerId: event.pointerId,
        side,
        startX: event.clientX,
        startY: event.clientY
      }
    }

    const onPointerMove = (event: globalThis.PointerEvent) => {
      const swipe = openingSwipe

      if (
        !swipe ||
        swipe.mode !== 'open' ||
        swipe.pointerId !== event.pointerId ||
        !completedHorizontalSwipe(swipe, event.clientX, event.clientY)
      ) {
        return
      }

      const sessionsPane = collapsiblesRef.current.find(isSessionsPane)
      openingSwipe = null

      if (sessionsPane) {
        event.preventDefault()
        setReveal({ id: sessionsPane.id, pinned: true })
      }
    }

    globalThis.addEventListener('pointerdown', onPointerDown, true)
    globalThis.addEventListener('pointermove', onPointerMove, true)
    globalThis.addEventListener('pointerup', clearSwipe, true)
    globalThis.addEventListener('pointercancel', clearSwipe, true)

    return () => {
      openingSwipe = null
      globalThis.removeEventListener('pointerdown', onPointerDown, true)
      globalThis.removeEventListener('pointermove', onPointerMove, true)
      globalThis.removeEventListener('pointerup', clearSwipe, true)
      globalThis.removeEventListener('pointercancel', clearSwipe, true)
    }
  }, [narrow, reveal])

  // ⌘B / ⌘G's narrow branch dispatches the app's toggle-reveal event with the
  // REAL pane id — accept those via each contribution's revealAliases.
  useEffect(() => {
    if (!narrow) {
      setReveal(null)

      return
    }

    const onToggle = (event: Event) => {
      const detail = (event as CustomEvent<{ id?: string; mode?: 'close' | 'open' | 'toggle' }>).detail
      const id = detail?.id

      if (!id) {
        return
      }

      const match = collapsiblesRef.current.find(p => p.id === id || paneChrome(p).revealAliases?.includes(id))

      if (!match) {
        return
      }

      // `open`/`close` are explicit intents (programmatic reveal, titlebar show);
      // `toggle` (default) is the ⌘B/⌘G flip.
      const mode = detail?.mode ?? 'toggle'
      setReveal(current => {
        if (mode === 'open') {
          return { id: match.id, pinned: true }
        }

        if (mode === 'close') {
          return current?.id === match.id ? null : current
        }

        return current?.id === match.id && current.pinned ? null : { id: match.id, pinned: true }
      })
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented || !isTopEscapeLayer(ESCAPE_PRIORITY.narrowOverlay)) {
        return
      }

      event.preventDefault()
      setReveal(null)
    }

    window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, onToggle)
    window.addEventListener('keydown', onKeyDown)

    return () => {
      window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, onToggle)
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [narrow])

  if (!narrow || collapsibles.length === 0) {
    return null
  }

  const sideOf = (c: Contribution): SwipeSide => (paneChrome(c).placement === 'left' ? 'left' : 'right')
  const revealed = reveal ? collapsibles.find(p => p.id === reveal.id) : undefined
  const sessionsRevealed = isSessionsPane(revealed)
  const sides = [...new Set(collapsibles.map(sideOf))]

  const beginSwipe = (event: ReactPointerEvent<HTMLElement>, mode: SwipeMode, side: SwipeSide) => {
    if ((event.pointerType !== 'touch' && event.pointerType !== 'pen') || event.button !== 0) {
      return
    }

    swipeRef.current = {
      mode,
      pointerId: event.pointerId,
      side,
      startX: event.clientX,
      startY: event.clientY
    }

    if (mode === 'open') {
      event.currentTarget.setPointerCapture?.(event.pointerId)
    }
  }

  const advanceSwipe = (event: ReactPointerEvent<HTMLElement>, sessionsPane?: Contribution) => {
    const swipe = swipeRef.current

    if (!swipe || swipe.pointerId !== event.pointerId || !completedHorizontalSwipe(swipe, event.clientX, event.clientY)) {
      return
    }

    swipeRef.current = null
    event.preventDefault()

    if (swipe.mode === 'open' && sessionsPane) {
      setReveal({ id: sessionsPane.id, pinned: true })
    } else if (swipe.mode === 'close') {
      setReveal(null)
    }
  }

  const endSwipe = (event: ReactPointerEvent<HTMLElement>) => {
    if (swipeRef.current?.pointerId === event.pointerId) {
      swipeRef.current = null
    }
  }

  // The revealed pane's ZONE-mates that also left the grid (the sessions zone
  // stacks SESSIONS | BOTS): the overlay mirrors the zone's tab strip so a
  // pane docked into a collapsed zone stays reachable on narrow viewports —
  // without this, only the zone's first pane ever surfaces again.
  const zonePanes = (() => {
    if (!revealed || !tree) {
      return [revealed].filter((p): p is Contribution => Boolean(p))
    }

    const zone = findGroupOfPane(tree, revealed.id)
    const mates = zone ? zone.panes.map(id => collapsibles.find(p => p.id === id)) : []
    const shown = mates.filter((p): p is Contribution => Boolean(p))

    return shown.length > 0 ? shown : [revealed]
  })()

  return (
    <div className="contents" data-narrow-overlay-layer="" ref={overlayLayerRef}>
      {/* Desktop hover intent remains a narrow edge strip. Phone opening
          gestures are captured across the entire left half above. */}
      {sides.map(side => (
        <div
          className={cn('absolute inset-y-0 z-30 w-1.5', side === 'left' ? 'left-0' : 'right-0')}
          data-narrow-overlay-edge={side}
          key={side}
          onMouseEnter={() => {
            const first = collapsibles.find(p => sideOf(p) === side)

            if (first) {
              setReveal(current => (current?.pinned ? current : { id: first.id, pinned: false }))
            }
          }}
        />
      ))}

      {revealed && (
        <>
          {reveal?.pinned && (
            <div
              aria-hidden="true"
              className="absolute inset-0 z-[var(--z-narrow-overlay-backdrop)] cursor-default bg-transparent"
              data-narrow-overlay-dismiss=""
              onClick={event => {
                event.preventDefault()
                event.stopPropagation()
                setReveal(null)
              }}
              onPointerDown={event => event.stopPropagation()}
            />
          )}
          <div
            aria-label={revealed.title ?? revealed.id}
            aria-modal={reveal?.pinned ? 'true' : undefined}
            className={cn(
              'absolute inset-y-0 z-[var(--z-narrow-overlay)] flex flex-col overflow-hidden bg-(--ui-sidebar-surface-background) shadow-2xl',
              sessionsRevealed && 'touch-pan-y',
              sideOf(revealed) === 'left'
                ? 'left-0 border-r border-(--ui-stroke-secondary)'
                : 'right-0 border-l border-(--ui-stroke-secondary)'
            )}
            // Floats OVER the layout, so under glass its surface must mask the
            // panes beneath it — a see-through overlay reads as text bleeding
            // through text. Contract: `[data-glass-opaque]` in styles.css.
            data-glass-opaque=""
            data-narrow-overlay-pane=""
            data-sessions-swipe-close={sessionsRevealed ? sideOf(revealed) : undefined}
            onMouseLeave={() => setReveal(current => (current?.pinned ? current : null))}
            onPointerCancel={endSwipe}
            onPointerDown={event => {
              if (sessionsRevealed) {
                beginSwipe(event, 'close', sideOf(revealed))
              }
            }}
            onPointerMove={advanceSwipe}
            onPointerUp={endSwipe}
            role={reveal?.pinned ? 'dialog' : undefined}
            // Match the pane's docked width (sessions ~237px, files its rail
            // width) instead of a fat fixed 20rem — capped for tiny screens.
            style={{ width: `min(${(revealed.data as { width?: string } | undefined)?.width ?? '18rem'}, 85vw)` }}
            tabIndex={reveal?.pinned ? -1 : undefined}
          >
            {/* Zone-mates share the overlay through the zone's own tab strip
                (SESSIONS | BOTS) — a lone pane keeps the stripless form. */}
            {zonePanes.length > 1 && (
              <PaneTabStrip>
                {zonePanes.map(pane => (
                  <PaneTab
                    active={pane.id === revealed.id}
                    aria-selected={pane.id === revealed.id}
                    data-narrow-overlay-tab={pane.id}
                    key={pane.id}
                    onPointerDown={event => {
                      if (event.button === 0) {
                        event.preventDefault()
                        setReveal(current => ({ id: pane.id, pinned: current?.pinned ?? false }))
                      }
                    }}
                  >
                    <PaneTabLabel>{pane.title ?? pane.id}</PaneTabLabel>
                  </PaneTab>
                ))}
              </PaneTabStrip>
            )}
            <ContribBoundary id={revealed.id}>
              {revealed.render && <ContribRender render={revealed.render} />}
            </ContribBoundary>
          </div>
        </>
      )}
    </div>
  )
}
