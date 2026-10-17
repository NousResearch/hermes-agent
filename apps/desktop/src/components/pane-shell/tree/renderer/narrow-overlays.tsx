/**
 * Narrow-viewport edge overlays — the tree's take on the app's hover-reveal
 * collapse. Collapsible panes leave the grid below the sidebar-collapse
 * breakpoint; an edge strip (hover) or PANE_TOGGLE_REVEAL_EVENT (⌘B / ⌘G /
 * titlebar toggles route here on narrow) slides the pane OVER the layout
 * instead of squeezing it. Event reveals pin; hover reveals follow the mouse.
 */

import { useStore } from '@nanostores/react'
import { type MouseEventHandler, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PaneTab, PaneTabLabel, PaneTabStrip } from '@/components/ui/pane-tab'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import type { Contribution } from '@/contrib/types'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { cn } from '@/lib/utils'
import { $chatOnboardingSolo } from '@/store/onboarding-intro'
import { $paneStates } from '@/store/panes'

import { PANE_TOGGLE_REVEAL_EVENT } from '../..'
import { useWindowControlsOverlap } from '../../geometry'
import { NO_PANE_GROUP } from '../../pane-visibility'
import { allPaneIds, findGroupOfPane, type LayoutNode } from '../model'
import { $hiddenTreePanes, $layoutTree, $narrowViewport } from '../store'

import { KeepAlivePaneSlot, useStablePaneHosts } from './keep-alive-panes'
import { fixedTrackSize, paneChrome, type TrackContext } from './track-model'

/** The width a revealed narrow overlay sizes itself to: the SAME resolution
 *  the pane's zone uses while docked — declared max() refined by the live
 *  widthOverride of the zone's shown panes (fixedTrackSize) — so the overlay
 *  and the docked zone can never disagree. Falls back to the pane's declared
 *  `data.width` (then 18rem) when no zone claims the pane. Pure, so the
 *  regression test asserts the resolution itself (jsdom's CSSOM drops the
 *  `min()` wrapper from style.width, hiding the rendered result). */
export function narrowOverlayWidth(ctx: TrackContext, tree: LayoutNode | null, revealed: Contribution): string {
  if (!tree) {
    return paneChrome(revealed).width ?? '18rem'
  }

  const zone = findGroupOfPane(tree, revealed.id)
  // A minimized zone's docked track is the 28px rail (MINIMIZED_TRACK) — but
  // the overlay is a REVEAL, not the rail: sizing it from the minimized track
  // rendered an unreadable sliver. Resolve the zone as if open; the reveal is
  // exactly the gesture that restores its real width.
  const track = zone ? fixedTrackSize({ ...zone, minimized: false }, 'row', ctx) : null

  return track ?? paneChrome(revealed).width ?? '18rem'
}

export function NarrowOverlays() {
  const narrow = useStore($narrowViewport)
  const solo = useStore($chatOnboardingSolo)
  const tree = useStore($layoutTree)
  const panes = useContributions('panes')
  const paneStates = useStore($paneStates)
  const stableHosts = useStablePaneHosts()
  const hiddenPanes = useStore($hiddenTreePanes)
  const [reveal, setReveal] = useState<{ id: string; pinned: boolean } | null>(null)
  const revealRef = useRef(reveal)
  revealRef.current = reveal

  // Hover hysteresis. The overlay opens OVER the 6px intent strip, so every
  // dismissal re-exposes the strip under the cursor and the next mouse move
  // re-fires mouseenter — the sidebar popped back exactly where the user was
  // trying to work ("overstays its welcome"). After a dismissal the edge
  // stays closed until the pointer LEAVES the edge zone, which re-arms it.
  const hoverArmed = useRef(true)

  // The revealed overlay spans the full viewport height (inset-y-0 below), so
  // its tab strip starts at the top edge — under the native window controls
  // (macOS traffic lights) when the sidebar is on the left. Reserve their rect
  // the same way a docked zone does (TreeGroup's wcOverlap -> paddingTop plus
  // an absolute drag-region spacer so the band stays a window-drag target).
  const overlayRef = useRef<HTMLDivElement>(null)
  const wcOverlap = useWindowControlsOverlap(overlayRef, reveal !== null)

  const onMouseLeave = useCallback<MouseEventHandler<HTMLDivElement>>(event => {
    // The overlay's chrome and its stable guest are DOM siblings, but one
    // hover boundary. Crossing between them must not dismiss an unpinned pane.
    const next = event.relatedTarget

    if (next instanceof Element && next.closest('[data-narrow-overlay], [data-pane-overlay]')) {
      return
    }

    if (!revealRef.current?.pinned) {
      hoverArmed.current = false
    }

    setReveal(current => (current?.pinned ? current : null))
  }, [])

  // Click-outside dismissal for hover reveals: a hover reveal is ambient, so
  // working in the chat should close it. A pinned reveal (⌘B / titlebar
  // 'open') is explicit intent and survives clicks elsewhere.
  const revealActive = reveal !== null
  useEffect(() => {
    if (!revealActive) {
      return
    }

    const onPointerDown = (event: PointerEvent) => {
      const target = event.target

      if (target instanceof Element && target.closest('[data-narrow-overlay], [data-pane-overlay]')) {
        return
      }

      setReveal(current => (current?.pinned ? current : null))
    }

    window.addEventListener('pointerdown', onPointerDown, true)

    return () => window.removeEventListener('pointerdown', onPointerDown, true)
  }, [revealActive])

  // Own an Escape layer only while something is revealed, so Escape closes the
  // overlay only when it's the top layer (never under a dialog / edit mode).
  useEffect(() => (revealActive ? pushEscapeLayer(ESCAPE_PRIORITY.narrowOverlay) : undefined), [revealActive])

  const inTree = useMemo(() => new Set(tree ? allPaneIds(tree) : []), [tree])

  const collapsibles = useMemo(
    // Solo adopts sidebar panes without their surrounding sidebar chrome.
    // Suppress every reveal path while those panes are intentionally hidden.
    () => (solo ? [] : panes.filter(p => paneChrome(p).collapsible && inTree.has(p.id) && !hiddenPanes.has(p.id))),
    [solo, panes, inTree, hiddenPanes]
  )

  const collapsiblesRef = useRef(collapsibles)
  collapsiblesRef.current = collapsibles

  // ⌘B / ⌘G's narrow branch dispatches the app's toggle-reveal event with the
  // REAL pane id — accept those via each contribution's revealAliases.
  useEffect(() => {
    if (!narrow || solo) {
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
  }, [narrow, solo])

  if (!narrow || solo || collapsibles.length === 0) {
    return null
  }

  const sideOf = (c: Contribution) => (paneChrome(c).placement === 'left' ? 'left' : 'right')
  const revealed = reveal ? collapsibles.find(p => p.id === reveal.id) : undefined
  const sides = [...new Set(collapsibles.map(sideOf))]

  // Size the overlay the way the pane's zone is sized while docked: declared
  // width refined by the user's drag override (fixedTrackSize), so a pane the
  // user narrowed stays narrowed here too — reading only data.width would
  // reset the overlay to the declared size on every reveal.
  const overlayWidth = revealed
    ? narrowOverlayWidth(
        { paneFor: id => panes.find(p => p.id === id), paneGone: () => false, overrides: paneStates },
        tree,
        revealed
      )
    : null

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
    <>
      {/* Hover-intent strips on each edge that has a collapsed pane. */}
      {sides.map(side => (
        <div
          className={cn('absolute inset-y-0 z-30 w-1.5', side === 'left' ? 'left-0' : 'right-0')}
          data-narrow-edge={side}
          key={side}
          onMouseEnter={() => {
            if (!hoverArmed.current) {
              return
            }

            const first = collapsibles.find(p => sideOf(p) === side)

            if (first) {
              setReveal(current => (current?.pinned ? current : { id: first.id, pinned: false }))
            }
          }}
          // Leaving the edge zone re-arms hover intent (see hoverArmed).
          onMouseLeave={() => {
            hoverArmed.current = true
          }}
        />
      ))}

      {revealed && (
        <div
          className={cn(
            'absolute inset-y-0 z-40 flex flex-col overflow-hidden bg-(--ui-sidebar-surface-background) shadow-2xl',
            sideOf(revealed) === 'left'
              ? 'left-0 border-r border-(--ui-stroke-secondary)'
              : 'right-0 border-l border-(--ui-stroke-secondary)'
          )}
          // Floats OVER the layout, so under glass its surface must mask the
          // panes beneath it — a see-through overlay reads as text bleeding
          // through text. Contract: `[data-glass-opaque]` in styles.css.
          data-glass-opaque=""
          data-narrow-overlay={revealed.id}
          onMouseLeave={onMouseLeave}
          ref={overlayRef}
          // Match the pane's docked width (sessions ~237px, files its rail
          // width) instead of a fat fixed 20rem — capped for tiny screens.
          // paddingTop keeps the tab strip below the native window controls
          // (macOS traffic lights); the spacer above keeps that band
          // draggable, mirroring TreeGroup's reservation.
          style={{
            paddingTop: wcOverlap ? wcOverlap.y + wcOverlap.height : undefined,
            width: `min(${overlayWidth}, 85vw)`
          }}
        >
          {wcOverlap && (
            <div
              aria-hidden="true"
              className="pointer-events-none absolute z-10 [-webkit-app-region:drag]"
              style={{ height: wcOverlap.height, left: wcOverlap.x, top: wcOverlap.y, width: wcOverlap.width }}
            />
          )}
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
          {stableHosts && paneChrome(revealed).lifecycleKeepAlive ? (
            <div className="relative min-h-0 min-w-0 flex-1">
              <KeepAlivePaneSlot
                groupId={(tree && findGroupOfPane(tree, revealed.id)?.id) || NO_PANE_GROUP}
                headerVisible={zonePanes.length > 1}
                onMouseLeave={onMouseLeave}
                overlay
                paneId={revealed.id}
                visible
              />
            </div>
          ) : (
            <ContribBoundary id={revealed.id}>
              {revealed.render && <ContribRender render={revealed.render} />}
            </ContribBoundary>
          )}
        </div>
      )}
    </>
  )
}
