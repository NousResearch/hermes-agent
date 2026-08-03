/**
 * Sidebar session drag — the session RESOLVER over the shared pointer drag
 * session (pane-shell drag-session.ts). Same machinery as a pane drag
 * (threshold, rAF moves, snapshots, Esc-as-top-layer with synchronous
 * teardown), session-specific targeting:
 *
 *   - a chat zone's TAB STRIP  → stack: open the session as a tab at the
 *     divider's slot (the strip caret shows it);
 *   - a chat zone's EDGE band  → split: open the session as a tile docked on
 *     that edge (the zone sheet morphs to the half);
 *   - a chat zone's CENTER / the composer → link: insert an `@session` chip
 *     into that surface's composer (ChatDropOverlay owns the visual);
 *   - anything else (sidebar, terminal, gutters) → deny.
 *
 * Zones that don't host a chat surface are NOT targets — the overlay never
 * lights them, so a release there must not commit either (one truth).
 *
 * This replaced the native-HTML5 drag + SessionTileDropBridge: riding the
 * native DnD layer meant macOS's cancel snap-back animation, a `dragend`
 * held hostage until that animation finished, an Esc the page never even
 * saw, and window-level armor against react-dnd/dnd-kit. A pointer session
 * has none of those failure modes. Native DnD remains only at the true OS
 * boundary (Finder file drops). Known trade: a session can no longer be
 * dragged into a separate BrowserWindow (native DnD was the only transport
 * that crossed windows).
 */

import type { PointerEvent as ReactPointerEvent } from 'react'

import { findGroup } from '@/components/pane-shell/tree/model'
import {
  type DoubleTapContext,
  rectContains,
  slotBefore,
  snapshotStrips,
  snapshotZones,
  startDragSession,
  type StripSnapshot,
  subZonePosition
} from '@/components/pane-shell/tree/renderer/drag-session'
import {
  $layoutTree,
  $treeDragging,
  type DropHint,
  revealTreePane,
  SESSION_TILE_DRAG
} from '@/components/pane-shell/tree/store'
import type { EngineZone, ZoneRect } from '@/components/pane-shell/tree/zones-engine'
import { openSessionTile, type TileDock } from '@/store/session-states'

import { requestComposerInsertRefs } from './composer/focus'
import { type SessionDragPayload, sessionInlineRef, sessionLabel } from './composer/inline-refs'

/** A chat surface's drag-start geometry: the anchor pane id it advertises
 *  (`data-session-anchor`) and the composer a link drop routes to
 *  (`data-composer-target`). */
interface SurfaceSnapshot {
  anchor: string
  composerTarget: string
  rect: ZoneRect
}

const snapRect = (el: HTMLElement): ZoneRect => {
  const r = el.getBoundingClientRect()

  return { left: r.left, top: r.top, right: r.right, bottom: r.bottom }
}

function snapshotSurfaces(): SurfaceSnapshot[] {
  return [...document.querySelectorAll<HTMLElement>('[data-session-anchor]')].map(el => ({
    anchor: el.dataset.sessionAnchor || 'workspace',
    composerTarget: el.dataset.composerTarget || 'main',
    rect: snapRect(el)
  }))
}

/** A session may land in a zone only if it hosts a chat surface — never the
 *  sidebar/terminal zones. Returns the pane a stack anchors to. */
function chatZonePane(groupId: string): null | string {
  const tree = $layoutTree.get()
  const panes = tree ? (findGroup(tree, groupId)?.panes ?? []) : []

  return panes.find(p => p === 'workspace' || p.startsWith('session-tile:')) ?? null
}

/**
 * Begin dragging a session — a sidebar row OR a tile's own tab (same drop
 * language either way: stack, split, or composer link). Sub-threshold releases
 * stay ordinary clicks, so `opts.onTap` (activate the tile) and `opts.double`
 * (hide the tab bar) ride the tab's gestures; Esc aborts instantly. A stack/
 * split commits through `openSessionTile`, which OPENS a new tile from a sidebar
 * row and MOVES the existing one when its tab is the drag source.
 */
export function startSessionDrag(
  payload: SessionDragPayload,
  e: ReactPointerEvent<HTMLElement>,
  opts?: { double?: DoubleTapContext; onTap?: () => void }
) {
  let zones: EngineZone[] = []
  let strips: StripSnapshot[] = []
  let surfaces: SurfaceSnapshot[] = []
  let composers: ZoneRect[] = []
  let zoneHost = new Map<string, null | string>()

  // Commit intent, updated per resolved move (the machinery flushes the final
  // move before commit, so these always match the released-at position).
  let split: { anchor: string; before?: null | string; pos: TileDock } | null = null
  let link: null | string = null

  // The drag SOURCE (sidebar row or tile tab). Captured synchronously — React
  // clears `currentTarget` after the pointerdown handler returns, but this runs
  // inside it. Dimmed while lifted so the source reads as "picked up" — the
  // same in-place feedback pane-tab drags use, replacing the old cursor chip.
  const source = e.currentTarget
  const restoreOpacity = source?.style.opacity ?? ''

  startDragSession(e, {
    double: opts?.double,
    ghost: { label: sessionLabel(payload) },
    onTap: opts?.onTap,

    onEngage() {
      zones = snapshotZones()
      strips = snapshotStrips()
      surfaces = snapshotSurfaces()
      composers = [...document.querySelectorAll<HTMLElement>('[data-slot="composer-root"]')].map(snapRect)
      zoneHost = new Map(zones.map(zone => [zone.id, chatZonePane(zone.id)]))
      source?.style.setProperty('opacity', '0.45')
      // The same sentinel the zone overlay + chat surfaces key off — the
      // whole drop language (sheets, pills, caret, link overlay) lights up.
      $treeDragging.set(SESSION_TILE_DRAG)
    },

    onEnd() {
      if (source) {
        source.style.opacity = restoreOpacity
      }
    },

    resolveMove(x, y): DropHint | null {
      const zone = zones.find(z => rectContains(z.rect, x, y))
      const host = zone ? zoneHost.get(zone.id) : null

      if (!zone || !host) {
        split = null
        link = null

        return null
      }

      // The zone's TAB STRIP stacks the session at the divider's slot.
      const strip = strips.find(s => s.groupId === zone.id && rectContains(s.rect, x, y))

      if (strip) {
        // Exclude the tile's OWN tab from the slots so re-dropping it in its
        // home strip reorders cleanly (a no-op for a sidebar-row drag).
        const stack = slotBefore(strip.slots, x, `session-tile:${payload.id}`)
        split = { anchor: host, before: stack.before, pos: 'center' }
        link = null

        return { kind: 'group', groupId: zone.id, groupIds: [zone.id], pos: 'center', stack }
      }

      // The composer (and everything in it) is always the link/attach drop;
      // elsewhere the shared radial targeting decides center vs edge.
      const pos = composers.some(rect => rectContains(rect, x, y)) ? 'center' : subZonePosition(zones, zone.id, x, y)
      const surface = surfaces.find(s => rectContains(s.rect, x, y))

      if (pos === 'center') {
        split = null
        link = surface?.composerTarget ?? 'main'
      } else {
        split = { anchor: surface?.anchor ?? 'workspace', pos }
        link = null
      }

      return { kind: 'group', groupId: zone.id, groupIds: [zone.id], pos }
    },

    onCommit() {
      if (split) {
        openSessionTile(payload.id, split.pos, split.anchor, split.before)
        // A tile for this session may already exist (openSessionTile is
        // idempotent — e.g. persisted from an earlier run): a drop must never
        // feel dead, so front/unhide/un-dismiss it either way.
        revealTreePane(`session-tile:${payload.id}`)
      } else if (link) {
        // The "link to chat" drop: an @session chip in that surface's composer.
        requestComposerInsertRefs([sessionInlineRef(payload)], { target: link })
      }
    }
  })
}
