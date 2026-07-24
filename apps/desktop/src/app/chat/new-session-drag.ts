/**
 * Sidebar "New session" drag — the NEW-session resolver over the shared
 * pointer drag session (pane-shell drag-session.ts). Same machinery and drop
 * language as a session drag (session-drag.ts): a chat zone's TAB STRIP stacks,
 * a chat zone's EDGE splits, and the CENTER stacks a fresh tab.
 *
 * The ONE deliberate difference from a session drag: there is no "link to chat"
 * (composer `@session` chip) target. A session that doesn't exist yet can't be
 * linked, so a center drop creates + stacks instead. This is why the drag rides
 * the distinct NEW_SESSION_DRAG sentinel (not SESSION_TILE_DRAG): the overlay's
 * `sessionDrag` checks that gate the link affordance stay false here, so the
 * zone sheet shows its normal "stack here" wash over a chat center and the
 * composer's "Drop to link this chat" overlay never lights — with zero edits to
 * those hot overlay paths.
 *
 * Create-on-commit: the session is only created when the drag commits over a
 * valid target. A sub-threshold release stays an ordinary click (the button's
 * own onClick), an Esc abort or a release on a deny zone creates nothing — no
 * orphan empty sessions.
 */

import type { PointerEvent as ReactPointerEvent } from 'react'

import { findGroup } from '@/components/pane-shell/tree/model'
import {
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
  NEW_SESSION_DRAG
} from '@/components/pane-shell/tree/store'
import type { EngineZone, ZoneRect } from '@/components/pane-shell/tree/zones-engine'
import { translateNow } from '@/i18n'
import type { TileDock } from '@/store/session-states'

/** Where a dragged new session lands. `center` stacks a fresh tab into the
 *  anchor's zone (optionally at a strip slot via `before`); an edge dir splits
 *  a new tile docked to that edge of the anchor. */
export interface NewSessionPlacement {
  anchor: string
  before?: null | string
  dir: TileDock
}

const snapRect = (el: HTMLElement): ZoneRect => {
  const r = el.getBoundingClientRect()

  return { bottom: r.bottom, left: r.left, right: r.right, top: r.top }
}

interface SurfaceSnapshot {
  anchor: string
  rect: ZoneRect
}

function snapshotSurfaces(): SurfaceSnapshot[] {
  return [...document.querySelectorAll<HTMLElement>('[data-session-anchor]')].map(el => ({
    anchor: el.dataset.sessionAnchor || 'workspace',
    rect: snapRect(el)
  }))
}

/** A new session may land in a zone only if it hosts a chat surface — never the
 *  sidebar/terminal zones. Returns the pane a stack anchors to. (Identical gate
 *  to a session drag: a new chat is a chat, so it stacks/splits relative to the
 *  existing chat surfaces only.) */
function chatZonePane(groupId: string): null | string {
  const tree = $layoutTree.get()
  const panes = tree ? (findGroup(tree, groupId)?.panes ?? []) : []

  return panes.find(p => p === 'workspace' || p.startsWith('session-tile:')) ?? null
}

/**
 * Begin dragging a brand-new session from the sidebar's "New session" row. The
 * drop language mirrors a session drag (stack / split), but commit CREATES the
 * session at the resolved placement via `onCreate` rather than moving an
 * existing one. Sub-threshold releases stay ordinary clicks (`opts.onTap`), so
 * the row's normal new-session action is untouched; Esc aborts instantly and
 * creates nothing.
 */
export function startNewSessionDrag(
  onCreate: (placement: NewSessionPlacement) => void,
  e: ReactPointerEvent<HTMLElement>,
  opts?: { onTap?: () => void }
) {
  let zones: EngineZone[] = []
  let strips: StripSnapshot[] = []
  let surfaces: SurfaceSnapshot[] = []
  let composers: ZoneRect[] = []
  let zoneHost = new Map<string, null | string>()

  // Commit intent, updated per resolved move (the machinery flushes the final
  // move before commit, so this always matches the released-at position).
  let placement: NewSessionPlacement | null = null

  // The drag SOURCE (the "New session" row). Dimmed while lifted so it reads as
  // "picked up" — the same in-place feedback a sidebar session row uses.
  const source = e.currentTarget
  const restoreOpacity = source?.style.opacity ?? ''

  startDragSession(e, {
    ghost: { label: translateNow('sidebar.nav.new-session') },
    onTap: opts?.onTap,

    onEngage() {
      zones = snapshotZones()
      strips = snapshotStrips()
      surfaces = snapshotSurfaces()
      composers = [...document.querySelectorAll<HTMLElement>('[data-slot="composer-root"]')].map(snapRect)
      zoneHost = new Map(zones.map(zone => [zone.id, chatZonePane(zone.id)]))
      source?.style.setProperty('opacity', '0.45')
      // The distinct sentinel: the zone overlay lights its normal targets, but
      // the "link to chat" affordance (gated on SESSION_TILE_DRAG) stays dark.
      $treeDragging.set(NEW_SESSION_DRAG)
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
        placement = null

        return null
      }

      // The zone's TAB STRIP stacks the new session at the divider's slot.
      const strip = strips.find(s => s.groupId === zone.id && rectContains(s.rect, x, y))

      if (strip) {
        const stack = slotBefore(strip.slots, x)
        placement = { anchor: host, before: stack.before, dir: 'center' }

        return { groupId: zone.id, groupIds: [zone.id], kind: 'group', pos: 'center', stack }
      }

      // Over the composer (and everything in it) counts as the zone CENTER —
      // dropping on a chat's input stacks into that chat, never splits below it.
      const surface = surfaces.find(s => rectContains(s.rect, x, y))
      const anchor = surface?.anchor ?? host
      const pos = composers.some(rect => rectContains(rect, x, y)) ? 'center' : subZonePosition(zones, zone.id, x, y)

      if (pos === 'center') {
        placement = { anchor, dir: 'center' }
      } else {
        placement = { anchor, dir: pos }
      }

      return { groupId: zone.id, groupIds: [zone.id], kind: 'group', pos }
    },

    onCommit() {
      if (!placement) {
        return
      }

      // The create path (openNewSessionTile) owns the post-create reveal — it
      // round-trips session.create, then revealTreePane's the fresh tile. A
      // commit with no placement (release on a deny zone) creates nothing.
      onCreate(placement)
    }
  })
}
