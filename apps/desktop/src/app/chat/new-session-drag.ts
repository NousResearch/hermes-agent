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

import { queryAllVisible } from '@/components/pane-shell/pane-visibility'
import {
  rectContains,
  slotBefore,
  snapshotStrips,
  snapshotZones,
  startDragSession,
  type StripSnapshot,
  subZonePosition
} from '@/components/pane-shell/tree/renderer/drag-session'
import { $treeDragging, type DropHint, NEW_SESSION_DRAG } from '@/components/pane-shell/tree/store'
import type { EngineZone, ZoneRect } from '@/components/pane-shell/tree/zones-engine'
import { translateNow } from '@/i18n'
import type { AgentProfileRoute } from '@/store/profile'
import type { TileDock } from '@/store/session-states'

import { tileZoneHost } from './tile-zone-host'

/** Create a new session as a tile at a resolved drop target. The shared
 *  handler shape threaded from every create-drag source through the sidebar
 *  into `openNewSessionTile` — one declaration, so the payload cannot drift
 *  between call sites. */
export type NewSessionSplitHandler = (
  dir: TileDock,
  opts?: {
    anchor?: string
    before?: null | string
    cwd?: null | string
    profile?: string
    route?: AgentProfileRoute | null
  }
) => void

/** Where a dragged new session lands. `center` stacks a fresh tab into the
 *  anchor's zone (optionally at a strip slot via `before`); an edge dir splits
 *  a new tile docked to that edge of the anchor. `cwd` pins the new session to
 *  a project's path when the drag started from a project row (null = the
 *  default new-session cwd). `profile` / `route` pins the new session to a
 *  specific profile when dragged from a profile group. */
export interface NewSessionPlacement {
  anchor: string
  before?: null | string
  cwd?: null | string
  dir: TileDock
  profile?: string
  route?: AgentProfileRoute | null
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
  return queryAllVisible('[data-session-anchor]').map(el => ({
    anchor: el.dataset.sessionAnchor || 'workspace',
    rect: snapRect(el)
  }))
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
  opts?: {
    /** Pin the created session to a project's path (drag from a project row).
     *  Omitted/null = the default new-session cwd (drag from "New session"). */
    cwd?: null | string
    /** Ghost chip label — the project's name for a project-row drag, else the
     *  default "New session". */
    label?: string
    /** Pin the created session to a specific profile (drag from a profile group). */
    profile?: string
    route?: AgentProfileRoute | null
    onTap?: () => void
  }
) {
  let zones: EngineZone[] = []
  let strips: StripSnapshot[] = []
  let surfaces: SurfaceSnapshot[] = []
  let composers: ZoneRect[] = []
  let zoneHost = new Map<string, { chat: boolean; pane: string }>()

  // Commit intent, updated per resolved move (the machinery flushes the final
  // move before commit, so this always matches the released-at position).
  let placement: NewSessionPlacement | null = null

  // The drag SOURCE (the "New session" row or a project's + button). Dimmed
  // while lifted so it reads as "picked up" — the same in-place feedback a
  // sidebar session row uses.
  const source = e.currentTarget
  const restoreOpacity = source?.style.opacity ?? ''

  startDragSession(e, {
    ghost: { label: opts?.label || translateNow('sidebar.nav.new-session') },
    onTap: opts?.onTap,

    onEngage() {
      zones = snapshotZones()
      strips = snapshotStrips()
      surfaces = snapshotSurfaces()
      composers = queryAllVisible('[data-slot="composer-root"]').map(snapRect)
      zoneHost = new Map(
        zones.flatMap(z => {
          const host = tileZoneHost(z.id)

          return host ? [[z.id, host]] : []
        })
      )
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
        placement = {
          anchor: host.pane,
          before: stack.before,
          cwd: opts?.cwd,
          dir: 'center',
          profile: opts?.profile,
          route: opts?.route
        }

        return { groupId: zone.id, groupIds: [zone.id], kind: 'group', pos: 'center', stack }
      }

      // Over the composer (and everything in it) counts as the zone CENTER —
      // dropping on a chat's input stacks into that chat, never splits below it.
      const surface = surfaces.find(s => rectContains(s.rect, x, y))
      const anchor = surface?.anchor ?? host.pane
      const pos = composers.some(rect => rectContains(rect, x, y)) ? 'center' : subZonePosition(zones, zone.id, x, y)

      if (pos === 'center') {
        placement = { anchor, cwd: opts?.cwd, dir: 'center', profile: opts?.profile, route: opts?.route }
      } else {
        placement = { anchor, cwd: opts?.cwd, dir: pos, profile: opts?.profile, route: opts?.route }
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

/**
 * Begin dragging a brand-new PROJECT from the project-overview header's
 * "New project" + button. Same machinery and drop language as
 * {@link startNewSessionDrag} — tab strip / pane edge / pane center — but the
 * gesture arms a placement that is CONSUMED BY THE DIALOG FLOW rather than
 * creating anything at release:
 *
 * 1. Engaging the drag records the placement via `onArm`.
 * 2. A sub-threshold release stays an ordinary click (`opts.onTap` → the
 *    project dialog); an Esc abort or a deny-zone release creates nothing and
 *    clears the armed placement.
 * 3. A valid commit opens the exact same "New project" dialog. When that flow
 *    later creates a project, the completion side replays the placement so the
 *    project's fresh session draft opens precisely where it was dropped — and
 *    stays there.
 */
export function startNewProjectDrag(
  onArm: (placement: NewSessionPlacement | null) => void,
  e: ReactPointerEvent<HTMLElement>,
  opts?: { onTap?: () => void }
) {
  let zones: EngineZone[] = []
  let strips: StripSnapshot[] = []
  let surfaces: SurfaceSnapshot[] = []
  let composers: ZoneRect[] = []
  let zoneHost = new Map<string, { chat: boolean; pane: string }>()

  let placement: NewSessionPlacement | null = null

  const source = e.currentTarget
  const restoreOpacity = source?.style.opacity ?? ''

  startDragSession(e, {
    ghost: { label: translateNow('sidebar.projects.newButton') },
    onTap: opts?.onTap,

    onEngage() {
      zones = snapshotZones()
      strips = snapshotStrips()
      surfaces = snapshotSurfaces()
      composers = queryAllVisible('[data-slot="composer-root"]').map(snapRect)
      zoneHost = new Map(
        zones.flatMap(z => {
          const host = tileZoneHost(z.id)

          return host ? [[z.id, host]] : []
        })
      )
      source?.style.setProperty('opacity', '0.45')
      $treeDragging.set(NEW_SESSION_DRAG)
    },

    onEnd() {
      if (source) {
        source.style.opacity = restoreOpacity
      }

      // A drag that never committed (Esc, deny-zone release) leaves no armed
      // placement behind, so a later plain dialog create can't inherit it.
      if (!placement) {
        onArm(null)
      }
    },

    resolveMove(x, y): DropHint | null {
      const zone = zones.find(z => rectContains(z.rect, x, y))
      const host = zone ? zoneHost.get(zone.id) : null

      if (!zone || !host) {
        placement = null

        return null
      }

      const strip = strips.find(s => s.groupId === zone.id && rectContains(s.rect, x, y))

      if (strip) {
        const stack = slotBefore(strip.slots, x)
        placement = { anchor: host.pane, before: stack.before, dir: 'center' }

        return { groupId: zone.id, groupIds: [zone.id], kind: 'group', pos: 'center', stack }
      }

      const surface = surfaces.find(s => rectContains(s.rect, x, y))
      const anchor = surface?.anchor ?? host.pane
      const pos = composers.some(rect => rectContains(rect, x, y)) ? 'center' : subZonePosition(zones, zone.id, x, y)

      if (pos === 'center') {
        placement = { anchor, dir: 'center' }
      } else {
        placement = { anchor, dir: pos }
      }

      return { groupId: zone.id, groupIds: [zone.id], kind: 'group', pos }
    },

    onCommit() {
      // A deny-zone release reads as a cancelled drop — same language as
      // Escape and the new-session drag: nothing happens. The dialog only
      // opens on a VALID commit (with the placement armed); a plain click
      // still reaches it through the drag's tap path.
      if (!placement) {
        return
      }

      // Arm BEFORE opening the dialog: the dialog's create path reads the
      // armed placement when it succeeds, so the created project lands
      // exactly here.
      onArm(placement)
      opts?.onTap?.()
    }
  })
}
