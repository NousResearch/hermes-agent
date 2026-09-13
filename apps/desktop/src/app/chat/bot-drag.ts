/**
 * Bot roster drag — the Bot Chat RESOLVER over the shared pointer drag
 * session (`components/pane-shell/tree/renderer/drag-session.ts`). The same
 * machinery as the sidebar session drag (`app/chat/session-drag.ts`):
 * identical ghost chip, zone sheets, strip caret, edge bands, Esc handling
 * and threshold logic, so dragging a bot from the Bots roster feels exactly
 * like dragging a session row from Sessions.
 *
 * Lives in core (not the plugin) because the resolver must call
 * `openSessionTile` — the pane tree's own API — and the plugin fence keeps
 * `hermes-bots` from importing `@/` internals. The SDK re-exports
 * `startBotChatDrag`, and the plugin row wires it on.
 *
 * Drop language (a SESSION's, not a tab move — a drag from the roster never
 * relocates an existing tab):
 *
 *   - over a chat zone's TAB STRIP  → stack: the Bot Chat opens as a tab in
 *     that zone (the strip caret shows the slot);
 *   - over a chat zone's EDGE band  → split: the Bot Chat docks on that edge;
 *   - over a chat zone's CENTER     → if a chat surface is under the pointer,
 *     link: an `@session` chip for the bot's chat drops into that composer
 *     (the same verb as a session drag over a composer); else split center;
 *   - anything else (the roster itself, terminal, gutters) → deny.
 *
 * A drag from the roster is never a MOVE of an existing tab: the payload
 * carries the workspace scope the tile should live in (the plugin computes
 * it with the same rule as its open path), so the minted tile stays owned
 * by the Bot workspace — exactly like a tab opened by clicking the row.
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
import { $treeDragging, type DropHint, revealTreePane, SESSION_TILE_DRAG } from '@/components/pane-shell/tree/store'
import type { EngineZone, ZoneRect } from '@/components/pane-shell/tree/zones-engine'
import { openSessionTile, type SessionTileWorkspaceScope, type TileDock } from '@/store/session-states'

import { requestComposerInsertRefs } from './composer/focus'
import { type SessionDragPayload, sessionInlineRef, sessionLabel } from './composer/inline-refs'
import { tileZoneHost } from './tile-zone-host'

/** Everything the resolver needs to open the bot's canonical Bot Chat with
 *  the same workspace identity a roster-row click would give it. */
export interface BotChatDragPayload extends SessionDragPayload {
  scope?: SessionTileWorkspaceScope
}

/** A chat surface's drag-start geometry: the anchor pane id it advertises
 *  (`data-session-anchor`) and the composer a link drop routes to
 *  (`data-composer-target`). Mirrors session-drag's own snapshot. */
interface BotSurfaceSnapshot {
  anchor: string
  composerTarget: string
  rect: ZoneRect
}

const snapRect = (el: HTMLElement): ZoneRect => {
  const r = el.getBoundingClientRect()

  return { left: r.left, top: r.top, right: r.right, bottom: r.bottom }
}

function snapshotBotSurfaces(): BotSurfaceSnapshot[] {
  return queryAllVisible('[data-session-anchor]').map(el => ({
    anchor: el.dataset.sessionAnchor || 'workspace',
    composerTarget: el.dataset.composerTarget || 'main',
    rect: snapRect(el)
  }))
}

/**
 * Begin dragging a BOT CHAT from the roster row. Same contract as
 * {@link startSessionDrag}: sub-threshold releases stay ordinary clicks (so
 * the row's own open click is untouched), Esc aborts instantly, and the
 * drag dims its source row while lifted.
 */
export function startBotChatDrag(payload: BotChatDragPayload, e: ReactPointerEvent<HTMLElement>) {
  let zones: EngineZone[] = []
  let strips: StripSnapshot[] = []
  let surfaces: BotSurfaceSnapshot[] = []
  let composers: ZoneRect[] = []
  let zoneHost = new Map<string, ReturnType<typeof tileZoneHost>>()

  let split: { anchor: string; before?: null | string; pos: TileDock } | null = null
  let link: null | string = null

  const source = e.currentTarget
  const restoreOpacity = source?.style.opacity ?? ''

  startDragSession(e, {
    ghost: { label: sessionLabel(payload) },
    onTap() {
      // The roster row owns its own click (openRosterBot); a sub-threshold
      // press on the row must keep that behavior, so nothing here.
    },

    onEngage() {
      zones = snapshotZones()
      strips = snapshotStrips()
      surfaces = snapshotBotSurfaces()
      composers = queryAllVisible('[data-slot="composer-root"]').map(snapRect)
      zoneHost = new Map(zones.map(zone => [zone.id, tileZoneHost(zone.id)]))
      source?.style.setProperty('opacity', '0.45')
      $treeDragging.set(SESSION_TILE_DRAG)
    },

    onEnd() {
      if (source) {
        source.style.opacity = restoreOpacity
      }
    },

    resolveMove(x, y): DropHint | null {
      const zone = zones.find(z => rectContains(z.rect, x, y))
      const zoneHostEntry = zone ? zoneHost.get(zone.id) : null

      if (!zone || !zoneHostEntry) {
        split = null
        link = null

        return null
      }

      const strip = strips.find(s => s.groupId === zone.id && rectContains(s.rect, x, y))

      if (strip) {
        const stack = slotBefore(strip.slots, x, `session-tile:${payload.id}`)
        split = { anchor: zoneHostEntry.pane, before: stack.before, pos: 'center' }
        link = null

        return { kind: 'group', groupId: zone.id, groupIds: [zone.id], pos: 'center', stack }
      }

      const pos = composers.some(rect => rectContains(rect, x, y)) ? 'center' : subZonePosition(zones, zone.id, x, y)
      const surface = surfaces.find(s => rectContains(s.rect, x, y))

      if (pos === 'center' && zoneHostEntry.chat) {
        split = null
        link = surface?.composerTarget ?? 'main'
      } else if (pos === 'center') {
        split = { anchor: zoneHostEntry.pane, pos: 'center' }
        link = null
      } else {
        split = { anchor: surface?.anchor ?? zoneHostEntry.pane, pos }
        link = null
      }

      return { kind: 'group', groupId: zone.id, groupIds: [zone.id], pos }
    },

    onCommit() {
      if (split) {
        // The scope travels in the payload (computed by the plugin with the
        // same rule as its open path), so the minted tile lands in the Bot
        // workspace instead of being re-bucketed into Sessions.
        openSessionTile(payload.id, split.pos, split.anchor, split.before, payload.scope)
        revealTreePane(`session-tile:${payload.id}`)
      } else if (link) {
        requestComposerInsertRefs([sessionInlineRef(payload)], { target: link })
      }
    }
  })
}
