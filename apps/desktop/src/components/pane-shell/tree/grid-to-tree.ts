/**
 * Grid -> tree bridge. A FancyZones grid whose zones can be produced by
 * recursive guillotine cuts (every FancyZones template, and almost every
 * practical layout) converts exactly to our runtime `LayoutNode` tree:
 * full-length cut lines become splits with weights from the cut positions.
 * Non-guillotine arrangements (interlocking pinwheels) return null and the
 * editor disables Save with an explanation.
 */

import type { GridLayout, GridZone } from './grid-model'
import { modelToZones } from './grid-model'
import { group, type LayoutNode, normalize, split } from './model'

function cutCandidates(zones: GridZone[], axis: 'x' | 'y'): number[] {
  const coords = new Set(zones.flatMap(z => (axis === 'x' ? [z.left, z.right] : [z.top, z.bottom])))

  // Interior lines only.
  return [...coords].sort((a, b) => a - b).slice(1, -1)
}

function isValidCut(zones: GridZone[], axis: 'x' | 'y', at: number): boolean {
  return zones.every(zone => (axis === 'x' ? zone.right <= at || zone.left >= at : zone.bottom <= at || zone.top >= at))
}

/**
 * Recursively cut the zone set. Collects ALL valid cuts on the chosen axis at
 * once so "three columns" becomes one flat 3-child split, not nested pairs.
 */
function cut(zones: GridZone[], assignPane: (zoneIndex: number) => string[]): LayoutNode | null {
  if (zones.length === 1) {
    // Zones without panes become EMPTY groups — editor-authored drop targets
    // that live until the first structural op prunes them (normalize).
    return group(assignPane(zones[0].index))
  }

  for (const axis of ['x', 'y'] as const) {
    const cuts = cutCandidates(zones, axis).filter(at => isValidCut(zones, axis, at))

    if (cuts.length === 0) {
      continue
    }

    const start = Math.min(...zones.map(z => (axis === 'x' ? z.left : z.top)))
    const end = Math.max(...zones.map(z => (axis === 'x' ? z.right : z.bottom)))
    const lines = [start, ...cuts, end]
    const children: LayoutNode[] = []
    const weights: number[] = []

    for (let i = 0; i < lines.length - 1; i++) {
      const lo = lines[i]
      const hi = lines[i + 1]

      const slice = zones.filter(zone =>
        axis === 'x' ? zone.left >= lo && zone.right <= hi : zone.top >= lo && zone.bottom <= hi
      )

      if (slice.length === 0) {
        continue
      }

      const child = cut(slice, assignPane)

      if (child) {
        children.push(child)
        weights.push(hi - lo)
      }
    }

    if (children.length === 0) {
      return null
    }

    if (children.length === 1) {
      return children[0]
    }

    return split(axis === 'x' ? 'row' : 'column', children, weights)
  }

  // No full-length cut exists on either axis: non-guillotine (pinwheel).
  return null
}

export type PanePlacementHint = 'main' | 'left' | 'right' | 'top' | 'bottom'

export interface PlacedPane {
  id: string
  placement?: PanePlacementHint
}

const CENTER = 5000 // MULTIPLIER / 2

interface ZoneGeo {
  index: number
  area: number
  cx: number
  cy: number
}

/**
 * Semantic zone assignment: panes claim zones by ROLE, matched on geometry —
 * `main` takes the largest zone, `left`/`right`/`top`/`bottom` take zones whose
 * centroid actually sits on that side (with a small size tiebreak). A hinted
 * pane with no acceptable zone left STACKS with its role-mates (or main)
 * instead of squatting in some random cell; unhinted panes fill what remains
 * biggest-first; extra zones stay empty.
 */
function assignZones(zones: GridZone[], panes: PlacedPane[]): Map<number, string[]> {
  const geo: ZoneGeo[] = zones.map(z => ({
    index: z.index,
    area: (z.right - z.left) * (z.bottom - z.top),
    cx: (z.left + z.right) / 2,
    cy: (z.top + z.bottom) / 2
  }))

  const remaining = new Map(geo.map(g => [g.index, g]))
  const assignments = new Map<number, string[]>()
  const zoneForRole = new Map<string, number>()

  // Score = fit for the role; accept = the zone genuinely sits on that side.
  const roles: Record<PanePlacementHint, { accept: (g: ZoneGeo) => boolean; score: (g: ZoneGeo) => number }> = {
    main: { accept: () => true, score: g => g.area },
    left: { accept: g => g.cx < CENTER, score: g => CENTER - g.cx + g.area / 1e8 },
    right: { accept: g => g.cx > CENTER, score: g => g.cx - CENTER + g.area / 1e8 },
    top: { accept: g => g.cy < CENTER, score: g => CENTER - g.cy + g.area / 1e8 },
    bottom: { accept: g => g.cy > CENTER, score: g => g.cy - CENTER + g.area / 1e8 }
  }

  const claim = (pane: PlacedPane, role: PanePlacementHint | '_') => {
    const spec = role === '_' ? roles.main : roles[role]
    let best: ZoneGeo | null = null

    for (const g of remaining.values()) {
      if (spec.accept(g) && (!best || spec.score(g) > spec.score(best))) {
        best = g
      }
    }

    if (best) {
      remaining.delete(best.index)
      assignments.set(best.index, [pane.id])
      zoneForRole.set(role, best.index)

      if (role === 'main' || !zoneForRole.has('main')) {
        zoneForRole.set('main', zoneForRole.get('main') ?? best.index)
      }

      return
    }

    // No acceptable zone left: stack with role-mates, else with main, else last.
    const home = zoneForRole.get(role) ?? zoneForRole.get('main') ?? [...assignments.keys()].pop()

    if (home !== undefined) {
      assignments.get(home)?.push(pane.id)
    }
  }

  // Placement priority: main anchors first, then sided panes, then the rest.
  const rank = (p: PlacedPane) => (p.placement === 'main' ? 0 : p.placement ? 1 : 2)

  for (const pane of [...panes].sort((a, b) => rank(a) - rank(b))) {
    claim(pane, pane.placement ?? '_')
  }

  return assignments
}

/**
 * Convert a grid to a tree. Panes carry placement hints (from their
 * contribution's `data.placement`) and land in geometrically fitting zones;
 * zones beyond the pane count stay as EMPTY zones.
 */
export function gridToTree(gridModel: GridLayout, panes: PlacedPane[]): LayoutNode | null {
  const zones = modelToZones(gridModel)

  if (!zones || zones.length === 0 || panes.length === 0) {
    return null
  }

  const assignments = assignZones(zones, panes)
  const result = cut(zones, zoneIndex => assignments.get(zoneIndex) ?? [])

  return result ? normalize(result) : null
}

/** True when the grid is expressible as a tree (guillotine-cuttable). */
export function gridIsTreeExpressible(gridModel: GridLayout): boolean {
  const zones = modelToZones(gridModel)

  if (!zones) {
    return false
  }

  const probe = cut(zones, () => ['probe'])

  return probe !== null
}
