import Dagre from '@dagrejs/dagre'
import type { Edge, Node, XYPosition } from '@xyflow/react'

// "Tidy up" — auto-arrange the graph left→right with Dagre (React Flow's
// official layouting recipe). The loop-back edge is excluded so the DAG stays a
// clean forward flow.
//
// Nodes with no edges aren't part of the flow, so Dagre would stack them all at
// the origin. They're parked in a column to the left of the graph instead,
// clear of everything Dagre placed.

// Kept in step with --node-w (13.25rem = 212px) so a freshly added card lands
// close to where the real layout will put it once it's measured. Also what
// add-step centres a dropped card on, for the same reason.
export const CARD_W = 212
export const FALLBACK_H = 92
/** Dagre ranksep — the empty hop between two ranks. n8n's NODE_X_SPACING. */
export const RANK_GAP = 120
const FALLBACK_W = CARD_W
const ORPHAN_GAP = 64

/** One rank over: a card's width plus the gap Dagre would leave. */
export const STEP = CARD_W + RANK_GAP

// Keep the graph clear of the floating chrome that is ALWAYS there: the brand
// mark up top, the timeline + composer along the bottom, the live log's lane on
// the right. The inspector is deliberately NOT reserved for — it floats over
// the canvas and the graph stays put, because re-framing the whole graph every
// time you open a panel is worse than the overlap it avoids.
export const FIT = {
  // The brand panel bottoms out at 66px (16px margin + 51px tall), so 56px
  // left the top rank grazing it.
  padding: { top: '78px', right: '150px', bottom: '208px', left: '40px' }
} as const

export const widthOf = (n: Node) => n.measured?.width ?? CARD_W
export const heightOf = (n: Node) => n.measured?.height ?? FALLBACK_H

/** x is a left edge, y is a centre — the canvas's nodeOrigin. */
const hits = (n: Node, at: XYPosition) =>
  Math.abs(n.position.x + widthOf(n) / 2 - (at.x + CARD_W / 2)) < (widthOf(n) + CARD_W) / 2 &&
  Math.abs(n.position.y - at.y) < (heightOf(n) + FALLBACK_H) / 2

/** n8n's getNewNodePosition: if a card would land on top of one that's already
 *  there, step one rank right until it doesn't. A small diagonal cascade isn't
 *  enough — 28px still leaves the new card inside the old one. This is the rule
 *  for a card you DROPPED, where you chose the y and only the x may give. */
export function freeSpot(nodes: Node[], at: XYPosition, dir: FlowDir = DEFAULT_DIR): XYPosition {
  const spot = { ...at }

  for (let i = 0; i < 40 && nodes.some(n => hits(n, spot)); i++) {
    if (dir === 'TB') {
      spot.y += FALLBACK_H + RANK_GAP
    } else {
      spot.x += STEP
    }
  }

  return spot
}

/** The rule for a card WIRED to another: its rank is decided by the wire, so
 *  the rank is fixed and it settles ACROSS past its siblings instead. Marching
 *  it along the flow would push a validator past the steps it feeds and lose
 *  the reading the ranks exist to give. */
export function freeRow(nodes: Node[], at: XYPosition, dir: FlowDir = DEFAULT_DIR): XYPosition {
  const spot = { ...at }

  for (let i = 0; i < 40; i++) {
    const clash = nodes.find(n => hits(n, spot))

    if (!clash) {
      break
    }

    if (dir === 'TB') {
      spot.x = clash.position.x + widthOf(clash) + ORPHAN_GAP
    } else {
      spot.y = clash.position.y + heightOf(clash) / 2 + FALLBACK_H / 2 + ORPHAN_GAP
    }
  }

  return spot
}

/** Which way the ranks run. Dagre's `rankdir` verbatim — the canvas doesn't
 *  have a second word for it. */
export type FlowDir = 'LR' | 'TB'

/** Top to bottom, because a workflow is a list of steps before it's a diagram
 *  and that's the direction a list is read in. The pane is also taller than it
 *  is wide once the inspector opens, so the ranks have further to run. */
export const DEFAULT_DIR: FlowDir = 'TB'

export function tidyLayout(nodes: Node[], edges: Edge[], dir: FlowDir = DEFAULT_DIR): Node[] {
  const wired = new Set<string>()

  for (const e of edges) {
    wired.add(e.source)
    wired.add(e.target)
  }

  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: dir, nodesep: 64, ranksep: RANK_GAP, marginx: 24, marginy: 24 })

  for (const n of nodes) {
    if (!wired.has(n.id)) {
      continue
    }

    g.setNode(n.id, {
      width: n.measured?.width ?? FALLBACK_W,
      height: n.measured?.height ?? FALLBACK_H
    })
  }

  for (const e of edges) {
    const d = e.data as { loop?: boolean } | undefined

    if (d?.loop) {
      continue
    }

    g.setEdge(e.source, e.target)
  }

  Dagre.layout(g)

  // Where the laid-out graph starts, so orphans can sit clear of its left edge.
  let minX = Infinity
  let minY = Infinity

  for (const n of nodes) {
    if (!wired.has(n.id)) {
      continue
    }

    const p = g.node(n.id)

    if (!p) {
      continue
    }

    minX = Math.min(minX, p.x - p.width / 2)
    minY = Math.min(minY, p.y - p.height / 2)
  }

  if (!Number.isFinite(minX)) {
    minX = 0
  }

  if (!Number.isFinite(minY)) {
    minY = 0
  }

  let orphanY = minY

  return nodes.map(n => {
    if (wired.has(n.id)) {
      const p = g.node(n.id)

      if (!p) {
        return n
      }

      // Dagre reports a node's centre. The canvas runs `nodeOrigin={[0, 0.5]}`,
      // so y is a centre too and passes straight through; only x still needs
      // converting to the left edge.
      return { ...n, position: { x: p.x - p.width / 2, y: p.y } }
    }

    // Orphans stack in a column to the LEFT of the graph, as the note above
    // says — they used to go below it, which is the one band the loop-back
    // needs: that connector leaves the gate downward and arcs back under the
    // whole flow, so anything parked beneath the last rank gets a wire drawn
    // straight through it. To the side, nothing is routed.
    const h = n.measured?.height ?? FALLBACK_H
    // orphanY walks the column's top edge; the position wants the centre.
    const y = orphanY + h / 2
    orphanY += h + ORPHAN_GAP
    const w = n.measured?.width ?? FALLBACK_W

    return { ...n, position: { x: minX - w - ORPHAN_GAP * 2, y } }
  })
}
