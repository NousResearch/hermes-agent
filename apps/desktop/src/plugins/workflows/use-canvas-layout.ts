// Where the cards sit and where the camera looks.
//
// Positions are Dagre's, not ours — the only thing computed here is WHEN to
// re-run it: once on load the moment every card has a real measured size, and
// again whenever you tidy or flip the direction. After that a card you dragged
// stays dragged.

import { useReactFlow } from '@xyflow/react'
import type { Edge, Node } from '@xyflow/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { DEFAULT_DIR, FIT, type FlowDir, tidyLayout } from './layout'
import { CANVAS_NOTE_ID } from './nodes'

interface CanvasLayout {
  edges: Edge[]
  nodes: Node[]
  setNodes: (update: (nodes: Node[]) => Node[]) => void
  takeSnapshot: () => void
}

export function useCanvasLayout({ edges, nodes, setNodes, takeSnapshot }: CanvasLayout) {
  const { fitView } = useReactFlow()

  // Two frames: the first lets React commit the new nodes, the second lets
  // React Flow measure them. Fitting on one frame uses fallback sizes and
  // leaves freshly added nodes tucked under the composer.
  const refit = useCallback(() => {
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => fitView({ ...FIT, duration: 400 })))
  }, [fitView])

  // Which way the ranks run. Dagre's own `rankdir` does the work — the handles
  // follow it (see nodes.tsx), so nothing here computes a position by hand.
  const [dir, setDirState] = useState<FlowDir>(DEFAULT_DIR)
  // Read by the tool bridge, which is registered once and would otherwise
  // close over the direction the canvas had when the page mounted.
  const dirRef = useRef(dir)
  dirRef.current = dir

  const tidy = useCallback(
    (to: FlowDir = dir) => {
      takeSnapshot()
      setNodes(ns => tidyLayout(ns, edges, to))
      refit()
    },
    [dir, edges, refit, setNodes, takeSnapshot]
  )

  // Flipping direction without re-laying out would leave every card where the
  // other orientation put it, wired through its own neighbours — so the toggle
  // IS a tidy, just one that changes rankdir on the way through.
  const setDir = useCallback(
    (to: FlowDir) => {
      setDirState(to)
      tidy(to)
    },
    [tidy]
  )

  // On-load layout only. The seed is laid out against the fallback constants in
  // layout.ts, which don't match the real cards, so the first paint is
  // approximate — this re-tidies ONCE, the moment React Flow has measured every
  // card, so what you SEE on load is already tidy.
  //
  // The trigger is the measurement itself, not useNodesInitialized: the
  // signature changes the moment real dimensions land, which is precisely the
  // instant a correct layout is possible.
  //
  // After that, dragging a card keeps its place. Adding or removing a step
  // re-tidies the ranks (see `laidOut`) without touching the camera — the
  // viewport is yours. The Tidy button is still there for a full rearrange
  // plus a fit.
  const didAutoTidy = useRef(false)
  const measuredSig = nodes.map(n => `${n.id}:${n.measured?.width ?? 0}x${n.measured?.height ?? 0}`).join()
  const allMeasured = nodes.length > 0 && nodes.every(n => n.measured?.width && n.measured?.height)

  // eslint-disable-next-line no-restricted-syntax -- `didAutoTidy` is a one-shot latch, not a mirrored atom.
  useEffect(() => {
    if (!allMeasured || didAutoTidy.current) {
      return
    }

    didAutoTidy.current = true

    if (!nodes.some(n => n.id !== CANVAS_NOTE_ID)) {
      // The note lives at the origin, which is the pane's top-left — narnia.
      // fitView with maxZoom 1 pans it into the padded centre (above the
      // composer, clear of the log) without blowing the label up to a title.
      // Two frames: commit, then measure — same reason `refit` waits.
      window.requestAnimationFrame(() =>
        window.requestAnimationFrame(() => void fitView({ ...FIT, duration: 0, maxZoom: 1 }))
      )

      return
    }

    setNodes(ns => tidyLayout(ns, edges, dir))
    refit()
    // measuredSig is the real dependency — it changes when a card's measured
    // size lands. eslint can't see that it stands in for `nodes`.
  }, [allMeasured, measuredSig, dir, edges, fitView, refit, setNodes])

  // Double-click on empty canvas. An empty canvas is just the note, and
  // framing one small card would zoom it to a title.
  const resetView = useCallback(() => {
    const empty = !nodes.some(n => n.id !== CANVAS_NOTE_ID)

    window.requestAnimationFrame(() =>
      window.requestAnimationFrame(() => void fitView({ ...FIT, duration: 400, maxZoom: empty ? 1 : undefined }))
    )
  }, [fitView, nodes])

  return { dir, dirRef, resetView, setDir, tidy, vertical: dir === 'TB' }
}
