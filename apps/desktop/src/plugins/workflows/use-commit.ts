// How an edit lands on the canvas.
//
// ONE commit path for every structural edit — the connect gesture, a delete,
// the inspector, and every tool the composer's agent calls. Undo and the
// transcript hang off `applyOp`, so nothing can mutate the document and leave
// one of them behind.
//
// The difference between a hand edit and an agent's is only pacing: yours is
// one op and lands at once, a chat turn is a batch and gets played out.

import type { Edge, Node } from '@xyflow/react'
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { lendCanvas, runOps } from './bridge'
import { type Graph, type OpResult, stepNodes } from './graph'
import type { RunControl } from './graph-dispatch'
import { type FlowDir, tidyLayout } from './layout'

// Pace of an agent build (see `paint`). The gap is per op — slow enough to
// read as one thing happening after another, fast enough that a three-step
// edit isn't a cutscene. The budget caps the whole batch, so a long build
// accelerates rather than making you sit through it.
const PAINT_GAP_MS = 130
const PAINT_BUDGET_MS = 2200

/** A step appeared or disappeared — not a title edit, not a wire. Those are
 *  the moments the ranks have to be redone, or the new card sits wherever
 *  freeSpot parked it and a hole stays where the old one was. */
function sameSteps(a: Graph, b: Graph): boolean {
  const was = stepNodes(a)
  const now = stepNodes(b)

  if (was.length !== now.length) {
    return false
  }

  const ids = new Set(was.map(n => n.id))

  return now.every(n => ids.has(n.id))
}

function laidOut(from: Graph, to: Graph, dir: FlowDir): Graph {
  return sameSteps(from, to) ? to : { edges: to.edges, nodes: tidyLayout(to.nodes, to.edges, dir) }
}

interface Commit {
  dir: FlowDir
  /** Live, because the bridge registers once and must not close over the
   *  direction the canvas had when the page mounted. */
  dirRef: RefObject<FlowDir>
  docId: string
  /** Live, for the same reason: the graph the agent edits is the one on screen
   *  right now, not the one from the render that lent the canvas. */
  graphRef: RefObject<Graph>
  runRef: RefObject<RunControl>
  setEdges: (edges: Edge[]) => void
  setNodes: (nodes: Node[]) => void
  takeSnapshot: () => void
}

export function useCommit({
  dir,
  dirRef,
  docId,
  graphRef,
  runRef,
  setEdges,
  setNodes,
  takeSnapshot
}: Commit) {
  // In-flight timers for an agent build, and the flag that lets cards glide to
  // their new ranks while one is playing.
  const brush = useRef<number[]>([])
  const [reflowing, setReflowing] = useState(false)

  const applyOp = useCallback(
    (op: OpResult) => {
      if (!op.ok) {
        return op
      }

      takeSnapshot()
      const next = laidOut(graphRef.current, op.graph, dir)
      setNodes(next.nodes)
      setEdges(next.edges)

      return { ...op, graph: next }
    },
    [dir, graphRef, setEdges, setNodes, takeSnapshot]
  )

  // A batch of agent ops arrives as a BUILD, not a jump cut. Landing six steps
  // and five wires in one frame reads as a page refresh — you can't tell what
  // Hermes did, only that the canvas is different now. Played out, you watch
  // it think: a step appears, a wire finds it, the graph makes room.
  //
  // The whole batch is still ONE undo. Stepping backwards through an agent's
  // reasoning one node at a time is not undo, it's archaeology.
  const paint = useCallback(
    (frames: Graph[]) => {
      if (!frames.length) {
        return
      }

      // A second batch mid-build cancels the first. Its last frame already
      // folded into the graph this one was computed from, so the pending
      // frames are stale — and two timers writing nodes is a flicker.
      brush.current.forEach(window.clearTimeout)
      brush.current = []
      takeSnapshot()

      // Long batches speed up rather than outstay their welcome. Nobody wants
      // a forty-op build to take six seconds.
      const gap = Math.min(PAINT_GAP_MS, PAINT_BUDGET_MS / frames.length)

      setReflowing(true)

      frames.forEach((frame, i) => {
        const last = i === frames.length - 1

        brush.current.push(
          window.setTimeout(
            () => {
              setNodes(frame.nodes)
              setEdges(frame.edges)

              if (last) {
                setReflowing(false)
              }
            },
            gap * (i + 1)
          )
        )
      })
    },
    [setEdges, setNodes, takeSnapshot]
  )

  // Unmount cleanup only: a build timer that fires into an unmounted canvas
  // writes state that no longer has anywhere to go.
  useEffect(
    () => () => {
      brush.current.forEach(window.clearTimeout)
    },
    []
  )

  // Hermes edits the canvas through here. While it's on screen the `workflow`
  // tool applies ops to the LIVE graph rather than the stored copy behind it,
  // so a chat turn paints the same way a hand edit does. Off screen the
  // bridge falls back to the document; you see it when you come back.
  useEffect(
    () =>
      lendCanvas({
        id: docId,
        apply: ops => {
          const from = graphRef.current
          const out = runOps(from, runRef.current, ops, dirRef.current)

          if (out.graph === from) {
            return out
          }

          // Add/remove re-tidies the ranks as the build plays. The camera
          // stays put — this is not a fitView.
          const frames = out.results.filter(r => r.ok).map(r => laidOut(from, r.graph, dirRef.current))

          paint(frames)

          return { graph: frames.at(-1) ?? out.graph, results: out.results }
        }
      }),
    [dirRef, docId, graphRef, paint, runRef]
  )

  return { applyOp, reflowing }
}
