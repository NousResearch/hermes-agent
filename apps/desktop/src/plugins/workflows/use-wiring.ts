// The pointer gestures that change the wiring: drawing a wire, re-routing one,
// dropping one in space, and deleting a selection.
//
// Everything here commits through `applyOp`, so the keyboard, the mouse and an
// agent heal the flow identically — these handlers decide WHAT the gesture
// meant, never how it lands.

import { type Connection, type Edge, type Node, useStore } from '@xyflow/react'
import { useCallback, useEffect, useRef } from 'react'

import { connect, disconnect, type Graph, type OpResult, removeStep } from './graph'

interface Wiring {
  applyOp: (op: OpResult) => OpResult
  graph: Graph
}

export function useWiring({ applyOp, graph }: Wiring) {
  // Escape drops the wire you're dragging. React Flow has no abort — the drag
  // only ends on pointerup — so we fake the pointerup and flag the drop as one
  // to ignore. Both flags matter: `aborted` stops a wire being drawn when you
  // happen to be over a valid port, and `landed` stops a reconnect being read
  // as dropped-in-space, which would cut the wire you were trying to keep.
  const aborted = useRef(false)
  // React Flow only tells you a reconnect LANDED. Dropping on nothing fires
  // nothing at all, so the wire silently springs back and the obvious way to
  // unplug something does nothing. The flag is the documented way to tell the
  // two endings apart: onReconnect marks it handled, and whatever reaches
  // onReconnectEnd unmarked was dropped in space.
  const landed = useRef(true)
  const connecting = useStore(s => s.connection.inProgress)

  // eslint-disable-next-line no-restricted-syntax -- the ref writes are inside the key handler, not a mirror of `connecting`.
  useEffect(() => {
    if (!connecting) {
      return
    }

    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') {
        return
      }

      e.preventDefault()
      e.stopPropagation()
      aborted.current = true
      landed.current = true
      document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }))
    }

    // Capture, so the drag dies before anything else reads the same Escape.
    window.addEventListener('keydown', onKey, true)

    return () => window.removeEventListener('keydown', onKey, true)
  }, [connecting])

  const onConnect = useCallback(
    (c: Connection) => {
      if (aborted.current) {
        return
      }

      applyOp(
        connect(graph, {
          source: c.source,
          target: c.target,
          sourceHandle: c.sourceHandle ?? undefined,
          targetHandle: c.targetHandle ?? undefined
        })
      )
    },
    [applyOp, graph]
  )

  const onConnectStart = useCallback(() => {
    aborted.current = false
  }, [])

  const onConnectEnd = useCallback(() => {
    aborted.current = false
  }, [])

  // Refuse the connection during the drag rather than after the drop, so the
  // wire never snaps into place and then vanishes.
  const isValidConnection = useCallback(
    (c: Connection | Edge) =>
      !!c.source &&
      !!c.target &&
      c.source !== c.target &&
      !graph.edges.some(e => e.source === c.source && e.target === c.target),
    [graph.edges]
  )

  const onReconnectStart = useCallback(() => {
    landed.current = false
  }, [])

  // Dragging a live endpoint onto another port re-routes rather than forcing a
  // cut-then-draw; dragging it into empty canvas cuts the wire.
  const onReconnect = useCallback(
    (old: Edge, c: Connection) => {
      landed.current = true

      if (aborted.current) {
        return
      }

      const cut = disconnect(graph, old.id)

      if (!cut.ok) {
        return
      }

      applyOp(
        connect(cut.graph, {
          source: c.source,
          target: c.target,
          sourceHandle: c.sourceHandle ?? undefined,
          targetHandle: c.targetHandle ?? undefined
        })
      )
    },
    [applyOp, graph]
  )

  const onReconnectEnd = useCallback(
    (_: unknown, edge: Edge) => {
      if (!landed.current) {
        applyOp(disconnect(graph, edge.id))
      }

      landed.current = true
    },
    [applyOp, graph]
  )

  // Deleting goes through the same primitive as the tools, so the keyboard and
  // an agent heal the flow identically. Returning false stops React Flow from
  // also removing what we've already removed.
  const onBeforeDelete = useCallback(
    async ({ nodes: dropNodes, edges: dropEdges }: { nodes: Node[]; edges: Edge[] }) => {
      if (!dropNodes.length) {
        return true
      }

      let next: Graph = graph

      for (const n of dropNodes) {
        const op = removeStep(next, n.id)

        if (op.ok) {
          next = op.graph
        }
      }

      if (dropEdges.length) {
        next = {
          ...next,
          edges: next.edges.filter(e => !dropEdges.some(d => d.id === e.id))
        }
      }

      applyOp({ ok: true, graph: next, message: '' })

      return false
    },
    [applyOp, graph]
  )

  const cutEdge = useCallback((id: string) => applyOp(disconnect(graph, id)), [applyOp, graph])

  const removeNode = useCallback((id: string) => applyOp(removeStep(graph, id)), [applyOp, graph])

  return {
    cutEdge,
    isValidConnection,
    onBeforeDelete,
    onConnect,
    onConnectEnd,
    onConnectStart,
    onReconnect,
    onReconnectEnd,
    onReconnectStart,
    removeNode
  }
}
