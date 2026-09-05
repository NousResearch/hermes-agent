import type { Edge, Node } from '@xyflow/react'
import { useCallback, useEffect, useRef, useState } from 'react'

// React Flow ships no built-in history — this is the documented "bring your own
// undo/redo" pattern: snapshot the graph before a mutation, then swap between
// past/future stacks. We operate on the controlled useNodesState/useEdgesState
// setters (not useReactFlow) so there's a single source of truth, and we keep
// each node's live runtime (data.rt) on restore so undoing a drag never rewinds
// an in-flight run.

interface Snapshot {
  nodes: Node[]
  edges: Edge[]
}

interface Args {
  nodes: Node[]
  edges: Edge[]
  setNodes: (updater: (ns: Node[]) => Node[]) => void
  setEdges: (updater: (es: Edge[]) => Edge[]) => void
  maxHistory?: number
}

const clone = <T>(arr: T[]): T[] => arr.map(x => ({ ...x }) as T)

const isEditable = (el: EventTarget | null): boolean => {
  const n = el as HTMLElement | null

  if (!n) {
    return false
  }

  return n.tagName === 'INPUT' || n.tagName === 'TEXTAREA' || n.tagName === 'SELECT' || n.isContentEditable
}

export function useUndoRedo({ nodes, edges, setNodes, setEdges, maxHistory = 100 }: Args) {
  // refs so the callbacks always read the latest graph without re-binding
  const nodesRef = useRef(nodes)
  const edgesRef = useRef(edges)
  nodesRef.current = nodes
  edgesRef.current = edges

  const past = useRef<Snapshot[]>([])
  const future = useRef<Snapshot[]>([])
  const [canUndo, setCanUndo] = useState(false)
  const [canRedo, setCanRedo] = useState(false)

  const sync = () => {
    setCanUndo(past.current.length > 0)
    setCanRedo(future.current.length > 0)
  }

  const takeSnapshot = useCallback(() => {
    past.current.push({
      nodes: clone(nodesRef.current),
      edges: clone(edgesRef.current)
    })

    if (past.current.length > maxHistory) {
      past.current.shift()
    }

    future.current = []
    sync()
  }, [maxHistory])

  // restore positions/config/edges from the snapshot but keep the CURRENT
  // runtime for each node, so history is purely structural.
  const restore = useCallback(
    (snap: Snapshot) => {
      setNodes(cur => {
        const liveData = new Map(cur.map(n => [n.id, n.data]))

        return snap.nodes.map(n => {
          const live = liveData.get(n.id) as { rt?: unknown } | undefined

          return live ? { ...n, data: { ...(n.data as object), rt: live.rt } } : n
        })
      })
      setEdges(() => clone(snap.edges))
    },
    [setNodes, setEdges]
  )

  const undo = useCallback(() => {
    const prev = past.current.pop()

    if (!prev) {
      return
    }

    future.current.push({
      nodes: clone(nodesRef.current),
      edges: clone(edgesRef.current)
    })
    restore(prev)
    sync()
  }, [restore])

  const redo = useCallback(() => {
    const next = future.current.pop()

    if (!next) {
      return
    }

    past.current.push({
      nodes: clone(nodesRef.current),
      edges: clone(edgesRef.current)
    })
    restore(next)
    sync()
  }, [restore])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey

      if (!mod) {
        return
      }

      const key = e.key.toLowerCase()
      const isUndo = key === 'z' && !e.shiftKey
      const isRedo = (key === 'z' && e.shiftKey) || key === 'y'

      if (!isUndo && !isRedo) {
        return
      }

      // let the browser's native text undo win inside form fields
      if (isEditable(e.target)) {
        return
      }

      e.preventDefault()

      if (isRedo) {
        redo()
      } else {
        undo()
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  return { takeSnapshot, undo, redo, canUndo, canRedo }
}
