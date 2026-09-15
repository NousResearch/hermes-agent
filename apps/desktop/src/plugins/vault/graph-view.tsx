import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, type SimulationNodeDatum } from 'd3-force'
import { useEffect, useRef, useState } from 'react'

import type { VaultGraph, VaultGraphEdge, VaultGraphNode } from './types'

interface SimNode extends VaultGraphNode, SimulationNodeDatum {
  x?: number
  y?: number
  vx?: number
  vy?: number
}

interface SimLink {
  source: SimNode | string
  target: SimNode | string
}

export function VaultGraphView({
  graph,
  onSelectNote
}: {
  graph: VaultGraph
  onSelectNote: (title: string) => void
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [nodes, setNodes] = useState<SimNode[]>([])
  const [links, setLinks] = useState<Array<{ source: { x: number; y: number }; target: { x: number; y: number } }>>([])
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [transform, setTransform] = useState({ x: 0, y: 0, k: 1 })
  const isDraggingRef = useRef(false)
  const dragStartRef = useRef({ x: 0, y: 0 })

  useEffect(() => {
    if (!containerRef.current || !graph.nodes.length) return

    const width = containerRef.current.clientWidth || 800
    const height = containerRef.current.clientHeight || 600

    const simNodes: SimNode[] = graph.nodes.map(n => ({ ...n }))
    const simLinks: SimLink[] = graph.edges.map(e => ({ source: e.source, target: e.target }))

    const sim = forceSimulation(simNodes)
      .force('charge', forceManyBody().strength(-120))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide().radius((d: any) => 8 + (d.weight || 1) * 2))
      .force(
        'link',
        forceLink(simLinks)
          .id((d: any) => d.id)
          .distance(70)
      )

    sim.on('tick', () => {
      setNodes([...simNodes])
      const renderedLinks: Array<{ source: { x: number; y: number }; target: { x: number; y: number } }> = []
      for (const link of simLinks) {
        if (typeof link.source === 'object' && typeof link.target === 'object') {
          const s = link.source as SimNode
          const t = link.target as SimNode
          if (s.x != null && s.y != null && t.x != null && t.y != null) {
            renderedLinks.push({
              source: { x: s.x, y: s.y },
              target: { x: t.x, y: t.y }
            })
          }
        }
      }
      setLinks(renderedLinks)
    })

    sim.alpha(1).restart()

    return () => {
      sim.stop()
    }
  }, [graph])

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9
    setTransform(prev => ({
      ...prev,
      k: Math.max(0.2, Math.min(4, prev.k * zoomFactor))
    }))
  }

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return
    isDraggingRef.current = true
    dragStartRef.current = { x: e.clientX - transform.x, y: e.clientY - transform.y }
  }

  const onMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current) return
    setTransform(prev => ({
      ...prev,
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y
    }))
  }

  const onMouseUp = () => {
    isDraggingRef.current = false
  }

  return (
    <div
      className="relative h-full w-full select-none overflow-hidden bg-(--ui-bg-quaternary)"
      onMouseDown={onMouseDown}
      onMouseLeave={onMouseUp}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onWheel={onWheel}
      ref={containerRef}
    >
      <div className="absolute right-3 top-3 z-10 flex gap-1 rounded bg-(--ui-bg-primary) p-1 text-xs shadow-md border border-(--ui-stroke-secondary)">
        <button
          className="px-2 py-1 hover:bg-(--ui-bg-tertiary) rounded"
          onClick={() => setTransform(p => ({ ...p, k: Math.min(4, p.k * 1.2) }))}
          type="button"
        >
          +
        </button>
        <button
          className="px-2 py-1 hover:bg-(--ui-bg-tertiary) rounded"
          onClick={() => setTransform(p => ({ ...p, k: Math.max(0.2, p.k * 0.8) }))}
          type="button"
        >
          -
        </button>
        <button
          className="px-2 py-1 hover:bg-(--ui-bg-tertiary) rounded"
          onClick={() => setTransform({ x: 0, y: 0, k: 1 })}
          type="button"
        >
          Reset
        </button>
      </div>

      <svg className="h-full w-full">
        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.k})`}>
          {/* Edges */}
          {links.map((link, idx) => (
            <line
              className="stroke-(--ui-stroke-secondary) opacity-50"
              key={idx}
              strokeWidth={1.5}
              x1={link.source.x}
              x2={link.target.x}
              y1={link.source.y}
              y2={link.target.y}
            />
          ))}

          {/* Nodes */}
          {nodes.map(node => {
            const isHovered = hoveredNode === node.id
            const radius = 5 + Math.min(node.weight * 2, 16)
            return (
              <g
                className="cursor-pointer"
                key={node.id}
                onClick={e => {
                  e.stopPropagation()
                  onSelectNote(node.id)
                }}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                transform={`translate(${node.x ?? 0}, ${node.y ?? 0})`}
              >
                <circle
                  className="transition-all duration-150"
                  fill={isHovered ? 'var(--ui-accent-primary, #6366f1)' : 'var(--ui-text-primary, #94a3b8)'}
                  r={radius}
                  stroke="var(--ui-bg-primary, #ffffff)"
                  strokeWidth={2}
                />
                <text
                  className="fill-(--ui-text-secondary) text-[10px] font-medium"
                  dx={radius + 4}
                  dy={3}
                >
                  {node.label}
                </text>
              </g>
            )
          })}
        </g>
      </svg>

      {hoveredNode && (
        <div className="pointer-events-none absolute bottom-4 left-4 z-10 rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-2 text-xs shadow-lg">
          <div className="font-semibold text-(--ui-text-primary)">{hoveredNode}</div>
          <div className="text-[10px] text-(--ui-text-tertiary)">Clique para abrir a nota</div>
        </div>
      )}
    </div>
  )
}
