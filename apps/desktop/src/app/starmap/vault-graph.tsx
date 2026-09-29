import { useEffect, useRef } from 'react'

import type { VaultGraph, VaultNoteNode } from '@/api/vault'

interface Body {
  fx?: number
  fy?: number
  hue: number
  node: VaultNoteNode
  r: number
  vx: number
  vy: number
  x: number
  y: number
}

interface Highlight {
  matches: null | ReadonlySet<string>
  selected: null | string
}

interface View {
  k: number
  x: number
  y: number
}

/** One hue per top-level vault folder, spread around the wheel. */
export function folderHues(nodes: readonly VaultNoteNode[]): Map<string, number> {
  const roots = [...new Set(nodes.map(n => n.folder.split('/')[0] ?? ''))].sort()

  return new Map(roots.map((root, i) => [root, roots.length <= 1 ? 200 : (200 + (i * 360) / roots.length) % 360]))
}

const radiusFor = (links: number) => 4 + Math.min(10, Math.sqrt(links) * 2.6)

/**
 * Obsidian-style note graph on one canvas: a small force simulation (charge,
 * springs, centring) that settles and then sleeps, so an idle vault costs no
 * frames. Wheel zooms, background drag pans, node drag pins, click selects.
 * The selection lights its neighbourhood and dims the rest.
 */
export function VaultGraphCanvas({
  graph,
  matches,
  onSelect,
  selected
}: {
  graph: VaultGraph
  /** Ids that satisfy the current search; null means "no filter". */
  matches: null | ReadonlySet<string>
  onSelect: (id: null | string) => void
  selected: null | string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // The simulation's imperative handle: the effect below builds it, and a
  // second effect pushes the current highlight into it as arguments.
  const sim = useRef<null | { highlight: (next: Highlight) => void }>(null)

  // eslint-disable-next-line no-restricted-syntax -- `sim` is the simulation's instance handle, not a mirrored prop
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')

    if (!canvas || !ctx) {
      return undefined
    }

    const hues = folderHues(graph.nodes)
    const index = new Map<string, Body>()
    const spread = 60 + Math.sqrt(graph.nodes.length) * 26

    const bodies = graph.nodes.map((node, i) => {
      const angle = i * 2.399963
      const dist = spread * Math.sqrt((i + 0.5) / Math.max(1, graph.nodes.length))

      const body: Body = {
        hue: hues.get(node.folder.split('/')[0] ?? '') ?? 200,
        node,
        r: radiusFor(node.links),
        vx: 0,
        vy: 0,
        x: Math.cos(angle) * dist,
        y: Math.sin(angle) * dist
      }

      index.set(node.id, body)

      return body
    })

    const links = graph.edges.flatMap(e => {
      const a = index.get(e.source)
      const b = index.get(e.target)

      return a && b ? [[a, b] as const] : []
    })

    const neighbours = new Map<string, Set<string>>()

    for (const e of graph.edges) {
      neighbours.set(e.source, (neighbours.get(e.source) ?? new Set()).add(e.target))
      neighbours.set(e.target, (neighbours.get(e.target) ?? new Set()).add(e.source))
    }

    const css = getComputedStyle(canvas)
    const ink = css.getPropertyValue('--ui-text-primary').trim() || 'currentColor'
    const view: View = { k: 1, x: 0, y: 0 }
    let width = 1
    let height = 1
    let alpha = 1
    let frame = 0
    let hover: Body | null = null
    let live: Highlight = { matches: null, selected: null }
    let userView = false
    let drag: { body?: Body; moved: boolean; px: number; py: number } | null = null

    const toWorld = (px: number, py: number) => ({ x: (px - width / 2 - view.x) / view.k, y: (py - height / 2 - view.y) / view.k })

    const pick = (px: number, py: number) => {
      const p = toWorld(px, py)
      let best: Body | null = null
      let bestD = Infinity

      for (const b of bodies) {
        const d = Math.hypot(b.x - p.x, b.y - p.y)

        if (d <= b.r + 4 / view.k && d < bestD) {
          best = b
          bestD = d
        }
      }

      return best
    }

    const step = () => {
      const n = bodies.length

      for (let i = 0; i < n; i += 1) {
        const a = bodies[i]!

        for (let j = i + 1; j < n; j += 1) {
          const b = bodies[j]!
          let dx = b.x - a.x
          let dy = b.y - a.y
          const d2 = dx * dx + dy * dy + 0.01

          if (d2 > 160000) {
            continue
          }

          const d = Math.sqrt(d2)
          const force = (3200 * alpha) / d2
          dx = (dx / d) * force
          dy = (dy / d) * force
          a.vx -= dx
          a.vy -= dy
          b.vx += dx
          b.vy += dy
        }
      }

      for (const [a, b] of links) {
        const dx = b.x - a.x
        const dy = b.y - a.y
        const d = Math.hypot(dx, dy) || 1
        const pull = ((d - 90) / d) * 0.05 * alpha
        a.vx += dx * pull
        a.vy += dy * pull
        b.vx -= dx * pull
        b.vy -= dy * pull
      }

      for (const b of bodies) {
        b.vx -= b.x * 0.006 * alpha
        b.vy -= b.y * 0.006 * alpha

        if (b.fx !== undefined && b.fy !== undefined) {
          b.x = b.fx
          b.y = b.fy
          b.vx = 0
          b.vy = 0
        } else {
          b.vx *= 0.8
          b.vy *= 0.8
          b.x += b.vx
          b.y += b.vy
        }
      }

      alpha *= 0.988
    }

    const paint = () => {
      const { matches: match, selected: sel } = live
      const focus = sel ?? hover?.node.id ?? null
      const near = focus ? neighbours.get(focus) : undefined

      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
      ctx.clearRect(0, 0, width, height)
      ctx.translate(width / 2 + view.x, height / 2 + view.y)
      ctx.scale(view.k, view.k)

      ctx.lineCap = 'round'

      for (const [a, b] of links) {
        const lit = focus !== null && (a.node.id === focus || b.node.id === focus)
        ctx.strokeStyle = lit ? `hsla(${a.hue}, 85%, 62%, 0.85)` : `hsla(${a.hue}, 45%, 50%, ${focus ? 0.12 : 0.42})`
        ctx.lineWidth = (lit ? 1.6 : 0.8) / view.k
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
      }

      for (const b of bodies) {
        const id = b.node.id
        const isFocus = id === focus
        const related = isFocus || near?.has(id) === true
        const dim = (focus !== null && !related) || (match !== null && !match.has(id))

        if (isFocus || (match?.has(id) ?? false)) {
          const glow = ctx.createRadialGradient(b.x, b.y, b.r * 0.6, b.x, b.y, b.r * 3.4)
          glow.addColorStop(0, `hsla(${b.hue}, 90%, 62%, 0.55)`)
          glow.addColorStop(1, `hsla(${b.hue}, 90%, 62%, 0)`)
          ctx.fillStyle = glow
          ctx.beginPath()
          ctx.arc(b.x, b.y, b.r * 3.4, 0, Math.PI * 2)
          ctx.fill()
        }

        ctx.globalAlpha = dim ? 0.22 : 1
        ctx.fillStyle = `hsl(${b.hue}, ${isFocus ? 90 : 72}%, ${isFocus ? 66 : 58}%)`
        ctx.beginPath()
        ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2)
        ctx.fill()
        ctx.globalAlpha = 1
      }

      ctx.font = `${12 / Math.max(view.k, 0.75)}px system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.fillStyle = ink

      for (const b of bodies) {
        const id = b.node.id
        const show = id === focus || near?.has(id) === true || bodies.length <= 40 || (view.k > 0.9 && b.node.links >= 3) || view.k > 1.8

        if (show && (match === null || match.has(id) || id === focus)) {
          ctx.fillText(b.node.label.slice(0, 28), b.x, b.y + b.r + 13 / view.k)
        }
      }
    }

    // Until the user pans or zooms, the view eases to keep the whole graph in
    // frame while it settles (and after every resize).
    const fit = () => {
      if (userView || bodies.length === 0) {
        return
      }

      let minX = Infinity
      let maxX = -Infinity
      let minY = Infinity
      let maxY = -Infinity

      for (const b of bodies) {
        minX = Math.min(minX, b.x - b.r)
        maxX = Math.max(maxX, b.x + b.r)
        minY = Math.min(minY, b.y - b.r)
        maxY = Math.max(maxY, b.y + b.r)
      }

      const k = Math.min(1.6, Math.max(0.25, Math.min((width * 0.8) / (maxX - minX || 1), (height * 0.74) / (maxY - minY || 1))))
      view.k += (k - view.k) * 0.12
      view.x += (-((minX + maxX) / 2) * view.k - view.x) * 0.12
      view.y += (-((minY + maxY) / 2) * view.k - view.y) * 0.12
    }

    const tick = () => {
      frame = 0
      fit()

      if (alpha > 0.02 || drag?.body) {
        step()
      }

      paint()

      if (alpha > 0.02 || drag || (!userView && alpha > 0.005)) {
        frame = requestAnimationFrame(tick)
      }
    }

    const kick = (boost = 0) => {
      alpha = Math.max(alpha, boost)

      if (!frame) {
        frame = requestAnimationFrame(tick)
      }
    }

    sim.current = {
      highlight: next => {
        live = next
        kick(0)
      }
    }

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      width = Math.max(1, rect.width)
      height = Math.max(1, rect.height)
      canvas.width = Math.round(width * devicePixelRatio)
      canvas.height = Math.round(height * devicePixelRatio)
      kick(0)
    }

    const local = (event: PointerEvent | WheelEvent) => {
      const rect = canvas.getBoundingClientRect()

      return { px: event.clientX - rect.left, py: event.clientY - rect.top }
    }

    const onDown = (event: PointerEvent) => {
      const { px, py } = local(event)
      const body = pick(px, py) ?? undefined
      drag = { body, moved: false, px, py }
      canvas.setPointerCapture(event.pointerId)
      kick(0)
    }

    const onMove = (event: PointerEvent) => {
      const { px, py } = local(event)

      if (!drag) {
        const next = pick(px, py)

        if (next !== hover) {
          hover = next
          canvas.style.cursor = next ? 'pointer' : 'grab'
          kick(0)
        }

        return
      }

      const dx = px - drag.px
      const dy = py - drag.py

      if (Math.abs(dx) + Math.abs(dy) > 3) {
        drag.moved = true
      }

      if (drag.body && drag.moved) {
        const p = toWorld(px, py)
        drag.body.fx = p.x
        drag.body.fy = p.y
        alpha = Math.max(alpha, 0.3)
      } else if (!drag.body) {
        userView = true
        view.x += dx
        view.y += dy
      }

      drag.px = px
      drag.py = py
    }

    const onUp = () => {
      const done = drag
      drag = null

      if (done?.body) {
        done.body.fx = undefined
        done.body.fy = undefined
      }

      if (done && !done.moved) {
        onSelect(done.body?.node.id ?? null)
      }

      kick(0)
    }

    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      userView = true
      const { px, py } = local(event)
      const before = toWorld(px, py)
      view.k = Math.min(4, Math.max(0.25, view.k * Math.exp(-event.deltaY * 0.0012)))
      view.x = px - width / 2 - before.x * view.k
      view.y = py - height / 2 - before.y * view.k
      kick(0)
    }

    const observer = new ResizeObserver(resize)
    observer.observe(canvas)
    canvas.addEventListener('pointerdown', onDown)
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointercancel', onUp)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.style.cursor = 'grab'
    resize()
    kick(1)

    return () => {
      cancelAnimationFrame(frame)
      sim.current = null
      observer.disconnect()
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointercancel', onUp)
      canvas.removeEventListener('wheel', onWheel)
    }
  }, [graph, onSelect])

  useEffect(() => {
    sim.current?.highlight({ matches, selected })
  }, [graph, matches, selected])

  return <canvas aria-label="Vault graph" className="absolute inset-0 size-full touch-none" ref={canvasRef} role="img" />
}
