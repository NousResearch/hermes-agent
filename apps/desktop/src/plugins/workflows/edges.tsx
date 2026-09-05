import { Codicon } from '@hermes/plugin-sdk'
import { BaseEdge, EdgeLabelRenderer, type EdgeProps, getBezierPath, Position, useNodesData } from '@xyflow/react'
import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'

import { useAddStep } from './add-step'
import { useFlowDir } from './direction'
import { RANK_GAP } from './layout'
import type { EdgeState } from './protocol'

// React Flow renders edges itself, so there are no props to thread a callback
// through — the + already comes down by context and cutting needs the same
// door. Routed through graph.disconnect on the other side, so the button and
// the agent's graph_disconnect are the same operation with the same undo.
const CutCtx = createContext<(edgeId: string) => void>(() => {})
export const CutEdgeProvider = CutCtx.Provider

/** n8n's hide delay, to the millisecond (CanvasEdge.vue `delayedHoveredTimeout`).
 *  Show is instant, hide waits — and that asymmetry is the whole trick. The
 *  toolbar doesn't sit ON the wire, so leaving the wire to reach it would put
 *  the pointer in dead space and take the toolbar with it. The wire hands off
 *  to the buttons, which hold themselves open once you arrive. */
const TOOLBAR_REST_MS = 600

// What a wire is worth looking at: where it came from and where it's going.
//
// A connector strokes with a gradient between the status colour of the step at
// each end, so the run's frontier is legible on the wires themselves — the
// link out of a finished step into a working one ramps from settled into
// accent, and you can watch work move across the graph without reading a
// single card. It's the one idea worth taking from Blender, where a link is
// always tinted by the two sockets it joins.
//
// The stops are deliberately not the status palette at full strength. `done`
// resolves to --scenario-edge-done — the ONE green (--ui-green, the composer's
// diff pair) carried 55% into the rest hairline — because once a run finishes
// every edge is done, and full-strength green would leave the settled graph
// glowing, which is the mistake the colour-per-node builders all make. Only
// the states that are actually happening get a full hue: live, reworking, or
// broken. At rest both stops resolve to --scenario-edge and the gradient
// collapses to the hairline the canvas already had.
const STATUS_HUE: Record<string, string> = {
  running: 'var(--scenario-running)',
  looping: 'var(--scenario-loop)',
  done: 'var(--scenario-edge-done)',
  failed: 'var(--scenario-fail)'
}

const hueFor = (status?: string) => (status && STATUS_HUE[status]) || 'var(--scenario-edge)'

/** One droplet crossing. Was 1.1s — 4.4s reads as travel you can follow (and
 *  gives the + refraction a real approach/contact/release arc) instead of a
 *  blip. The stretch keyTimes are fractions, so the deformation slows with it
 *  for free. */
const packetDur = '4.4s'

export function DataEdge(props: EdgeProps) {
  const { id, source, target, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data } = props

  const [from, to] = useNodesData([source, target])
  const fromHue = hueFor((from?.data as { rt?: { status?: string } })?.rt?.status)
  const toHue = hueFor((to?.data as { rt?: { status?: string } })?.rt?.status)

  const state = (data?.state as EdgeState) ?? 'idle'
  const isLoop = Boolean(data?.loop)
  const active = state === 'active' || state === 'loop'

  const addStep = useAddStep()
  const cutEdge = useContext(CutCtx)

  const [hot, setHot] = useState(false)
  const rest = useRef<number | undefined>(undefined)

  const wake = useCallback(() => {
    clearTimeout(rest.current)
    setHot(true)
  }, [])

  const cool = useCallback(() => {
    clearTimeout(rest.current)
    rest.current = window.setTimeout(() => setHot(false), TOOLBAR_REST_MS)
  }, [])

  useEffect(() => () => clearTimeout(rest.current), [])

  // Forward wires: a shallow bezier. Elbows were the single most generic thing
  // on the canvas — an orthogonal step path is what every flowchart tool draws
  // by default, and it makes a fan-out read as a routing diagram rather than
  // dataflow. A bezier keeps the same rank-to-rank geometry while letting the
  // branch separate by curve instead of by corner.
  const [bezier, bezierX, bezierY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition: sourcePosition ?? Position.Right,
    targetPosition: targetPosition ?? Position.Left,
    curvature: 0.32
  })

  // The loop-back is drawn by hand, because getBezierPath cannot draw it.
  //
  // Its endpoints sit at nearly the same y — a gate and the step it returns to
  // are on the same rank — and React Flow derives a control point from the
  // y-delta, so with the delta at ~0 they collapse onto the endpoints and the
  // "deep curvature" resolves to a STRAIGHT LINE through the middle of the
  // graph, behind whatever cards are in the way. The curvature prop never had
  // anything to bite on.
  //
  // So the belly is stated outright: both control points swing clear by a depth
  // scaled to the span, floored so a short hop still clears the rank it passes.
  // The source control point also leans OUT first, because the arm's port faces
  // the way every other output does — the wire has to be seen leaving its own
  // dot, not cutting back under the card it came from.
  //
  // Vertical is the same curve transposed: the flow runs down instead of right,
  // so the span is measured on y, the belly swings left (which is the face the
  // rework port moves to), and the lean goes down.
  const vertical = useFlowDir() === 'TB'
  const span = Math.abs(vertical ? targetY - sourceY : targetX - sourceX)
  const belly = Math.max(72, span * 0.3)
  const out = Math.max(32, span * 0.08)

  const loopPath = vertical
    ? `M ${sourceX},${sourceY} C ${sourceX - belly},${sourceY + out} ${targetX - belly},${targetY} ${targetX},${targetY}`
    : `M ${sourceX},${sourceY} C ${sourceX + out},${sourceY + belly} ${targetX},${targetY + belly} ${targetX},${targetY}`

  const path = isLoop ? loopPath : bezier

  // Where the toolbar sits. getBezierPath hands back a label point, but only
  // for the path IT drew — on a loop we drew our own, so the midpoint is the
  // cubic evaluated at t=0.5: (P0 + 3·P1 + 3·P2 + P3) / 8. Off by that and the
  // buttons float in open canvas, nowhere near the wire they belong to.
  const loopMidX = vertical ? (4 * sourceX + 4 * targetX - 6 * belly) / 8 : (4 * sourceX + 3 * out + 4 * targetX) / 8
  const loopMidY = vertical ? (4 * sourceY + 3 * out + 4 * targetY) / 8 : (4 * sourceY + 4 * targetY + 6 * belly) / 8
  const midX = isLoop ? loopMidX : bezierX
  const midY = isLoop ? loopMidY : bezierY
  const canSplice = !isLoop && Math.hypot(targetX - sourceX, targetY - sourceY) >= RANK_GAP * 0.75

  // n8n darkens the wire under the pointer. Ours can't take a colour — forward
  // wires stroke with a per-edge gradient set inline, which beats any rule —
  // so it answers in weight instead. Either way the point is the same: with
  // four wires leaving one port, you need to know which one the trash belongs
  // to before you click it.
  const cls =
    (isLoop ? `edge edge-loop-line${active ? ' edge-loop-active' : ''}` : `edge edge-${state}`) +
    (hot ? ' edge-hot' : '')

  // The loop-back keeps its flat amber: it's the one link that means "against
  // the flow", and a gradient would trade that identity for a second reading
  // of state the forward wires already carry.
  //
  // Edge ids are authored as "implement->review", and `>` is not valid in a
  // URL fragment — browsers happen to resolve it, but the id is also what a
  // querySelector would have to escape. Sanitised to keep the reference legal.
  const safeId = id.replace(/[^\w-]/g, '_')
  const gradId = `grad-${safeId}`
  const beadId = `bead-${safeId}`
  const stroke = isLoop ? undefined : `url(#${gradId})`

  return (
    <>
      {!isLoop && (
        <defs>
          {/* userSpaceOnUse so the ramp runs along the actual span between the
              two handles, not the path's bounding box — a deep curve would
              otherwise put the colour change in the wrong place. */}
          <linearGradient
            gradientUnits="userSpaceOnUse"
            id={gradId}
            x1={sourceX}
            x2={targetX}
            y1={sourceY}
            y2={targetY}
          >
            <stop offset="0%" stopColor={fromHue} />
            <stop offset="100%" stopColor={toHue} />
          </linearGradient>
        </defs>
      )}
      {/* interactionWidth 0 because the hit stroke below replaces it: same job,
          same 20-odd pixels of slack over a 1.25px wire, but this one also
          reports hover so the + can wake up. Two would just fight. */}
      <BaseEdge className={cls} id={id} interactionWidth={0} path={path} style={{ stroke }} />
      {/* 40px of slack, n8n's `interaction-width` (double Vue Flow's and React
          Flow's default) — over a 1.25px wire, that width is the difference
          between aiming and hitting. */}
      <path
        className="edge-hit"
        d={path}
        fill="none"
        onMouseEnter={wake}
        onMouseLeave={cool}
        stroke="transparent"
        strokeWidth={40}
      />
      {active && (
        <defs>
          {/* What makes it read as liquid rather than as a bead.

              The ramp runs the opposite way to a lit solid. A sphere is
              brightest where the light hits and falls off to a dark edge —
              which is a pearl, or an egg. Water is the inverse: the middle is
              nearly clear because you're looking straight through it, and the
              perimeter is where it goes bright, because at a glancing angle
              the surface turns reflective and light caught inside can't
              escape. Density at the rim, transparency in the body.

              `fx`/`fy` pull the clear part off-centre so the two halves aren't
              the same. Near the light the wall is thin; opposite it the ramp
              piles up into the fat bright crescent a droplet focuses onto its
              own far side. One focal offset does all of that, with no second
              shape to keep in sync with the motion — at this size a separate
              specular would land under a pixel anyway, which is the same
              reason the droplet is an ellipse and not a hand-drawn teardrop.

              Object-bounding-box units, so the ramp stretches with the rx/ry
              deformation below instead of sliding across a shape that's
              changing under it. `rotate="auto"` then keeps the thin edge
              facing the direction of travel for the whole trip.

              Deliberately NOT tinted per wire, for the same reason the fill
              never was: the packet is one object wherever it rides, and the
              wire under it already says which kind of hop this is. */}
          <radialGradient cx="50%" cy="50%" fx="34%" fy="28%" id={beadId} r="50%">
            {/* Almost nothing through the body — the wire underneath should
                read straight through it. Only the last quarter has any weight,
                and even the rim stops short of opaque, so the droplet is a
                curve of light rather than an object with a fill. */}
            <stop offset="0%" stopColor="var(--packet-glass)" stopOpacity="0.025" />
            <stop offset="52%" stopColor="var(--packet-glass)" stopOpacity="0.055" />
            <stop offset="78%" stopColor="var(--packet-glass)" stopOpacity="0.17" />
            <stop offset="93%" stopColor="var(--packet-glass)" stopOpacity="0.42" />
            {/* Feathered rather than cut, so a 3px shape doesn't alias into a
                sawtooth ring as it turns. */}
            <stop offset="100%" stopColor="var(--packet-glass)" stopOpacity="0.14" />
          </radialGradient>
        </defs>
      )}
      {active && (
        // A droplet, not a dot. Two things make it read as liquid:
        //
        //  1. `rotate="auto"` turns the shape to the path tangent, so its long
        //     axis is always the direction of travel — through the loop-back's
        //     deep belly it banks with the curve instead of sliding sideways.
        //  2. rx/ry animate against each other over the trip: compact leaving
        //     the source, drawn out across the span, gathering back up as it
        //     lands. Area is held roughly constant (rx*ry ≈ r²) so it reads as
        //     one body of water deforming, not as a thing resizing.
        //
        // The motion itself is deliberately unspecified beyond dur/path.
        // Easing it with keyPoints/keySplines (the textbook way) stopped the
        // droplet short of the target — the packet has to arrive, so the
        // stretch cycle carries the sense of pace instead.
        //
        // Ellipse rather than a teardrop path: at r=3 on a canvas sitting
        // around 0.6 zoom, the asymmetric taper of a real droplet is smaller
        // than a pixel — the elongation is the whole signal, and an ellipse
        // gets it without hand-authoring a shape per edge.
        <ellipse className={`packet ${isLoop ? 'packet-loop' : ''}`} fill={`url(#${beadId})`} rx={3} ry={3}>
          <animate
            attributeName="rx"
            calcMode="spline"
            dur={packetDur}
            keySplines=".4 0 .6 1;.4 0 .6 1;.4 0 .6 1;.4 0 .6 1"
            keyTimes="0;0.12;0.5;0.88;1"
            repeatCount="indefinite"
            values="3;3.1;5.6;3.1;3"
          />
          <animate
            attributeName="ry"
            calcMode="spline"
            dur={packetDur}
            keySplines=".4 0 .6 1;.4 0 .6 1;.4 0 .6 1;.4 0 .6 1"
            keyTimes="0;0.12;0.5;0.88;1"
            repeatCount="indefinite"
            values="3;2.9;1.7;2.9;3"
          />
          <animateMotion dur={packetDur} path={path} repeatCount="indefinite" rotate="auto" />
        </ellipse>
      )}

      {/* The wire's own toolbar — n8n's one and only way to cut a connection,
          and the reason theirs feels solved: the control is on the thing it
          acts on, at the place you were already looking.

          Splicing is offered on forward hops only. The loop-back is held out
          of Dagre (it's what makes the graph a DAG to lay out), so a step
          dropped into it would have no rank to be placed in and would land on
          the origin — and a hop shorter than a rank gap has no middle, so the
          + would sit on a port. Cutting is offered on every wire, including
          those two: a wire you can't delete is a worse problem than a wire you
          can't split. */}
      <EdgeLabelRenderer>
        {/* Placement and hover-grow are both transforms, and CSS applies the
            standalone `scale` property BEFORE `transform` — a scale here would
            multiply the translate that positions the group and fling it down
            the wire. The group places; the buttons draw. */}
        <div
          className={`edge-tools${hot ? ' hot' : ''}`}
          onMouseEnter={wake}
          onMouseLeave={cool}
          style={{ transform: `translate(-50%, -50%) translate(${midX}px, ${midY}px)` }}
        >
          {/* At rest the wire wears a bead, not a button. A + on every hop was
              a row of buttons across the graph all promising something, when
              only the one you're pointing at is offered anything. The bead
              says "there's something here" and costs a wire's worth of ink;
              the actions bloom out of it. */}
          <span aria-hidden="true" className="edge-nub" />
          <div className="edge-acts">
            {canSplice && (
              <button
                aria-label="Add a step here"
                className="edge-add"
                onClick={e => {
                  e.stopPropagation()
                  addStep({ on: 'edge', edgeId: id, at: { x: midX, y: midY } })
                }}
                onPointerDown={e => e.stopPropagation()}
                tabIndex={hot ? 0 : -1}
                title="Add a step here"
                type="button"
              >
                {/* Inline SVG, not the codicon font. The glyph box centres
                    perfectly (measured 1.74px on all four sides) but the +
                    INSIDE the font's em box sits optically low-left at this
                    size — icon fonts are drawn on a 16px grid and their
                    bearings don't survive a 9.86px button on fractional
                    pixels. A vector crosshair is geometry: two strokes through
                    the exact middle of the viewBox, centred by construction at
                    every zoom. */}
                <svg aria-hidden="true" className="edge-add-glyph" viewBox="0 0 10 10">
                  <path d="M5 1.5 V8.5 M1.5 5 H8.5" />
                </svg>
              </button>
            )}
            <button
              aria-label="Delete this wire"
              className="edge-cut"
              onClick={e => {
                e.stopPropagation()
                cutEdge(id)
              }}
              onPointerDown={e => e.stopPropagation()}
              tabIndex={hot ? 0 : -1}
              title="Delete this wire"
              type="button"
            >
              <Codicon name="trash" size={9} />
            </button>
          </div>
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

export const edgeTypes = { data: DataEdge }
