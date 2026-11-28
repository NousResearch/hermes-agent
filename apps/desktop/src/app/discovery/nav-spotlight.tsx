import { type RefObject, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

import { type Box, connectorPath, visiblePart } from '@/lib/spotlight-geometry'

/** A place in the navigation a splash row can point at. */
export type NavSpot = 'agents' | 'chats' | 'connections' | 'library' | 'machines' | 'notifications' | 'routines'

/**
 * Where each place is drawn, best first; the first candidate actually on screen is lit. Chats live
 * under their agent in the chat sidebar (Figma M1 rings an agent's group), else behind the rail's
 * Chats; a machine is its heading in the sidebar, else Settings, which holds Machines. Rail buttons
 * carry `data-nav-spot` with their area id; Connections is the redesign's word for the rail's
 * Capabilities destination (simplified-global-nav.tsx).
 */
const NAV_SPOT_CANDIDATES: Record<NavSpot, readonly string[]> = {
  chats: ['[data-slot="agent-group"]', '[data-nav-spot="chat"]'],
  agents: ['[data-nav-spot="profiles"]'],
  connections: ['[data-nav-spot="capabilities"]'],
  library: ['[data-nav-spot="artifacts"]'],
  routines: ['[data-nav-spot="cron"]'],
  machines: ['[data-slot="gateway-group"] > :first-child', '[data-nav-spot="settings"]'],
  notifications: ['[data-nav-spot="notifications"]']
}

/** Destinations a splash points at on the rail alone: Chats never reaches the chat sidebar's agent
 *  group, so highlighting it cannot read as opening the sidebar (Figma 18924:28898). */
const RAIL_ONLY_CANDIDATES: Partial<Record<NavSpot, readonly string[]>> = { chats: ['[data-nav-spot="chat"]'] }

const toBox = ({ left, top, right, bottom }: DOMRect): Box => ({ left, top, right, bottom })

/**
 * A measured box in the spotlight layer's own pixels. They are the viewport's only at 100%: the web
 * app's UI scale is a zoom on the page, under which a measured box is in screen pixels and the
 * layer's lengths are not, so a box used as measured landed short of its place by the scale.
 */
function inLayer(box: Box, layer: HTMLElement): Box {
  const origin = layer.getBoundingClientRect()
  const scale = layer.offsetWidth ? origin.width / layer.offsetWidth : 1

  return {
    left: (box.left - origin.left) / scale,
    top: (box.top - origin.top) / scale,
    right: (box.right - origin.left) / scale,
    bottom: (box.bottom - origin.top) / scale
  }
}

/** The viewport and every ancestor that clips its overflow. */
function clipsOf(element: Element): Box[] {
  const clips: Box[] = [{ left: 0, top: 0, right: window.innerWidth, bottom: window.innerHeight }]

  for (let node = element.parentElement; node; node = node.parentElement) {
    const { overflowX, overflowY } = getComputedStyle(node)

    if (overflowX !== 'visible' || overflowY !== 'visible') {
      clips.push(toBox(node.getBoundingClientRect()))
    }
  }

  return clips
}

/** A place found on screen: the element that carries it and what of it shows. */
interface Located {
  element: HTMLElement
  box: Box
}

/** What of the place is on screen and not covered (a sheet's scrim, a closed sheet), or null. */
function locate(spot: NavSpot, railOnly: boolean): Located | null {
  const candidates = (railOnly && RAIL_ONLY_CANDIDATES[spot]) || NAV_SPOT_CANDIDATES[spot]

  for (const selector of candidates) {
    for (const element of document.querySelectorAll<HTMLElement>(selector)) {
      const visible = visiblePart(toBox(element.getBoundingClientRect()), clipsOf(element))

      if (!visible) {
        continue
      }

      const hit = document.elementFromPoint((visible.left + visible.right) / 2, (visible.top + visible.bottom) / 2)

      if (hit && element.contains(hit)) {
        return { element, box: visible }
      }
    }
  }

  return null
}

/** The navigation a splash points into: the rail and the chat sidebar (docked, or as a sheet). */
const NAV_REGIONS = '[data-slot="global-nav"], [data-zone-pane="sessions"]'

const rect = ({ left, top, right, bottom }: Box) => `M ${left} ${top} H ${right} V ${bottom} H ${left} Z`

interface NavSpotlightProps {
  /** The rows are under the pointer or hold focus: the navigation steps back around the place, as
   *  the rest of the splash does around the row. */
  dim: boolean
  spot: NavSpot
  /** The row pointing at it: the connector leaves from its facing side. */
  anchor: RefObject<HTMLElement | null>
  /** The rail alone (Figma 18924:28898): Chats rings its rail button and never the chat sidebar's
   *  agent group, so a highlighted chat section does not read as opening the sidebar. */
  railOnly?: boolean
  /** The dashed elbow from the row to the place (the rename splash). Off, the place's own rail
   *  button carries the highlight instead (Figma 18924:28709): the card is read beside the nav, not
   *  joined to it. */
  connector?: boolean
}

/**
 * Rings `spot` and draws a dashed elbow to it from `anchor`, following both every frame (the sidebar
 * folds, the window narrows, a list scrolls). With `connector={false}`, the place's own rail button
 * carries a brand glow (`data-splash-highlight`) instead, so the card reads beside the nav rather
 * than joined to it. A place that is not on screen gets neither: the row's own text still says what
 * it is. Decoration only: hidden from assistive tech, never hit.
 */
export function NavSpotlight({ anchor, connector = true, dim, railOnly = false, spot }: NavSpotlightProps) {
  const ringRef = useRef<HTMLDivElement>(null)
  const pathRef = useRef<SVGPathElement>(null)
  const veilRef = useRef<SVGPathElement>(null)

  useEffect(() => {
    let frame = 0
    let painted = ''
    // The rail button lit in place of a connector (the first-time splash): cleared on change and
    // unmount so the rail returns to its own state.
    let highlighted: HTMLElement | null = null

    const paint = () => {
      frame = requestAnimationFrame(paint)
      const ring = ringRef.current
      const path = pathRef.current
      const row = anchor.current
      const layer = ring?.parentElement
      const veil = veilRef.current
      const located = row ? locate(spot, railOnly) : null

      if (!ring || !path || !veil || !layer) {
        return
      }

      const target = located && inLayer(located.box, layer)
      const line =
        connector && row && target
          ? connectorPath(inLayer(toBox(row.getBoundingClientRect()), layer), target)
          : null

      const lit = connector ? null : (located?.element ?? null)

      if (lit !== highlighted) {
        highlighted?.removeAttribute('data-splash-highlight')
        lit?.setAttribute('data-splash-highlight', 'true')
        highlighted = lit
      }

      // The veil covers each part of the navigation; the place is cut out of it (even-odd).
      const regions = target
        ? [...document.querySelectorAll(NAV_REGIONS)]
            .map(region => toBox(region.getBoundingClientRect()))
            .filter(box => box.right > box.left && box.bottom > box.top)
            .map(box => rect(inLayer(box, layer)))
        : []

      const cover = regions.length && target ? [...regions, rect(target)].join(' ') : ''
      const next = JSON.stringify([target, line, cover])

      if (next === painted) {
        return
      }

      painted = next
      ring.hidden = !target
      path.setAttribute('d', line ?? '')
      veil.setAttribute('d', cover)

      if (target) {
        ring.style.transform = `translate(${target.left}px, ${target.top}px)`
        ring.style.width = `${target.right - target.left}px`
        ring.style.height = `${target.bottom - target.top}px`
      }
    }

    frame = requestAnimationFrame(paint)

    return () => {
      cancelAnimationFrame(frame)
      highlighted?.removeAttribute('data-splash-highlight')
    }
  }, [anchor, connector, railOnly, spot])

  return createPortal(
    <div
      aria-hidden
      className="discovery-spotlight"
      data-dim={dim || undefined}
      data-rail-highlight={!connector || undefined}
      data-slot="nav-spotlight"
    >
      <svg className="discovery-spotlight-line">
        <path className="discovery-spotlight-veil" fillRule="evenodd" ref={veilRef} />
        <path ref={pathRef} />
      </svg>
      <div className="discovery-spotlight-ring" hidden ref={ringRef} />
    </div>,
    document.body
  )
}
