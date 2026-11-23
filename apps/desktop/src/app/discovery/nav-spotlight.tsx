import { type RefObject, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

import { type Box, connectorPath, visiblePart } from '@/lib/spotlight-geometry'

/** A place in the navigation a splash row can point at. */
export type NavSpot = 'agents' | 'chats' | 'library' | 'machines' | 'notifications' | 'routines'

/**
 * Where each place is drawn, best first; the first candidate actually on screen is lit. Chats live
 * under their agent in the chat sidebar (Figma M1 rings an agent's group), else behind the rail's
 * Chats; a machine is its heading in the sidebar, else Settings, which holds Machines. Rail buttons
 * carry `data-nav-spot` with their area id.
 */
const NAV_SPOT_CANDIDATES: Record<NavSpot, readonly string[]> = {
  chats: ['[data-slot="agent-group"]', '[data-nav-spot="chat"]'],
  agents: ['[data-nav-spot="profiles"]'],
  library: ['[data-nav-spot="artifacts"]'],
  routines: ['[data-nav-spot="cron"]'],
  machines: ['[data-slot="gateway-group"] > :first-child', '[data-nav-spot="settings"]'],
  notifications: ['[data-nav-spot="notifications"]']
}

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

/** What of the place is on screen and not covered (a sheet's scrim, a closed sheet), or null. */
function locate(spot: NavSpot): Box | null {
  for (const selector of NAV_SPOT_CANDIDATES[spot]) {
    for (const element of document.querySelectorAll(selector)) {
      const visible = visiblePart(toBox(element.getBoundingClientRect()), clipsOf(element))

      if (!visible) {
        continue
      }

      const hit = document.elementFromPoint((visible.left + visible.right) / 2, (visible.top + visible.bottom) / 2)

      if (hit && element.contains(hit)) {
        return visible
      }
    }
  }

  return null
}

interface NavSpotlightProps {
  spot: NavSpot
  /** The row pointing at it: the connector leaves from its facing side. */
  anchor: RefObject<HTMLElement | null>
}

/**
 * Rings `spot` and draws a dashed elbow to it from `anchor`, following both every frame (the sidebar
 * folds, the window narrows, a list scrolls). A place that is not on screen gets neither: the row's
 * own text still says what it is. Decoration only: hidden from assistive tech, never hit.
 */
export function NavSpotlight({ anchor, spot }: NavSpotlightProps) {
  const ringRef = useRef<HTMLDivElement>(null)
  const pathRef = useRef<SVGPathElement>(null)

  useEffect(() => {
    let frame = 0
    let painted = ''

    const paint = () => {
      frame = requestAnimationFrame(paint)
      const ring = ringRef.current
      const path = pathRef.current
      const row = anchor.current
      const layer = ring?.parentElement
      const place = row ? locate(spot) : null

      if (!ring || !path || !layer) {
        return
      }

      const target = place && inLayer(place, layer)
      const line = row && target ? connectorPath(inLayer(toBox(row.getBoundingClientRect()), layer), target) : null
      const next = JSON.stringify([target, line])

      if (next === painted) {
        return
      }

      painted = next
      ring.hidden = !target
      path.setAttribute('d', line ?? '')

      if (target) {
        ring.style.transform = `translate(${target.left}px, ${target.top}px)`
        ring.style.width = `${target.right - target.left}px`
        ring.style.height = `${target.bottom - target.top}px`
      }
    }

    frame = requestAnimationFrame(paint)

    return () => cancelAnimationFrame(frame)
  }, [anchor, spot])

  return createPortal(
    <div aria-hidden className="discovery-spotlight" data-slot="nav-spotlight">
      <svg className="discovery-spotlight-line">
        <path ref={pathRef} />
      </svg>
      <div className="discovery-spotlight-ring" hidden ref={ringRef} />
    </div>,
    document.body
  )
}
