/**
 * The canvas's answer to a tour's reveal request (`revealFor` in
 * lib/tour/engine.ts).
 *
 * A tour step points at a card, and driver.js brings a target into view by
 * scrolling the scrollport above it. This canvas has none — cards are placed by
 * a transform, so a node parked off screen simply stays there and the step
 * spotlights a rectangle nobody can see. The camera is ours to move, so the
 * engine asks and we answer.
 *
 * It's a PAN, never a fit: framing the step would re-zoom on every card, and a
 * tour of a graph is about the graph. The move is a viewport translate at the
 * zoom the user chose, and it only happens for a card that isn't fully in view
 * — a camera that re-centres on every step is the annoying kind.
 */

import { useReactFlow } from '@xyflow/react'
import { type RefObject, useEffect } from 'react'

/** How long the camera takes to arrive. Under driver.js's own 520ms cutout
 *  ease, so the spotlight is still settling when the pan lands and the
 *  re-measure reads as the end of one movement rather than a correction. */
const PAN_MS = 420

const HANDLE = 'step:'

type Pad = Record<'bottom' | 'left' | 'right' | 'top', string>

/** Where a card can actually be read: the pane, less the floating chrome the
 *  page always draws over it (brand, timeline, composer, the log's lane). A
 *  card under the composer is as invisible as one past the edge, so the same
 *  reserve `fitView` frames into is what "in view" means here. */
const readable = (pane: DOMRect, pad: Pad) => ({
  bottom: pane.bottom - parseFloat(pad.bottom),
  left: pane.left + parseFloat(pad.left),
  right: pane.right - parseFloat(pad.right),
  top: pane.top + parseFloat(pad.top)
})

const contains = (area: ReturnType<typeof readable>, card: DOMRect) =>
  card.top >= area.top && card.bottom <= area.bottom && card.left >= area.left && card.right <= area.right

/** Listen on `wrap` (any ancestor of the cards) for the engine's reveal, and
 *  pan the named node into view. */
export function useTourPan(wrap: RefObject<HTMLElement | null>, fitOptions: { padding: Pad }) {
  const { getViewport, setViewport } = useReactFlow()

  useEffect(() => {
    const pane = wrap.current

    if (!pane) {
      return
    }

    const onReveal = (event: Event) => {
      const from = event.target

      if (!(from instanceof HTMLElement)) {
        return
      }

      const card = from.closest<HTMLElement>(`[data-tour^="${HANDLE}"]`)

      if (!card) {
        return
      }

      const box = card.getBoundingClientRect()
      const area = readable(pane.getBoundingClientRect(), fitOptions.padding)

      if (contains(area, box)) {
        return
      }

      // Screen pixels both, so the shortfall IS the translate — no flow
      // coordinates, no node lookup, and nothing that depends on nodeOrigin.
      // Centring rather than nudging to the edge: a step whose subject sits
      // against the composer is technically visible and still reads as
      // half-cut.
      const viewport = getViewport()

      ;(event as CustomEvent<{ settled?: PromiseLike<unknown> }>).detail.settled = setViewport(
        {
          ...viewport,
          x: viewport.x + (area.left + area.right - box.left - box.right) / 2,
          y: viewport.y + (area.top + area.bottom - box.top - box.bottom) / 2
        },
        { duration: PAN_MS }
      )
    }

    pane.addEventListener('hermes:tour-reveal', onReveal)

    return () => pane.removeEventListener('hermes:tour-reveal', onReveal)
  }, [fitOptions, getViewport, setViewport, wrap])
}
