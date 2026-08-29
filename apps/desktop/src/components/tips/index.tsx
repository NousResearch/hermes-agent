/**
 * TIP HOST — one per window, renders nothing until the app has something to
 * say.
 *
 * It owns what the bubble itself must not: keeping the anchor honest, marking
 * it while the arrow is on it, and knowing when a tip has had its moment.
 *
 * The anchor is re-resolved on a slow poll rather than captured once. A tip
 * outlives several React renders, and the element it points at can be replaced
 * (a re-render swaps the node) or leave entirely (a pane closes, a route
 * changes). Re-resolving handles the first and closes the tip on the second,
 * which is the difference between an arrow that follows its subject and one
 * left pointing at empty screen.
 */

import './tips.css'

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'
import { resolveTipAnchor } from '@/lib/tips/anchor'
import { $activeTip, dismissTip, retireActiveTip } from '@/store/tips'

import { TipBubble } from './tip-bubble'
import { useTipRotation } from './use-tip-rotation'

/** How long a tip stays before it steps aside for the rotation. */
const LINGER_MS = 22_000
const ANCHOR_POLL_MS = 1_000

export function TipHost() {
  const { t } = useI18n()
  const tip = useStore($activeTip)
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)

  useTipRotation(t.tips)

  useEffect(() => {
    if (!tip) {
      setAnchor(null)

      return
    }

    const sync = () => {
      const found = resolveTipAnchor(document, tip.targets)

      if (found) {
        setAnchor(previous => (previous === found ? previous : found))
      } else {
        // Whatever it was about is gone; so is the reason to point at it.
        dismissTip()
      }
    }

    sync()

    const poll = window.setInterval(sync, ANCHOR_POLL_MS)
    const linger = window.setTimeout(dismissTip, LINGER_MS)

    return () => {
      window.clearInterval(poll)
      window.clearTimeout(linger)
    }
  }, [tip])

  // Mark the subject for as long as the arrow is on it.
  //
  // Almost always the subject IS the anchor. The exception is an anchor that
  // exists only to place the arrow — a nav row's label span, which is anchored
  // so the arrow lands at the end of the word instead of out at the sidebar's
  // edge, and which would otherwise get an outline around one word. Those say
  // so with `data-tip-arrow-only` and hand the outline to their region.
  //
  // Opt-in rather than "walk up to the nearest region", because every anchor
  // has a region above it somewhere: the model pill sits inside the composer,
  // and a walk marks the composer for a tip about the pill.
  //
  // An attribute and not a class, so React — which owns these elements'
  // className — has nothing to clobber, and so a target the tip system has
  // never heard of still gets marked.
  useEffect(() => {
    if (!anchor) {
      return
    }

    const marked = anchor.hasAttribute('data-tip-arrow-only') ? (anchor.closest('[data-tip-region]') ?? anchor) : anchor

    marked.setAttribute('data-tip-target', '')

    return () => marked.removeAttribute('data-tip-target')
  }, [anchor])

  if (!tip || !anchor) {
    return null
  }

  return (
    <TipBubble
      anchor={anchor}
      keybind={tip.keybind}
      onClose={retireActiveTip}
      side={tip.side}
      text={tip.text}
      title={tip.title}
    />
  )
}
