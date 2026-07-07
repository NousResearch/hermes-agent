import { useEffect, useState } from 'react'

// Theme repaints (themes/context.tsx) toggle `.dark` + rewrite inline custom
// props/data-hermes-* on <html>. Canvas/probe consumers that rasterize the
// *computed* color-mix()/oklch tokens must re-resolve AFTER the paint — useTheme()
// can't, since a child's effect runs before the provider's applyTheme. A
// MutationObserver fires post-mutation, so the next getComputedStyle is fresh.
// One observer, fanned out to every listener.
const ATTRS = ['class', 'style', 'data-hermes-mode', 'data-hermes-theme']
const listeners = new Set<() => void>()
let observer: MutationObserver | null = null

/** Subscribe to theme repaints imperatively (ref/canvas, no re-render). */
export function onThemeRepaint(fn: () => void): () => void {
  if (!observer && typeof document !== 'undefined') {
    observer = new MutationObserver(() => listeners.forEach(l => l()))
    observer.observe(document.documentElement, { attributeFilter: ATTRS, attributes: true })
  }

  listeners.add(fn)

  return () => void listeners.delete(fn)
}

/** A counter that ticks on every theme repaint — depend on it to re-resolve colors. */
export function useThemeEpoch(): number {
  const [epoch, setEpoch] = useState(0)

  useEffect(() => onThemeRepaint(() => setEpoch(e => e + 1)), [])

  return epoch
}
