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

// These root vars position chrome, not palette/font tokens. Keep every other
// inline declaration (including arbitrary skin vars and !important) observable.
function themeStyle(style: CSSStyleDeclaration): string {
  let snapshot = ''

  for (let i = 0; i < style.length; i += 1) {
    const name = style.item(i)

    if (name === '--workspace-left' || name === '--workspace-right' || name === '--glass-rail-edge') {
      continue
    }

    snapshot += `${name}:${style.getPropertyValue(name)}${style.getPropertyPriority(name) ? ' !important' : ''};`
  }

  return snapshot
}

/** Subscribe to theme repaints imperatively (ref/canvas, no re-render). */
export function onThemeRepaint(fn: () => void): () => void {
  if (!observer && typeof document !== 'undefined') {
    const root = document.documentElement
    let previousStyle = themeStyle(root.style)
    observer = new MutationObserver(records => {
      const nextStyle = themeStyle(root.style)
      // applyTheme also writes these attributes when only customCSS changes.
      // Even same-value attribute writes must still publish that repaint.
      // Authored CSS can derive colors/fonts from geometry variables too.
      const customCSS = document.getElementById('hermes-desktop-custom-css')?.textContent

      const repaint =
        Boolean(customCSS) || nextStyle !== previousStyle || records.some(record => record.attributeName !== 'style')

      previousStyle = nextStyle

      if (repaint) {
        listeners.forEach(l => l())
      }
    })
    observer.observe(root, { attributeFilter: ATTRS, attributes: true })
  }

  listeners.add(fn)

  return () => {
    listeners.delete(fn)

    if (listeners.size === 0) {
      observer?.disconnect()
      observer = null
    }
  }
}

/** A counter that ticks on every theme repaint — depend on it to re-resolve colors. */
export function useThemeEpoch(): number {
  const [epoch, setEpoch] = useState(0)

  useEffect(() => onThemeRepaint(() => setEpoch(e => e + 1)), [])

  return epoch
}
