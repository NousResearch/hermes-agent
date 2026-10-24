// jsdom implements no layout and no animation, so component libraries that
// call those APIs unconditionally throw on mount. These install inert
// stand-ins: enough for the component to render, never enough to assert on. A
// test that needs one of them to actually report should install its own.

import { vi } from 'vitest'

export class InertResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

/** A ResizeObserver that accepts observers and never calls them back. */
export function stubResizeObserver() {
  vi.stubGlobal('ResizeObserver', InertResizeObserver)
}

/** The pointer-capture and scroll calls Radix and cmdk make while opening a
 *  popover, menu, or combobox — and again on the item they focus. */
export function stubMenuDomApis() {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
}

/**
 * jsdom has no CSS Custom Highlight API, which is the ONLY painting path for
 * find-in-page matches (lib/find-in-page-scope.ts). These stand-ins give the
 * registry real Map/Set semantics so a test can assert exactly what the walker
 * registered — the ranges are jsdom `Range`s over the test DOM, so
 * `range.toString()` is the highlighted text. A test that needs the real paint
 * has to run in a browser, not jsdom.
 */
export class HighlightStandIn extends Set<AbstractRange> {}

export function installHighlightRegistry(): void {
  const target = globalThis as unknown as { CSS?: { highlights?: unknown }; Highlight?: unknown }

  target.Highlight ??= HighlightStandIn

  const css = (target.CSS ??= {})

  if (!css.highlights) {
    Object.defineProperty(css, 'highlights', { configurable: true, value: new Map(), writable: true })
  }
}

/** Every painted match's text, in document order. */
export function findHitTexts(): string[] {
  const registry = highlightRegistryMap()

  return [...(registry?.get('hermes-find') ?? [])].map(range => range.toString())
}

/** How many painted matches sit inside `root`. The registry is process-wide, so
 *  a test that plants more than one chat surface has to scope the count to the
 *  surface it is asserting on. */
export function findHitsWithin(root: Node): number {
  const registry = highlightRegistryMap()

  return [...(registry?.get('hermes-find') ?? [])].filter(range => root.contains(range.startContainer)).length
}

/** The text of the single ACTIVE match, or null when the bar sits on none. */
export function activeFindHitText(): null | string {
  const registry = highlightRegistryMap()

  return [...(registry?.get('hermes-find-active') ?? [])].map(range => range.toString())[0] ?? null
}

function highlightRegistryMap(): undefined | Map<string, Set<AbstractRange>> {
  return (globalThis as unknown as { CSS?: { highlights?: Map<string, Set<AbstractRange>> } }).CSS?.highlights
}
