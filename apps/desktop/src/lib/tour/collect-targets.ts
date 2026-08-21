/**
 * TOUR TARGET COLLECTOR — discovers what a tour can point at.
 *
 * Fully generic: works on ANY document (the Hermes app's own DOM or an
 * arbitrary page in the preview webview) by scanning for the things that make
 * an element addressable — explicit `data-tour` markers first, then the
 * standard affordances (aria-labels, roles, headings, labelled interactive
 * elements). Each target carries a CSS selector, a human label, and whether
 * that selector is STABLE (an identity attribute that survives a re-render) or
 * merely positional, so a caller can prefer the durable ones.
 *
 * Injected into preview webviews as source (`collectTourTargets.toString()`),
 * so it must stay self-contained: no imports, no closures, no renderer
 * globals. Types erase at compile time.
 */

export interface TourTarget {
  /** Human-readable label (aria-label, text content, alt, title …). */
  label: string
  /** Viewport rect, rounded: [x, y, width, height]. */
  rect: [number, number, number, number]
  /** Element role: explicit ARIA role or the tag name. */
  role: string
  /** A CSS selector that uniquely matches this element right now. */
  selector: string
  /** True when the selector keys off identity (data-tour, id, data-testid,
   *  aria-label) rather than DOM position — i.e. it survives a re-render. */
  stable: boolean
}

/** Scan `doc` for tourable elements (capped at `max`). Self-contained. */
export function collectTourTargets(doc: Document, max: number): TourTarget[] {
  const results: TourTarget[] = []
  const seen = new Set<Element>()

  const cssEscape = (value: string) =>
    typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(value) : value.replace(/["\\]/g, '\\$&')

  const labelOf = (el: Element): string => {
    for (const attr of ['aria-label', 'title', 'alt', 'placeholder']) {
      const value = el.getAttribute(attr)

      if (value) {
        return value
      }
    }

    const text = (el.textContent || '').trim().replace(/\s+/g, ' ')

    return text.length > 80 ? text.slice(0, 77) + '…' : text
  }

  /** Identity-based selector, or '' when the element has no durable handle. */
  const stableSelector = (el: Element): string => {
    const dataTour = el.getAttribute('data-tour')

    if (dataTour) {
      return '[data-tour="' + cssEscape(dataTour) + '"]'
    }

    if (el.id) {
      return '#' + cssEscape(el.id)
    }

    const testId = el.getAttribute('data-testid')

    if (testId) {
      return '[data-testid="' + cssEscape(testId) + '"]'
    }

    const aria = el.getAttribute('aria-label')
    const byAria = aria ? el.tagName.toLowerCase() + '[aria-label="' + cssEscape(aria) + '"]' : ''

    return byAria && doc.querySelectorAll(byAria).length === 1 ? byAria : ''
  }

  /** Positional fallback: tag + nth-child path up to an id or <body>. */
  const positionalSelector = (el: Element): string => {
    const path: string[] = []
    let node: Element | null = el

    while (node && node !== doc.body && path.length < 8) {
      if (node.id) {
        path.unshift('#' + cssEscape(node.id))

        break
      }

      const parent: Element | null = node.parentElement
      const index = parent ? Array.prototype.indexOf.call(parent.children, node) : -1

      path.unshift(node.tagName.toLowerCase() + (index >= 0 ? ':nth-child(' + (index + 1) + ')' : ''))
      node = parent
    }

    return path.join(' > ')
  }

  const visible = (el: Element): boolean => {
    const rect = el.getBoundingClientRect()

    if (rect.width < 4 || rect.height < 4) {
      return false
    }

    const win = doc.defaultView

    return !win || (rect.bottom > 0 && rect.top < win.innerHeight && rect.right > 0 && rect.left < win.innerWidth)
  }

  const push = (el: Element) => {
    if (results.length >= max || seen.has(el) || !visible(el)) {
      return
    }

    const label = labelOf(el)

    if (!label) {
      return
    }

    const stable = stableSelector(el)
    const selector = stable || positionalSelector(el)

    // Only report selectors that resolve back to this exact element.
    if (!selector || doc.querySelector(selector) !== el) {
      return
    }

    const rect = el.getBoundingClientRect()

    seen.add(el)
    results.push({
      label,
      rect: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)],
      role: el.getAttribute('role') || el.tagName.toLowerCase(),
      selector,
      stable: !!stable
    })
  }

  // Priority order: explicit tour markers, then landmarks/headings, then
  // labelled interactive elements. Each tier only fills remaining capacity.
  for (const el of doc.querySelectorAll('[data-tour]')) {
    push(el)
  }

  for (const el of doc.querySelectorAll('nav, main, aside, header, footer, [role], h1, h2, h3')) {
    push(el)
  }

  for (const el of doc.querySelectorAll('a[href], button, input, select, textarea, [tabindex], [aria-label]')) {
    push(el)
  }

  // Durable handles first: a caller picking off the top gets selectors that
  // survive a re-render, and positional ones stay available as a last resort.
  return results.sort((a, b) => Number(b.stable) - Number(a.stable))
}
