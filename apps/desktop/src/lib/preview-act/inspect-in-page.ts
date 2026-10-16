import type {
  PreviewActAction,
  PreviewActHolder,
  PreviewActResult,
  PreviewInspectionNode,
  PreviewTargetInspection
} from './types'

/** Stringified into the guest: keep all runtime helpers inside this function.
 * No engine preamble, inventory, watcher, DOM writes, or input. Identity hints
 * belong only in the tool result, never the diagnostic trace. */
export function inspectTargetInPage(
  doc: Document,
  holder: PreviewActHolder | undefined,
  action: Pick<PreviewActAction, 'ref' | 'selector' | 'max'>
): PreviewActResult {
  const fail = (error: string): PreviewActResult => ({ success: false, error })
  const win = doc.defaultView

  if (!win) {
    return fail('Target inspection needs a live document.')
  }

  let matches: ArrayLike<Element>
  const ref = typeof action.ref === 'string' ? action.ref.trim() : ''

  // Preserve the engine's ref-first resolution, but NEVER survey or rebind.
  if (action.ref != null) {
    if (!ref) {
      return fail('Pass a non-empty ref for target inspection, or omit ref to use a selector.')
    }

    if (!holder || holder.url !== doc.location.href) {
      return fail('The page navigated or has no snapshot. Call plain elements for current refs.')
    }

    const bound = holder.book?.find(entry => entry.ref === ref)

    if (!bound) {
      return fail('Unknown element ref. Call plain elements for current refs.')
    }

    if (!doc.contains(bound.el)) {
      return fail('The referenced element was removed. Call plain elements for current refs.')
    }

    matches = [bound.el]
  } else {
    const selector = typeof action.selector === 'string' ? action.selector.trim() : ''

    if (!selector) {
      return fail('Pass a non-empty ref or CSS selector for target inspection.')
    }

    try {
      matches = doc.querySelectorAll(selector)
    } catch {
      // Never echo selectors or browser exception text, which can contain data.
      return fail('Not a valid CSS selector for target inspection.')
    }
  }

  const finite = (n: number) => (Number.isFinite(n) ? n : null)
  const short = (s: string | null) => (s || '').slice(0, 80)

  const identify = (el: Element): PreviewInspectionNode => {
    let nthOfType = 1

    for (let sibling = el.previousElementSibling; sibling; sibling = sibling.previousElementSibling) {
      if (sibling.localName === el.localName && sibling.namespaceURI === el.namespaceURI) {
        nthOfType++
      }
    }

    const node: PreviewInspectionNode = { tag: short(el.localName), nthOfType }
    const id = short(el.getAttribute('id'))
    const cls = short(el.getAttribute('class'))
    const testId = short(el.getAttribute('data-testid'))

    if (id) {
      node.id = id
    }

    if (cls) {
      node.class = cls
    }

    if (testId) {
      node.testId = testId
    }

    return node
  }

  const ancestors = (el: Element | null) => {
    const nodes: PreviewInspectionNode[] = []

    for (let parent = el?.parentElement; parent && nodes.length < 4; parent = parent.parentElement) {
      nodes.push(identify(parent))
    }

    return nodes
  }

  const inspection: PreviewTargetInspection = {
    coordinateSpace: 'guest-viewport-css-pixels',
    candidateCount: matches.length,
    truncated: false,
    viewport: {
      width: finite(win.innerWidth),
      height: finite(win.innerHeight),
      scrollX: finite(win.scrollX),
      scrollY: finite(win.scrollY),
      devicePixelRatio: finite(win.devicePixelRatio)
    },
    candidates: []
  }

  const limit = Number.isFinite(action.max) ? Math.max(1, Math.min(5, Math.floor(action.max!))) : 5

  for (let i = 0; i < matches.length && i < limit; i++) {
    const el = matches[i]
    const r = el.getBoundingClientRect()

    const rect =
      [r.left, r.top, r.right, r.bottom, r.width, r.height].every(Number.isFinite) && r.width > 0 && r.height > 0
        ? { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height }
        : null

    const x = rect ? finite(Math.round(rect.left + rect.width / 2)) : null
    const y = rect ? finite(Math.round(rect.top + rect.height / 2)) : null
    const point = x !== null && y !== null ? { x, y } : null

    const centerInViewport = !!(
      point &&
      point.x >= 0 &&
      point.y >= 0 &&
      point.x < win.innerWidth &&
      point.y < win.innerHeight
    )

    const hit = centerInViewport && point ? doc.elementFromPoint(point.x, point.y) : null
    const style = win.getComputedStyle(el)

    inspection.candidates.push({
      node: identify(el),
      ancestors: ancestors(el),
      rect,
      point,
      centerInViewport,
      style: {
        pointerEvents: short(style.pointerEvents),
        visibility: short(style.visibility),
        display: short(style.display)
      },
      hit: {
        node: hit ? identify(hit) : null,
        ancestors: ancestors(hit),
        relationship: !hit
          ? 'none'
          : hit === el
            ? 'self'
            : el.contains(hit)
              ? 'descendant'
              : hit.contains(el)
                ? 'ancestor'
                : 'unrelated'
      }
    })
  }

  const result: PreviewActResult = { success: true, inspection }

  inspection.truncated = inspection.candidates.length < matches.length

  // Hard cap including JSON escaping: 12k UTF-16 units, at most 36k UTF-8 bytes.
  // Drop whole entries, never produce partial JSON or misreport the match count.
  while (JSON.stringify(result).length > 12_000 && inspection.candidates.length) {
    inspection.candidates.pop()
    inspection.truncated = true
  }

  return result
}
