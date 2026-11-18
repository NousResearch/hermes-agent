// Chromium's native selection copy (Cmd+C, right-click Copy) serializes the
// selection as `text/html` with every element's COMPUTED paint inlined as
// style attributes. Copied from a dark theme, body text lands on the
// clipboard near-white; pasted into a light-background target (an email, a
// shared doc) it is invisible.
//
// The serializer only runs AFTER `copy` handlers decline to intercept —
// inside the handler, `clipboardData` reads back empty — so the guard cannot
// inspect or patch Chromium's payload. Instead it decides from the LIVE DOM:
// it scores the ink that paints the current selection against the theme
// Hermes is rendering, and when the two are opposite schemes it takes over
// the clipboard write entirely, emitting `text/plain` plus a tag-structured
// `text/html` with no paint declarations. Structure — headings, lists,
// tables, links, bold/italic, code layout — survives; colors come from the
// paste target's own defaults. Same-scheme selections are left to Chromium's
// default copy untouched.

type RenderedMode = 'light' | 'dark'

/** At or above this perceived luma, text reads as "light" (dark-theme ink). */
const LIGHT_TEXT_LUMA_MIN = 150
/** At or below this perceived luma, text reads as "dark" (light-theme ink). */
const DARK_TEXT_LUMA_MAX = 105

/** Upper bound on text nodes probed per copy; long selections stay O(50). */
const MAX_PROBE_NODES = 50

const RGB_FN_RE = /rgba?\(\s*([\d.]+)\s*[,\s]\s*([\d.]+)\s*[,\s]\s*([\d.]+)\s*(?:[/,]\s*([\d.%]+)\s*)?\)/i
const SRGB_FN_RE = /^color\(\s*srgb\s+([\d.]+%?)\s+([\d.]+%?)\s+([\d.]+%?)(?:\s*[/\s]+\s*([\d.%]+))?\s*\)$/i
const HEX_RE = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i

/** Parse one component: percentage or 0–255 integer. */
const channelValue = (raw: string): number => {
  const v = raw.trim()

  return v.endsWith('%') ? (Number.parseFloat(v) / 100) * 255 : Number.parseFloat(v)
}

/** color(srgb …) components are 0–1 floats (or percentages), not 0–255. */
const srgbChannelValue = (raw: string): number => {
  const v = raw.trim()

  return v.endsWith('%') ? (Number.parseFloat(v) / 100) * 255 : Number.parseFloat(v) * 255
}

const alphaValue = (raw: string): number => (raw.endsWith('%') ? Number.parseFloat(raw) / 100 : Number.parseFloat(raw))

function expandHex(hex: string): [number, number, number, number] | null {
  const h = hex.slice(1)
  const wide = h.length >= 6

  const pick = (i: number) => Number.parseInt(wide ? h.slice(i, i + 2) : h[i]! + h[i]!, 16)

  const r = pick(0)
  const g = pick(1 * (wide ? 2 : 1))
  const b = pick(2 * (wide ? 2 : 1))

  if ([r, g, b].some(Number.isNaN)) {
    return null
  }

  const aRaw = wide ? h.slice(6, 8) : h[3]

  return [r, g, b, aRaw ? Number.parseInt(aRaw, 16) / 255 : 1]
}

function parseCssColor(cssColor: string): { r: number; g: number; b: number; alpha: number } | null {
  const value = cssColor.trim()

  const rgb = value.match(RGB_FN_RE)

  if (rgb) {
    return {
      r: Number(rgb[1]),
      g: Number(rgb[2]),
      b: Number(rgb[3]),
      alpha: rgb[4] ? alphaValue(rgb[4]) : 1
    }
  }

  const srgb = value.match(SRGB_FN_RE)

  if (srgb) {
    return {
      r: srgbChannelValue(srgb[1] ?? ''),
      g: srgbChannelValue(srgb[2] ?? ''),
      b: srgbChannelValue(srgb[3] ?? ''),
      alpha: srgb[4] ? alphaValue(srgb[4]) : 1
    }
  }

  if (HEX_RE.test(value)) {
    const expanded = expandHex(value)

    if (expanded) {
      return { r: expanded[0], g: expanded[1], b: expanded[2], alpha: expanded[3] }
    }
  }

  return null
}

/**
 * Perceived luma (0–255, ITU-R BT.601 weighting) of a CSS color, with alpha
 * composited over mid-gray so partial opacity neither overstates nor hides
 * the paint. Accepts the serialization forms Chromium computes for the app's
 * tokens: rgb()/rgba(), color(srgb …) (the computed form of modern functions
 * like color-mix()), and hex. Returns null for anything else (keywords,
 * variables, non-sRGB spaces) or effectively invisible paint (alpha ≤ 0.05).
 */
export function textColorLuma(cssColor: string): number | null {
  const parsed = parseCssColor(cssColor)

  if (!parsed || parsed.alpha <= 0.05) {
    return null
  }

  const composite = (channel: number) => channel * parsed.alpha + 128 * (1 - parsed.alpha)

  return 0.2126 * composite(parsed.r) + 0.7152 * composite(parsed.g) + 0.0722 * composite(parsed.b)
}

function isOppositeScheme(textLuma: number, mode: RenderedMode): boolean {
  return mode === 'dark' ? textLuma >= LIGHT_TEXT_LUMA_MIN : textLuma <= DARK_TEXT_LUMA_MAX
}

function renderedMode(doc: Document): RenderedMode {
  const attr = doc.documentElement.dataset.hermesMode

  if (attr === 'light' || attr === 'dark') {
    return attr
  }

  // Boot frame or a host that skipped theme application: fall back to the OS
  // preference, which is what `system` mode resolves to anyway.
  try {
    return doc.defaultView?.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function rangeContainsNode(range: Range, node: Node): boolean {
  try {
    return range.intersectsNode(node)
  } catch {
    // Environments without intersectsNode: fall back to a boundary compare.
    try {
      range.comparePoint(node, 0)

      return true
    } catch {
      return false
    }
  }
}

/**
 * Mean perceived luma of the ink that paints the current selection, read
 * from the LIVE DOM: the computed color (preferring -webkit-text-fill-color
 * when set) of each selected text node's parent. Returns null when the
 * selection holds no scoreable ink.
 */
export function selectionInkLuma(sel: Selection, doc: Document): number | null {
  const view = doc.defaultView

  if (!view) {
    return null
  }

  const lumas: number[] = []
  const seen = new Set<Element>()

  for (let i = 0; i < sel.rangeCount && lumas.length < MAX_PROBE_NODES; i++) {
    const range = sel.getRangeAt(i)
    const root = range.commonAncestorContainer

    const walker = doc.createTreeWalker(
      root.nodeType === 1 ? root : (root.parentElement ?? doc.body),
      NodeFilter.SHOW_TEXT
    )

    let node = walker.nextNode()

    while (node && lumas.length < MAX_PROBE_NODES) {
      if (rangeContainsNode(range, node)) {
        const el = node.parentElement

        if (el && !seen.has(el)) {
          seen.add(el)

          const style = view.getComputedStyle(el)
          const fill = style.getPropertyValue('-webkit-text-fill-color')
          const ink = textColorLuma(fill) ?? textColorLuma(style.color)

          if (ink !== null) {
            lumas.push(ink)
          }
        }
      }

      node = walker.nextNode()
    }
  }

  if (lumas.length === 0) {
    return null
  }

  return lumas.reduce((sum, l) => sum + l, 0) / lumas.length
}

/**
 * Serialize the selection as tag-structured HTML with no paint and no
 * app-internal attributes: semantic elements (p, strong, em, a, ul, pre,
 * table, …) survive with their content and hrefs; style/class attributes —
 * the only carriers of Hermes' palette — are dropped so the paste target's
 * own color defaults apply.
 *
 * One exception to the strip-everything rule: the result carries a GENERIC
 * font-family (`sans-serif`, `monospace` inside code). With no family at
 * all, receivers that convert HTML to rich text (macOS Mail, Notes,
 * TextEdit) fall back to the BROWSER default — Times — instead of their own
 * compose font. Naming the generic family keeps the paste sans like the app
 * renders it, while each platform resolves it to its own system face.
 */
export function serializeSelectionStructure(sel: Selection, doc: Document): string {
  const container = doc.createElement('div')

  for (let i = 0; i < sel.rangeCount; i++) {
    container.append(sel.getRangeAt(i).cloneContents())
  }

  for (const el of container.querySelectorAll('[style]')) {
    el.removeAttribute('style')
  }

  for (const el of container.querySelectorAll('[class]')) {
    el.removeAttribute('class')
  }

  const wrapper = doc.createElement('div')

  wrapper.style.fontFamily = 'sans-serif'

  while (container.firstChild) {
    wrapper.append(container.firstChild)
  }

  for (const el of wrapper.querySelectorAll<HTMLElement>('pre, code')) {
    el.style.fontFamily = 'monospace'
  }

  // outerHTML, not innerHTML: the styled wrapper IS the font anchor.
  return wrapper.outerHTML
}

function selectionStartsInEditable(sel: Selection): boolean {
  const anchor = sel.anchorNode

  if (!anchor) {
    return false
  }

  const el = anchor.nodeType === 1 ? (anchor as Element) : anchor.parentElement

  return Boolean(el?.closest('input, textarea, [contenteditable="true"], [contenteditable=""]'))
}

/**
 * Install the document-level `copy` interceptor. Returns a dispose function.
 *
 * Runs in the CAPTURE phase so inner handlers cannot run first. Only an
 * off-scheme selection is intercepted (payload owned and rewritten); every
 * other copy — same-scheme, editable-field, empty — passes through to
 * Chromium's default untouched.
 */
export function installSelectionCopyColorGuard(doc: Document = document): () => void {
  const trace = (entry: Record<string, unknown>) => {
    if (import.meta.env.DEV) {
      const view = doc.defaultView as (Window & { __copyGuardLog?: unknown[] }) | null

      if (view) {
        ;(view.__copyGuardLog ??= []).push(entry)
      }
    }
  }

  const onCopy = (event: ClipboardEvent) => {
    const sel = doc.getSelection()

    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
      return
    }

    if (selectionStartsInEditable(sel)) {
      trace({ step: 'editable-skip' })

      return
    }

    const mode = renderedMode(doc)
    const inkLuma = selectionInkLuma(sel, doc)

    if (inkLuma === null || !isOppositeScheme(inkLuma, mode)) {
      trace({ step: inkLuma === null ? 'no-ink' : 'same-scheme', inkLuma, mode })

      return
    }

    const clipboard = event.clipboardData

    if (!clipboard) {
      return
    }

    const plain = sel.toString()
    const html = serializeSelectionStructure(sel, doc)

    // Owning the payload is the only way to change it: Chromium's own
    // serialization is produced after handlers decline, and preventDefault
    // is required for setData writes to stick.
    event.preventDefault()
    clipboard.setData('text/plain', plain)
    clipboard.setData('text/html', html)
    trace({ step: 'owned-payload', mode, inkLuma, plainChars: plain.length, htmlChars: html.length })
  }

  doc.addEventListener('copy', onCopy, true)

  return () => doc.removeEventListener('copy', onCopy, true)
}
