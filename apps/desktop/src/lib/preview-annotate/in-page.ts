/**
 * Guest-page comment overlay. Injected as source into the preview webview —
 * no imports, no module constants, every value the body needs lives inside
 * `annotateInPage`. Same contract as `watchInPage`.
 *
 * pointer-events on the host capture clicks while mode is on; hit-testing
 * briefly disables them so elementFromPoint sees the real page. Teardown
 * removes the host entirely so nothing can stick over the page.
 */

export const ANNOTATE_HOST_TAG = 'hermes-annotate'

export interface AnnotatePageRect {
  height: number
  width: number
  x: number
  y: number
}

export interface AnnotatePageIdentity {
  css: Record<string, string>
  html: string
  selector: string
  tag: string
  text: string
}

export type AnnotatePageEvent =
  | { identity: AnnotatePageIdentity; rect: AnnotatePageRect; type: 'pick-element' }
  | { rect: AnnotatePageRect; type: 'pick-area' }
  | { rect: AnnotatePageRect; type: 'reposition' }
  | { type: 'end' }

export interface AnnotatePinChrome {
  kind: 'area' | 'element'
  number: number
  page?: AnnotatePageRect
  rect: AnnotatePageRect
  selector?: string
}

/** Markup tools: freehand, arrow, and the two shapes. `rectangle` and `ellipse`
 *  are drawn corner to corner, the arrow from tail to head. */
export type AnnotateMarkupTool = 'arrow' | 'draw' | 'ellipse' | 'rect'

export interface AnnotateMarkupPoint {
  x: number
  y: number
}

export interface AnnotateMarkupStroke {
  /** PAGE coordinates (client + scroll offset): a circle drawn around a heading
   *  stays around it when the page scrolls, and lands in the crop wherever it
   *  was drawn. */
  points: AnnotateMarkupPoint[]
  tool: AnnotateMarkupTool
}

export interface AnnotateInPage {
  beginCapture: () => Promise<boolean>
  beginMarkup: (tool: AnnotateMarkupTool) => void
  clearMarkup: () => void
  endCapture: () => void
  endMarkup: () => void
  getMarkup: () => AnnotateMarkupStroke[]
  getMarkerNumbers: () => number[]
  getOutlineColor: () => string
  hideDraft: () => void
  install: () => void
  isInstalled: () => boolean
  relocate: () => void
  showDraft: (rect: AnnotatePageRect, number: number) => void
  showPins: (pins: AnnotatePinChrome[]) => void
  teardown: () => void
  undoMarkup: () => void
  wait: () => Promise<AnnotatePageEvent>
}

export function annotateInPage(doc: Document): AnnotateInPage {
  const win = doc.defaultView
  // Literals, not imports: this function is stringified into the guest page.
  const blue = '#2F80ED'
  const fill = 'rgba(47, 128, 237, 0.14)'
  const markerSize = 22
  const stroke = '2px'

  const cssKeys = [
    'color',
    'background-color',
    'font-size',
    'font-family',
    'font-weight',
    'line-height',
    'letter-spacing',
    'text-align',
    'display',
    'position',
    'width',
    'height',
    'max-width',
    'padding',
    'margin',
    'border',
    'border-radius',
    'box-shadow',
    'opacity',
    'overflow',
    'z-index',
    'transform',
    'flex-direction',
    'gap',
    'grid-template-columns',
    'justify-content',
    'align-items'
  ]

  const htmlBudget = 600

  let host: HTMLElement | null = null
  let shadow: ShadowRoot | null = null
  let hoverBox: HTMLElement | null = null
  let draftBox: HTMLElement | null = null
  let marquee: HTMLElement | null = null
  let pinsLayer: HTMLElement | null = null
  let markupLayer: SVGSVGElement | null = null
  let markupTool: AnnotateMarkupTool | null = null
  let marking: { node: SVGGElement; points: AnnotateMarkupPoint[]; tool: AnnotateMarkupTool } | null = null
  const markupStrokes: { node: SVGGElement; points: AnnotateMarkupPoint[]; tool: AnnotateMarkupTool }[] = []
  let waiting: ((event: AnnotatePageEvent) => void) | null = null
  const queued: AnnotatePageEvent[] = []
  let drag: { x: number; y: number } | null = null
  let areaForced = false
  let liveDraft: { el: Element | null; page: AnnotatePageRect } | null = null
  let livePins: { el: Element | null; number: number; page: AnnotatePageRect; wrap: HTMLElement }[] = []
  let raf = 0
  let lastDraftKey = ''

  const style = (el: HTMLElement, props: Record<string, string>) => {
    for (const name of Object.keys(props)) {
      el.style.setProperty(name, props[name])
    }
  }

  const emit = (event: AnnotatePageEvent) => {
    if (waiting) {
      const resolve = waiting
      waiting = null
      resolve(event)

      return
    }

    queued.push(event)
  }

  const box = (el: Element): AnnotatePageRect => {
    const rect = el.getBoundingClientRect()

    return { height: rect.height, width: rect.width, x: rect.left, y: rect.top }
  }

  const pageOffset = () => {
    const scrolling = doc.scrollingElement || doc.documentElement

    return {
      x: win?.scrollX ?? scrolling.scrollLeft ?? 0,
      y: win?.scrollY ?? scrolling.scrollTop ?? 0
    }
  }

  const cssEscape = (value: string): string => {
    const CSS = win && win.CSS

    if (CSS && CSS.escape) {
      return CSS.escape(value)
    }

    return value.replace(/([^\w-])/g, '\\$1')
  }

  const canScroll = (node: Element, axis: 'x' | 'y') => {
    if (!win) {
      return false
    }

    const style = win.getComputedStyle(node)
    const overflow = axis === 'y' ? style.overflowY : style.overflowX
    const scrollable = overflow === 'auto' || overflow === 'scroll' || overflow === 'overlay'

    if (!scrollable) {
      return false
    }

    return axis === 'y' ? node.scrollHeight > node.clientHeight + 1 : node.scrollWidth > node.clientWidth + 1
  }

  const toPage = (rect: AnnotatePageRect): AnnotatePageRect => {
    const offset = pageOffset()

    return { height: rect.height, width: rect.width, x: rect.x + offset.x, y: rect.y + offset.y }
  }

  const toView = (pageRect: AnnotatePageRect): AnnotatePageRect => {
    const offset = pageOffset()

    return { height: pageRect.height, width: pageRect.width, x: pageRect.x - offset.x, y: pageRect.y - offset.y }
  }

  const viewOf = (track: { el: Element | null; page: AnnotatePageRect }): AnnotatePageRect => {
    if (track.el && track.el.isConnected) {
      return box(track.el)
    }

    return toView(track.page)
  }

  const cssPath = (el: Element): string => {
    const parts: string[] = []
    let node: Element | null = el

    while (node && node.nodeType === 1 && parts.length < 5) {
      const tag = node.tagName.toLowerCase()

      if (node.id) {
        parts.unshift(`#${cssEscape(node.id)}`)

        break
      }

      const cls = (node.getAttribute('class') || '')
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map(name => `.${cssEscape(name)}`)
        .join('')

      let sel = tag + cls
      const parent: Element | null = node.parentElement

      if (parent) {
        const same = Array.from(parent.children).filter(child => child.tagName === node!.tagName)

        if (same.length > 1) {
          sel += `:nth-of-type(${same.indexOf(node) + 1})`
        }
      }

      parts.unshift(sel)
      node = parent

      if (node && (node === doc.documentElement || tag === 'body')) {
        break
      }
    }

    return parts.join('>')
  }

  const readCss = (el: Element): Record<string, string> => {
    const computed = win?.getComputedStyle(el)
    const out: Record<string, string> = {}

    if (!computed) {
      return out
    }

    for (const key of cssKeys) {
      const value = computed.getPropertyValue(key).trim()

      if (!value || value === 'normal' || value === 'none' || value === 'auto' || value === '0px') {
        continue
      }

      out[key] = value.length > 80 ? `${value.slice(0, 79)}…` : value
    }

    return out
  }

  /**
   * Markup for the picked element, with anything secret-shaped stripped first.
   *
   * A comment is user-authored context, but the element under the cursor is
   * whatever the page put there: a filled password box, a token in a hidden
   * input, an api-key data attribute. Redaction happens on a clone here, in
   * the guest, so the secret never reaches the host, the composer, or the
   * model. Same reason `browser_type` masks what it types.
   *
   * Walk every descendant, not just form controls. A comment on a parent of a
   * `data-api-key` node still serializes that child. Also clear the live
   * `.value` of secret-shaped fields: typed passwords and secret textareas do
   * not always show up as attributes. Target text is taken from this same
   * clone so a textarea body cannot leak after the HTML is clean.
   */
  const secretName = /key|token|secret|password|auth|session|credential/

  const looksSecret = (value: string): boolean => secretName.test(value.toLowerCase())

  const isSecretControl = (node: Element): boolean => {
    const tag = node.tagName.toLowerCase()
    const type = (node.getAttribute('type') || '').toLowerCase()

    if (tag === 'input' && (type === 'password' || type === 'hidden')) {
      return true
    }

    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      return (
        looksSecret(node.getAttribute('name') || '') ||
        looksSecret(node.id || '') ||
        looksSecret(node.getAttribute('autocomplete') || '')
      )
    }

    return node.hasAttribute('data-secret')
  }

  const clearControlValue = (node: Element): void => {
    const tag = node.tagName.toLowerCase()

    if (tag === 'input' || tag === 'select') {
      const field = node as HTMLInputElement
      field.value = '[redacted]'
      node.setAttribute('value', '[redacted]')
    }

    if (tag === 'textarea') {
      const field = node as HTMLTextAreaElement
      field.value = '[redacted]'
      node.textContent = '[redacted]'
    }

    if (node.hasAttribute('data-secret') && tag !== 'input' && tag !== 'textarea' && tag !== 'select') {
      node.textContent = '[redacted]'
    }
  }

  const redactClone = (el: Element): Element | null => {
    let clone: Element

    try {
      clone = el.cloneNode(true) as Element
    } catch {
      return null
    }

    const nodes: Element[] = [clone]
    const nested = clone.querySelectorAll('*')

    for (let i = 0; i < nested.length; i++) {
      nodes.push(nested[i])
    }

    for (const node of nodes) {
      const names = node.getAttributeNames()

      for (const name of names) {
        if (looksSecret(name)) {
          node.setAttribute(name, '[redacted]')
        }
      }

      if (isSecretControl(node)) {
        clearControlValue(node)
      }
    }

    return clone
  }

  const budgetHtml = (html: string): string => {
    if (html.length <= htmlBudget) {
      return html
    }

    // Keep the opening tag, where the classes and props live, over the tail.
    return `${html.slice(0, htmlBudget - 1)}…`
  }

  const identityOf = (el: Element): AnnotatePageIdentity => {
    const clone = redactClone(el)
    const html = clone ? budgetHtml(clone.outerHTML || '') : ''
    const text = ((clone ?? el).textContent || '').replace(/\s+/g, ' ').trim()

    return {
      css: readCss(el),
      html,
      selector: cssPath(el),
      tag: el.tagName.toLowerCase(),
      text: text.length > 80 ? `${text.slice(0, 79)}…` : text
    }
  }

  const underPoint = (x: number, y: number): Element | null => {
    if (!host) {
      return doc.elementFromPoint(x, y)
    }

    const previous = host.style.getPropertyValue('pointer-events')
    host.style.setProperty('pointer-events', 'none')
    const hit = doc.elementFromPoint(x, y)
    host.style.setProperty('pointer-events', previous || 'auto')

    if (!hit || hit === host || hit === doc.documentElement || hit === doc.body) {
      return null
    }

    return hit
  }

  const place = (el: HTMLElement, rect: AnnotatePageRect) => {
    style(el, {
      height: `${Math.max(1, rect.height)}px`,
      left: `${rect.x}px`,
      top: `${rect.y}px`,
      width: `${Math.max(1, rect.width)}px`
    })
  }

  const relocate = () => {
    // Strokes are stored in page coordinates, so the fixed layer rides the scroll.
    paintMarkupLayer()

    for (let i = 0; i < livePins.length; i++) {
      const pin = livePins[i]

      if (pin.el && pin.el.isConnected) {
        pin.page = toPage(box(pin.el))
      }

      place(pin.wrap, viewOf(pin))
    }

    if (liveDraft && draftBox && draftBox.style.getPropertyValue('display') !== 'none') {
      if (liveDraft.el && liveDraft.el.isConnected) {
        liveDraft.page = toPage(box(liveDraft.el))
      }

      const rect = viewOf(liveDraft)
      place(draftBox, rect)
      const key = [rect.x, rect.y, rect.width, rect.height].join(',')

      if (key !== lastDraftKey) {
        lastDraftKey = key
        emit({ rect, type: 'reposition' })
      }
    }
  }

  const loop = () => {
    if (!host || !host.isConnected) {
      raf = 0

      return
    }

    relocate()

    if (win) {
      raf = win.requestAnimationFrame(loop)
    }
  }

  const ensureLoop = () => {
    if (win && !raf) {
      raf = win.requestAnimationFrame(loop)
    }
  }

  const paintOutline = (el: HTMLElement, heavy: boolean) => {
    style(el, {
      background: heavy ? fill : 'transparent',
      border: `${stroke} solid ${blue}`,
      'border-radius': '4px',
      'box-shadow': heavy ? `0 0 0 1px ${blue}, 0 8px 24px rgba(15, 23, 42, 0.18)` : `0 0 0 1px ${blue}`,
      'box-sizing': 'border-box',
      'pointer-events': 'none',
      position: 'fixed'
    })
  }

  const makeMarker = (number: number): HTMLElement => {
    const mark = doc.createElement('div')
    mark.setAttribute('data-annotate-marker', String(number))
    mark.textContent = String(number)
    style(mark, {
      'align-items': 'center',
      background: blue,
      border: '2px solid #fff',
      'border-radius': '999px',
      'box-shadow': '0 2px 8px rgba(15, 23, 42, 0.28)',
      color: '#fff',
      display: 'flex',
      'font-family': "-apple-system, 'Segoe UI', sans-serif",
      'font-size': '11px',
      'font-weight': '700',
      height: `${markerSize}px`,
      'justify-content': 'center',
      left: `${-markerSize / 2}px`,
      'letter-spacing': '0',
      'line-height': '1',
      'pointer-events': 'none',
      position: 'absolute',
      top: `${-markerSize / 2}px`,
      width: `${markerSize}px`,
      'z-index': '2'
    })

    return mark
  }

  const SVG_NS = 'http://www.w3.org/2000/svg'

  const svgNode = (name: string): SVGElement => doc.createElementNS(SVG_NS, name) as SVGElement

  /** Client point → page point, so a stroke stays on its target when the page scrolls. */
  const pagePoint = (clientX: number, clientY: number): AnnotateMarkupPoint => {
    const offset = pageOffset()

    return { x: clientX + offset.x, y: clientY + offset.y }
  }

  const strokePaint = { fill: 'none', stroke: blue, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '3' }

  /** The markup layer covers the viewport; its children are page-positioned. */
  const paintMarkupLayer = () => {
    if (!markupLayer) {
      return
    }

    const offset = pageOffset()

    style(markupLayer as unknown as HTMLElement, {
      height: '100%',
      left: '0',
      overflow: 'visible',
      'pointer-events': 'none',
      position: 'fixed',
      top: '0',
      transform: 'translate(' + -offset.x + 'px,' + -offset.y + 'px)',
      width: '100%'
    })
  }

  /** Repaint one stroke from its points: the shape follows the pointer. */
  const paintStroke = (stroke: { node: SVGGElement; points: AnnotateMarkupPoint[]; tool: AnnotateMarkupTool }) => {
    const start = stroke.points[0]

    stroke.node.replaceChildren()

    if (!start) {
      return
    }

    const add = (name: string, attrs: Record<string, string>, paint: Record<string, string> = strokePaint) => {
      const node = svgNode(name)

      for (const key of Object.keys(attrs)) {
        node.setAttribute(key, attrs[key])
      }

      for (const key of Object.keys(paint)) {
        node.setAttribute(key, paint[key])
      }

      stroke.node.appendChild(node)
    }

    if (stroke.tool === 'draw') {
      add('polyline', { points: stroke.points.map(point => point.x + ',' + point.y).join(' ') })

      return
    }

    const end = stroke.points[stroke.points.length - 1]

    if (!end) {
      return
    }

    if (stroke.tool === 'rect') {
      add('rect', {
        height: String(Math.abs(end.y - start.y)),
        width: String(Math.abs(end.x - start.x)),
        x: String(Math.min(start.x, end.x)),
        y: String(Math.min(start.y, end.y))
      })

      return
    }

    if (stroke.tool === 'ellipse') {
      add('ellipse', {
        cx: String((start.x + end.x) / 2),
        cy: String((start.y + end.y) / 2),
        rx: String(Math.abs(end.x - start.x) / 2),
        ry: String(Math.abs(end.y - start.y) / 2)
      })

      return
    }

    add('line', { x1: String(start.x), x2: String(end.x), y1: String(start.y), y2: String(end.y) })

    const angle = Math.atan2(end.y - start.y, end.x - start.x)
    const head = 14

    const wing = (spread: number) => ({
      x: String(end.x - head * Math.cos(angle - spread)),
      y: String(end.y - head * Math.sin(angle - spread))
    })

    const left = wing(0.42)
    const right = wing(-0.42)

    add(
      'polygon',
      { points: end.x + ',' + end.y + ' ' + left.x + ',' + left.y + ' ' + right.x + ',' + right.y },
      { fill: blue, stroke: blue, 'stroke-linejoin': 'round', 'stroke-width': '2' }
    )
  }

  const beginMarkup = (tool: AnnotateMarkupTool) => {
    ensure()
    markupTool = tool
    drag = null

    if (hoverBox) {
      style(hoverBox, { display: 'none' })
    }

    if (marquee) {
      style(marquee, { display: 'none' })
    }
  }

  const endMarkup = () => {
    markupTool = null
    marking = null
  }

  const undoMarkup = () => {
    const last = markupStrokes.pop()

    last?.node.remove()
  }

  const clearMarkup = () => {
    markupStrokes.length = 0
    marking = null
    markupLayer?.replaceChildren()
  }

  const ensure = () => {
    if (host && shadow && hoverBox && draftBox && marquee && pinsLayer) {
      return
    }

    const stale = doc.querySelectorAll('hermes-annotate')

    for (let i = 0; i < stale.length; i++) {
      stale[i].remove()
    }

    host = doc.createElement('hermes-annotate')
    host.setAttribute('aria-hidden', 'true')
    host.setAttribute('data-annotate-host', 'true')
    style(host, {
      background: 'transparent',
      border: '0',
      cursor: 'crosshair',
      display: 'block',
      height: '100%',
      inset: '0',
      margin: '0',
      overflow: 'visible',
      padding: '0',
      'pointer-events': 'auto',
      position: 'fixed',
      'user-select': 'none',
      width: '100%',
      'z-index': '2147483000'
    })

    shadow = host.attachShadow({ mode: 'open' })
    hoverBox = doc.createElement('div')
    hoverBox.setAttribute('data-annotate-outline', 'hover')
    paintOutline(hoverBox, false)
    style(hoverBox, { display: 'none', opacity: '0.85' })

    draftBox = doc.createElement('div')
    draftBox.setAttribute('data-annotate-outline', 'draft')
    paintOutline(draftBox, true)
    style(draftBox, { display: 'none' })
    draftBox.appendChild(makeMarker(1))

    marquee = doc.createElement('div')
    marquee.setAttribute('data-annotate-marquee', 'true')
    style(marquee, {
      background: fill,
      border: `1.5px dashed ${blue}`,
      'border-radius': '2px',
      display: 'none',
      'pointer-events': 'none',
      position: 'fixed'
    })

    pinsLayer = doc.createElement('div')
    style(pinsLayer, { inset: '0', 'pointer-events': 'none', position: 'fixed' })

    shadow.appendChild(hoverBox)
    shadow.appendChild(draftBox)
    shadow.appendChild(marquee)
    shadow.appendChild(pinsLayer)

    // Markup sits in its own layer ABOVE the page and above the pins, and it is
    // deliberately NOT hidden by beginCapture: a circle drawn around the target
    // is the point of the screenshot.
    markupLayer = svgNode('svg') as SVGSVGElement
    markupLayer.setAttribute('data-annotate-markup', 'true')
    shadow.appendChild(markupLayer)
    paintMarkupLayer()
    doc.documentElement.appendChild(host)
  }

  const onMove = (event: MouseEvent) => {
    if (!hoverBox) {
      return
    }

    if (marking) {
      if (marking.tool === 'draw') {
        marking.points.push(pagePoint(event.clientX, event.clientY))
      } else {
        marking.points[1] = pagePoint(event.clientX, event.clientY)
      }

      paintStroke(marking)

      return
    }

    if (drag && marquee) {
      const x = Math.min(drag.x, event.clientX)
      const y = Math.min(drag.y, event.clientY)
      const width = Math.abs(event.clientX - drag.x)
      const height = Math.abs(event.clientY - drag.y)
      style(marquee, { display: 'block' })
      place(marquee, { height, width, x, y })
      style(hoverBox, { display: 'none' })

      return
    }

    const hit = underPoint(event.clientX, event.clientY)

    if (!hit) {
      style(hoverBox, { display: 'none' })

      return
    }

    style(hoverBox, { display: 'block' })
    place(hoverBox, box(hit))
  }

  const onDown = (event: MouseEvent) => {
    if (event.button !== 0) {
      return
    }

    // A tool is armed: this gesture draws, it does not pick or box.
    if (markupTool && markupLayer) {
      event.preventDefault()
      event.stopPropagation()

      const node = svgNode('g') as SVGGElement

      node.setAttribute('data-annotate-stroke', markupTool)
      markupLayer.appendChild(node)
      marking = { node, points: [pagePoint(event.clientX, event.clientY)], tool: markupTool }
      paintStroke(marking)

      return
    }

    event.preventDefault()
    event.stopPropagation()
    drag = { x: event.clientX, y: event.clientY }
    areaForced = event.shiftKey
  }

  const onUp = (event: MouseEvent) => {
    if (marking) {
      event.preventDefault()
      event.stopPropagation()

      const stroke = marking
      const [start, end] = [stroke.points[0], stroke.points[stroke.points.length - 1]]

      const drawn =
        stroke.tool === 'draw'
          ? stroke.points.length > 2
          : Boolean(start && end && Math.abs(end.x - start.x) + Math.abs(end.y - start.y) > 12)

      marking = null

      // A click with a tool armed is a mis-click, not a one-pixel squiggle.
      if (!drawn) {
        stroke.node.remove()

        return
      }

      markupStrokes.push(stroke)

      return
    }

    if (!drag) {
      return
    }

    event.preventDefault()
    event.stopPropagation()

    const start = drag
    drag = null
    const width = Math.abs(event.clientX - start.x)
    const height = Math.abs(event.clientY - start.y)
    const isArea = areaForced || width > 8 || height > 8
    areaForced = false

    if (marquee) {
      style(marquee, { display: 'none' })
    }

    if (isArea) {
      const rect = {
        height: Math.max(8, height),
        width: Math.max(8, width),
        x: Math.min(start.x, event.clientX),
        y: Math.min(start.y, event.clientY)
      }

      liveDraft = { el: null, page: toPage(rect) }
      lastDraftKey = ''
      emit({ rect, type: 'pick-area' })

      return
    }

    const hit = underPoint(event.clientX, event.clientY)

    if (!hit) {
      return
    }

    const rect = box(hit)
    liveDraft = { el: hit, page: toPage(rect) }
    lastDraftKey = ''
    emit({ identity: identityOf(hit), rect, type: 'pick-element' })
  }

  const onWheel = (event: WheelEvent) => {
    event.preventDefault()
    host?.style.setProperty('pointer-events', 'none')
    const under = doc.elementFromPoint(event.clientX, event.clientY)
    host?.style.setProperty('pointer-events', 'auto')
    let scroller: Element | null = under

    while (scroller) {
      if (canScroll(scroller, 'y') || canScroll(scroller, 'x')) {
        scroller.scrollTop += event.deltaY
        scroller.scrollLeft += event.deltaX
        relocate()

        return
      }

      scroller = scroller.parentElement
    }

    if (win) {
      win.scrollBy(event.deltaX, event.deltaY)
    } else {
      const root = doc.scrollingElement || doc.documentElement
      root.scrollTop += event.deltaY
      root.scrollLeft += event.deltaX
    }

    relocate()
  }

  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      emit({ type: 'end' })
    }
  }

  const bind = () => {
    if (!host) {
      return
    }

    host.addEventListener('mousedown', onDown, true)
    host.addEventListener('mousemove', onMove, true)
    host.addEventListener('mouseup', onUp, true)
    host.addEventListener('wheel', onWheel, { passive: false })
    win?.addEventListener('keydown', onKey, true)
    win?.addEventListener('scroll', relocate, true)
    win?.addEventListener('resize', relocate)
    doc.addEventListener('scroll', relocate, true)
    win?.visualViewport?.addEventListener('scroll', relocate)
    win?.visualViewport?.addEventListener('resize', relocate)
  }

  const unbind = () => {
    host?.removeEventListener('mousedown', onDown, true)
    host?.removeEventListener('mousemove', onMove, true)
    host?.removeEventListener('mouseup', onUp, true)
    host?.removeEventListener('wheel', onWheel)
    win?.removeEventListener('keydown', onKey, true)
    win?.removeEventListener('scroll', relocate, true)
    win?.removeEventListener('resize', relocate)
    doc.removeEventListener('scroll', relocate, true)
    win?.visualViewport?.removeEventListener('scroll', relocate)
    win?.visualViewport?.removeEventListener('resize', relocate)

    if (win && raf) {
      win.cancelAnimationFrame(raf)
      raf = 0
    }
  }

  const showPins = (pins: AnnotatePinChrome[]) => {
    ensure()

    if (!pinsLayer) {
      return
    }

    const previous = livePins
    pinsLayer.replaceChildren()
    livePins = []

    for (const pin of pins) {
      let el: Element | null = null
      const old = previous.find(item => item.number === pin.number)

      if (old && old.el && old.el.isConnected) {
        el = old.el
      } else if (liveDraft && liveDraft.el && liveDraft.el.isConnected) {
        el = liveDraft.el
      }

      if (!el && pin.selector) {
        try {
          el = doc.querySelector(pin.selector)
        } catch {
          el = null
        }
      }

      const page = el ? toPage(box(el)) : pin.page || old?.page || toPage(pin.rect)
      const wrap = doc.createElement('div')
      wrap.setAttribute('data-annotate-pin', String(pin.number))
      paintOutline(wrap, true)
      place(wrap, el ? box(el) : toView(page))
      wrap.appendChild(makeMarker(pin.number))
      pinsLayer.appendChild(wrap)
      livePins.push({ el, number: pin.number, page, wrap })
    }

    liveDraft = null
    lastDraftKey = ''
    ensureLoop()
  }

  const showDraft = (rect: AnnotatePageRect, number: number) => {
    ensure()

    if (!draftBox) {
      return
    }

    if (!liveDraft) {
      liveDraft = { el: null, page: toPage(rect) }
    }

    draftBox.replaceChildren(makeMarker(number))
    style(draftBox, { display: 'block' })
    place(draftBox, liveDraft.el && liveDraft.el.isConnected ? box(liveDraft.el) : viewOf(liveDraft))

    if (hoverBox) {
      style(hoverBox, { display: 'none' })
    }

    ensureLoop()
  }

  const hideDraft = () => {
    liveDraft = null
    lastDraftKey = ''

    if (draftBox) {
      style(draftBox, { display: 'none' })
    }
  }

  /**
   * Two frames. `executeJavaScript` resolving only means the style property is
   * set — the compositor has not drawn it yet, and `capturePage` grabs whatever
   * is on screen. Capturing straight after `showDraft` therefore photographs
   * the page one frame before the marker exists, which is why saved crops came
   * back outlined but unnumbered. One rAF schedules us before the next paint;
   * the second lands after it.
   */
  const afterPaint = (): Promise<void> =>
    new Promise(resolve => {
      if (!win) {
        resolve()

        return
      }

      win.requestAnimationFrame(() => win.requestAnimationFrame(() => resolve()))
    })

  /**
   * Dress the page for one crop: the draft's own marker, nothing else.
   *
   * Saved pins live in the page, so a neighbour's marker lands inside this
   * crop whenever the two elements sit within the crop padding of each other —
   * a comment on a heading came back carrying the marker of the comment on the
   * paragraph below it, and "Image 2 marks the target in blue" then pointed at
   * a 1. Hover chrome is transient but would be captured just the same.
   */
  const beginCapture = async (): Promise<boolean> => {
    if (!host || !host.isConnected) {
      return false
    }

    if (pinsLayer) {
      style(pinsLayer, { display: 'none' })
    }

    if (hoverBox) {
      style(hoverBox, { display: 'none' })
    }

    await afterPaint()

    return true
  }

  const endCapture = () => {
    if (pinsLayer) {
      style(pinsLayer, { display: 'block' })
    }
  }

  const teardown = () => {
    unbind()
    emit({ type: 'end' })
    host?.remove()
    host = null
    shadow = null
    hoverBox = null
    draftBox = null
    marquee = null
    pinsLayer = null
    markupLayer = null
    markupTool = null
    marking = null
    markupStrokes.length = 0
    drag = null
    liveDraft = null
    livePins = []
    waiting = null
    queued.length = 0
  }

  return {
    beginCapture,
    beginMarkup,
    clearMarkup,
    endCapture,
    endMarkup,
    getMarkup: () =>
      markupStrokes.map(stroke => ({
        points: stroke.points.map(point => ({ x: point.x, y: point.y })),
        tool: stroke.tool
      })),
    getMarkerNumbers: () => {
      if (!shadow) {
        return []
      }

      return Array.from(shadow.querySelectorAll('[data-annotate-pin] [data-annotate-marker]')).map(node =>
        Number(node.getAttribute('data-annotate-marker') || '0')
      )
    },
    getOutlineColor: () => {
      const outline = shadow?.querySelector('[data-annotate-outline], [data-annotate-pin]') as HTMLElement | null

      return outline?.style.getPropertyValue('border-color') || outline?.style.borderColor || blue
    },
    hideDraft,
    install: () => {
      ensure()
      bind()
      ensureLoop()
    },
    isInstalled: () => Boolean(host && host.isConnected),
    relocate,
    showDraft,
    showPins,
    teardown,
    undoMarkup,
    wait: () => {
      if (queued.length) {
        return Promise.resolve(queued.shift() as AnnotatePageEvent)
      }

      return new Promise(resolve => {
        waiting = resolve
      })
    }
  }
}

/** Source for `executeJavaScript` — the guest page has no module graph. */
export function annotateInPageSource(): string {
  return `(${annotateInPage.toString()})(document)`
}
