export const PREVIEW_DESIGN_COMMAND_CHANNEL = 'hermes-preview-design-command'
export const PREVIEW_DESIGN_EVENT_CHANNEL = 'hermes-preview-design-event'

export interface DesignLayer {
  segment: string
  tag: string
}

/** Ephemeral selection in the current Browser document; never a source mapping. */
export interface DesignSelection {
  tag: string
  selector: string
  /** DOM-only ancestry; never source ownership. */
  layers: DesignLayer[]
  selectedLayer: number
  rect: { x: number; y: number; width: number; height: number }
  css: Record<string, string>
}

export type DesignPageEvent = { type: 'select'; selection: DesignSelection } | { type: 'end' }

export type PreviewDesignCommand =
  | { type: 'start'; session: number }
  | { type: 'stop'; session: number }
  | { type: 'select-layer'; session: number; index: number }

export type PreviewDesignBridgeEvent =
  | { type: 'ready'; session: number }
  | { type: 'select'; session: number; selection: DesignSelection }
  | { type: 'end'; session: number }
  | { type: 'stopped'; session: number }
  | { type: 'error'; session: number; message: string }

export interface DesignInPage {
  install(): void
  teardown(): void
  wait(): Promise<DesignPageEvent>
  selectLayer(index: number): boolean
}

const CSS_KEYS = ['color', 'background-color', 'font-size', 'font-weight', 'border-radius', 'padding'] as const
const TAG_RE = /^[a-z][a-z0-9-]{0,63}$/
const SEGMENT_RE = /^[a-z][a-z0-9-]{0,63}(?::nth-of-type\([1-9][0-9]{0,5}\))?$/

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const finiteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= 10_000_000

const exactKeys = (value: Record<string, unknown>, allowed: readonly string[]): boolean =>
  Object.keys(value).every(key => allowed.includes(key))

/** Reject guest-originated data that exceeds the narrow DOM-context contract. */
export function parseDesignSelection(value: unknown): DesignSelection | null {
  if (!isRecord(value) || !exactKeys(value, ['tag', 'selector', 'layers', 'selectedLayer', 'rect', 'css'])) {
    return null
  }

  if (typeof value.tag !== 'string' || !TAG_RE.test(value.tag)) {
    return null
  }

  if (typeof value.selector !== 'string' || value.selector.length === 0 || value.selector.length > 512) {
    return null
  }

  if (!Array.isArray(value.layers) || value.layers.length === 0 || value.layers.length > 16) {
    return null
  }

  const layers: DesignLayer[] = []

  for (const layer of value.layers) {
    if (!isRecord(layer) || !exactKeys(layer, ['tag', 'segment'])) {
      return null
    }

    if (
      typeof layer.tag !== 'string' ||
      !TAG_RE.test(layer.tag) ||
      typeof layer.segment !== 'string' ||
      !SEGMENT_RE.test(layer.segment)
    ) {
      return null
    }

    layers.push({ tag: layer.tag, segment: layer.segment })
  }

  if (!Number.isSafeInteger(value.selectedLayer) || Number(value.selectedLayer) < 0 || Number(value.selectedLayer) >= layers.length) {
    return null
  }

  if (!isRecord(value.rect) || !exactKeys(value.rect, ['x', 'y', 'width', 'height'])) {
    return null
  }

  const { x, y, width, height } = value.rect

  if (!finiteNumber(x) || !finiteNumber(y) || !finiteNumber(width) || !finiteNumber(height) || width < 0 || height < 0) {
    return null
  }

  if (!isRecord(value.css) || !exactKeys(value.css, CSS_KEYS)) {
    return null
  }

  const css: Record<string, string> = {}

  for (const [key, raw] of Object.entries(value.css)) {
    if (!CSS_KEYS.includes(key as (typeof CSS_KEYS)[number]) || typeof raw !== 'string' || raw.length > 80) {
      return null
    }

    css[key] = raw
  }

  return {
    css,
    layers,
    rect: { x, y, width, height },
    selectedLayer: Number(value.selectedLayer),
    selector: value.selector,
    tag: value.tag
  }
}

export function parsePreviewDesignCommand(value: unknown): PreviewDesignCommand | null {
  if (!isRecord(value) || !Number.isSafeInteger(value.session) || Number(value.session) <= 0) {
    return null
  }

  if (value.type === 'start' || value.type === 'stop') {
    return exactKeys(value, ['type', 'session']) ? { type: value.type, session: Number(value.session) } : null
  }

  if (
    value.type === 'select-layer' &&
    exactKeys(value, ['type', 'session', 'index']) &&
    Number.isSafeInteger(value.index) &&
    Number(value.index) >= 0 &&
    Number(value.index) < 16
  ) {
    return { type: 'select-layer', session: Number(value.session), index: Number(value.index) }
  }

  return null
}

export function parsePreviewDesignBridgeEvent(value: unknown, expectedSession?: number): PreviewDesignBridgeEvent | null {
  if (
    !isRecord(value) ||
    !Number.isSafeInteger(value.session) ||
    Number(value.session) < 1 ||
    (expectedSession !== undefined && value.session !== expectedSession) ||
    typeof value.type !== 'string'
  ) {
    return null
  }

  const session = Number(value.session)

  if (value.type === 'ready' || value.type === 'end' || value.type === 'stopped') {
    return exactKeys(value, ['type', 'session']) ? { type: value.type, session } : null
  }

  if (value.type === 'select' && exactKeys(value, ['type', 'session', 'selection'])) {
    const selection = parseDesignSelection(value.selection)

    return selection ? { type: 'select', session, selection } : null
  }

  if (
    value.type === 'error' &&
    exactKeys(value, ['type', 'session', 'message']) &&
    typeof value.message === 'string' &&
    value.message.length > 0 &&
    value.message.length <= 240
  ) {
    return { type: 'error', session, message: value.message }
  }

  return null
}

/** Self-contained DOM controller used only from the isolated preview preload. */
export function designInPage(doc: Document): DesignInPage {
  const win = doc.defaultView
  let host: HTMLElement | null = null
  let selected: HTMLElement | null = null
  let picked: Element | null = null
  let layerOrigin: Element | null = null
  let waiting: ((event: DesignPageEvent) => void) | null = null
  const queue: DesignPageEvent[] = []

  const emit = (event: DesignPageEvent) => {
    if (waiting) {
      const resolve = waiting
      waiting = null
      resolve(event)
    } else {
      queue.push(event)
    }
  }

  const place = (box: HTMLElement, target: Element | null) => {
    if (!target || !target.isConnected) {
      box.style.display = 'none'

      return
    }

    const rect = target.getBoundingClientRect()
    box.style.display = 'block'
    box.style.left = `${rect.left}px`
    box.style.top = `${rect.top}px`
    box.style.width = `${rect.width}px`
    box.style.height = `${rect.height}px`
  }

  const relocate = () => {
    if (host && selected) {
      place(selected, picked)
    }
  }

  const selectable = (node: Element | null): node is Element =>
    Boolean(node && node !== host && node !== doc.body && node !== doc.documentElement)

  const hit = (x: number, y: number): Element | null => {
    const node = doc.elementFromPoint(x, y)

    return selectable(node) ? node : null
  }

  const block = (event: Event) => {
    event.preventDefault()
    event.stopImmediatePropagation()
  }

  const onPointerBoundary = (event: Event) => block(event)

  const layerTrail = (): Element[] => {
    const trail: Element[] = []
    let node = layerOrigin

    while (node && node !== doc.documentElement && trail.length < 16) {
      trail.unshift(node)
      node = node.parentElement
    }

    return trail
  }

  const segment = (node: Element): string => {
    const tag = node.tagName.toLowerCase()
    const parent = node.parentElement
    const siblings = parent ? Array.from(parent.children).filter(child => child.tagName === node.tagName) : []

    return tag + (siblings.length > 1 ? `:nth-of-type(${siblings.indexOf(node) + 1})` : '')
  }

  const select = (node: Element) => {
    picked = node
    relocate()
    const rect = node.getBoundingClientRect()
    const parts: string[] = []
    let cursor: Element | null = node

    while (cursor && cursor !== doc.documentElement && parts.length < 6) {
      parts.unshift(segment(cursor))
      cursor = cursor.parentElement
    }

    const css: Record<string, string> = {}
    const computed = win?.getComputedStyle(node)

    for (const key of CSS_KEYS) {
      const value = computed?.getPropertyValue(key).trim() || ''

      if (value && value.length <= 80) {
        css[key] = value
      }
    }

    const layers = layerTrail()
    emit({
      type: 'select',
      selection: {
        css,
        layers: layers.map(layer => ({ segment: segment(layer), tag: layer.tagName.toLowerCase() })),
        rect: { x: rect.left, y: rect.top, width: rect.width, height: rect.height },
        selectedLayer: layers.indexOf(node),
        selector: parts.join(' > '),
        tag: node.tagName.toLowerCase()
      }
    })
  }

  const onClick = (event: MouseEvent) => {
    block(event)

    if (event.button !== 0) {
      return
    }

    const node = hit(event.clientX, event.clientY)

    if (node) {
      layerOrigin = node
      select(node)
    }
  }

  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      block(event)
      teardown()

      return
    }

    if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
      block(event)
      const node = doc.activeElement

      if (selectable(node)) {
        layerOrigin = node
        select(node)
      }
    }
  }

  const onKeyUp = (event: KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
      block(event)
    }
  }

  const pointerBoundaryEvents = [
    'pointerdown',
    'pointerup',
    'pointercancel',
    'mousedown',
    'mouseup',
    'touchstart',
    'touchend',
    'touchcancel',
    'dblclick',
    'auxclick',
    'contextmenu'
  ] as const

  const teardown = () => {
    for (const type of pointerBoundaryEvents) {
      win?.removeEventListener(type, onPointerBoundary, true)
    }

    win?.removeEventListener('click', onClick, true)
    win?.removeEventListener('keydown', onKey, true)
    win?.removeEventListener('keyup', onKeyUp, true)
    win?.removeEventListener('scroll', relocate, true)
    win?.removeEventListener('resize', relocate)
    host?.remove()
    host = selected = null
    picked = layerOrigin = null
    queue.length = 0
    emit({ type: 'end' })
  }

  return {
    install: () => {
      if (host) {
        return
      }

      doc.querySelectorAll('hermes-design').forEach(node => node.remove())
      host = doc.createElement('hermes-design')
      host.setAttribute('aria-hidden', 'true')
      host.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147482999'
      const shadow = host.attachShadow({ mode: 'open' })
      selected = doc.createElement('div')
      selected.setAttribute('data-design-selected', '')
      selected.style.cssText =
        'position:fixed;display:none;pointer-events:none;box-sizing:border-box;border:2px solid #2F80ED;outline:1px solid rgba(255,255,255,.9);outline-offset:1px;background:rgba(47,128,237,.14)'
      shadow.appendChild(selected)
      doc.documentElement.appendChild(host)

      for (const type of pointerBoundaryEvents) {
        win?.addEventListener(type, onPointerBoundary, { capture: true, passive: false })
      }

      win?.addEventListener('click', onClick, true)
      win?.addEventListener('keydown', onKey, true)
      win?.addEventListener('keyup', onKeyUp, true)
      win?.addEventListener('scroll', relocate, true)
      win?.addEventListener('resize', relocate)
    },
    teardown,
    selectLayer: index => {
      if (!host || !layerOrigin?.isConnected || !Number.isSafeInteger(index)) {
        return false
      }

      const node = layerTrail()[index]

      if (!node?.isConnected) {
        return false
      }

      select(node)

      return true
    },
    wait: () =>
      queue.length
        ? Promise.resolve(queue.shift()!)
        : new Promise(resolve => {
            waiting = resolve
          })
  }
}
