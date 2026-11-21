// Mirrored by the renderer's text-comment contract. No UI is injected into
// the guest: this isolated preload only reports selections made by the user.
export const PREVIEW_TEXT_SELECTION_CHANNEL = 'preview-text-selection'
export const PREVIEW_TEXT_SELECTION_MODE_CHANNEL = 'preview-text-selection-mode'

interface PreviewTextSelection {
  text: string
  url: string
  viewportWidth: number
  rect: { x: number; y: number; width: number; height: number }
}

interface GuestTextSelectionHost {
  onMode: (listener: (enabled: unknown) => void) => () => void
  sendToHost: (channel: string, payload: PreviewTextSelection | null) => void
}

const PRIVATE_SELECTION = 'input, textarea, select, [contenteditable]:not([contenteditable="false"]), [data-secret]'
const SELECTION_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'])

function selectedText(doc: Document): PreviewTextSelection | null {
  const selection = doc.getSelection()

  if (!selection || selection.isCollapsed || selection.rangeCount !== 1) {
    return null
  }

  const text = selection.toString()

  // Reject rather than truncate: the agent must receive the exact selection.
  if (!text.trim() || text.length > 32_768) {
    return null
  }

  const range = selection.getRangeAt(0)
  const ancestor = range.commonAncestorContainer
  const element = ancestor.nodeType === 1 ? (ancestor as Element) : ancestor.parentElement

  if (!element || element.closest(PRIVATE_SELECTION)) {
    return null
  }

  // Endpoints alone miss a private control in the middle of a broad selection.
  for (const field of element.querySelectorAll(PRIVATE_SELECTION)) {
    if (range.intersectsNode(field)) {
      return null
    }
  }

  const { x, y, width, height } = range.getBoundingClientRect()
  const viewportWidth = doc.defaultView?.innerWidth

  if (
    ![x, y, width, height, viewportWidth].every(Number.isFinite) ||
    width <= 0 ||
    height <= 0 ||
    !viewportWidth ||
    viewportWidth <= 0
  ) {
    return null
  }

  return { text, url: doc.location.href, viewportWidth, rect: { x, y, width, height } }
}

export function installGuestTextSelection(doc: Document, host: GuestTextSelectionHost): () => void {
  let enabled = false
  let showing = false

  const clear = () => {
    if (showing) {
      showing = false
      host.sendToHost(PREVIEW_TEXT_SELECTION_CHANNEL, null)
    }
  }

  const capture = () => {
    const selection = selectedText(doc)

    if (selection) {
      showing = true
      host.sendToHost(PREVIEW_TEXT_SELECTION_CHANNEL, selection)
    } else {
      clear()
    }
  }

  const onMouseDown = (event: MouseEvent) => {
    if (enabled && event.isTrusted) {
      clear()
    }
  }

  const onMouseUp = (event: MouseEvent) => {
    if (enabled && event.isTrusted && event.button === 0) {
      capture()
    }
  }

  const onKeyDown = (event: KeyboardEvent) => {
    if (enabled && event.isTrusted && event.key === 'Escape') {
      clear()
    }
  }

  const onKeyUp = (event: KeyboardEvent) => {
    if (
      enabled &&
      event.isTrusted &&
      ((event.shiftKey && SELECTION_KEYS.has(event.key)) ||
        ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a'))
    ) {
      capture()
    }
  }

  const unsubscribe = host.onMode(value => {
    enabled = value === true

    if (!enabled) {
      clear()
    }
  })

  doc.addEventListener('mousedown', onMouseDown, true)
  doc.addEventListener('mouseup', onMouseUp, true)
  doc.addEventListener('keydown', onKeyDown, true)
  doc.addEventListener('keyup', onKeyUp, true)
  // Cancel stale geometry without interfering with the page's scroll behavior.
  doc.addEventListener('scroll', clear, true)
  doc.defaultView?.addEventListener('resize', clear)

  return () => {
    unsubscribe()
    enabled = false
    clear()
    doc.removeEventListener('mousedown', onMouseDown, true)
    doc.removeEventListener('mouseup', onMouseUp, true)
    doc.removeEventListener('keydown', onKeyDown, true)
    doc.removeEventListener('keyup', onKeyUp, true)
    doc.removeEventListener('scroll', clear, true)
    doc.defaultView?.removeEventListener('resize', clear)
  }
}
