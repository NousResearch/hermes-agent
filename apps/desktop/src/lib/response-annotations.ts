/** Self-contained response references: the draft, queue and saved message all
 * carry the same data. No local-storage lookup is required to recover a reply. */
export interface ResponseAnnotation {
  id: string
  text: string
  comment: string
}

export interface ResponseAnnotations {
  type: 'response-annotations'
  version: 1
  label: string
  annotations: ResponseAnnotation[]
}

export interface ResponseAnnotationOpen {
  value: string
  editable: boolean
  remove?: boolean
  anchor?: { left: number; top: number; bottom: number; width: number }
}

const OPEN_EVENT = 'hermes:response-annotations-open'
const MAX_WIRE_CHARS = 512_000

/** Capture at the native boundary so plugins never depend on transcript markup. */
export function captureResponseSelection(): { text: string; range: Range; anchor: DOMRect } | null {
  const selection = window.getSelection()

  if (!selection?.rangeCount || selection.isCollapsed) {return null}
  const range = selection.getRangeAt(0).cloneRange()
  const ancestor = range.commonAncestorContainer
  const element = ancestor instanceof Element ? ancestor : ancestor.parentElement
  const response = element?.closest('[data-slot="aui_assistant-message-root"]')

  if (!response || !response.contains(range.startContainer) || !response.contains(range.endContainer)) {return null}

  if (element?.closest('input, textarea, [contenteditable="true"]')) {return null}
  const text = selection.toString()

  if (!text.trim() || text.length > 64_000) {return null}

  return { text, range, anchor: range.getBoundingClientRect() }
}

export function parseResponseAnnotations(value: string): ResponseAnnotations | null {
  if (value.length > MAX_WIRE_CHARS) {
    return null
  }

  try {
    const data = JSON.parse(value) as ResponseAnnotations

    if (
      data?.type !== 'response-annotations' ||
      data.version !== 1 ||
      typeof data.label !== 'string' ||
      data.label.length > 80 ||
      !Array.isArray(data.annotations) ||
      !data.annotations.length ||
      data.annotations.length > 100
    ) {
      return null
    }

    const ids = new Set<string>()

    for (const item of data.annotations) {
      if (
        !item ||
        typeof item.id !== 'string' ||
        !item.id ||
        ids.has(item.id) ||
        typeof item.text !== 'string' ||
        !item.text ||
        item.text.length > 64_000 ||
        typeof item.comment !== 'string' ||
        item.comment.length > 64_000
      ) {
        return null
      }

      ids.add(item.id)
    }

    return data
  } catch {
    return null
  }
}

export function formatResponseAnnotations(data: ResponseAnnotations): string {
  const value = JSON.stringify(data)
    // Keep a single reference token and prevent selected @file:/slash content
    // from becoming a second executable reference in another surface.
    .replace(/[`@<>/]/g, char => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`)

  if (!parseResponseAnnotations(value)) {
    throw new Error('Invalid response annotations')
  }

  return `@annotation:\`${value}\``
}

export function openResponseAnnotations(request: ResponseAnnotationOpen): void {
  if (parseResponseAnnotations(request.value)) {
    window.dispatchEvent(new CustomEvent(OPEN_EVENT, { detail: request }))
  }
}

export function onResponseAnnotationsOpen(listener: (request: ResponseAnnotationOpen) => void): () => void {
  const receive = (event: Event) => listener((event as CustomEvent<ResponseAnnotationOpen>).detail)
  window.addEventListener(OPEN_EVENT, receive)

  return () => window.removeEventListener(OPEN_EVENT, receive)
}
