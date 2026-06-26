export const THREAD_SELECTION_VIEWPORT_SELECTOR = '[data-slot="aui_thread-viewport"]'

export interface BrowserSelectionMatch {
  range: Range
  selection: Selection
  text: string
  viewport: HTMLElement
}

function rangeNode(range: Range): Node | null {
  const node = range.commonAncestorContainer

  return node.nodeType === Node.ELEMENT_NODE ? node : node.parentNode
}

export function selectionIntersectsElement(selection: Selection | null, element: Element | null): boolean {
  if (!selection || !element || selection.isCollapsed || !selection.rangeCount) {
    return false
  }

  for (let index = 0; index < selection.rangeCount; index += 1) {
    const node = rangeNode(selection.getRangeAt(index))

    if (node && element.contains(node)) {
      return true
    }
  }

  return false
}

export function selectionIntersectsSelector(selection: Selection | null, selector: string): boolean {
  return selectionIntersectsElement(selection, document.querySelector(selector))
}

export function currentThreadSelection(): BrowserSelectionMatch | null {
  const selection = window.getSelection()
  const viewport = document.querySelector<HTMLElement>(THREAD_SELECTION_VIEWPORT_SELECTOR)

  if (!selection || !viewport || selection.isCollapsed || !selection.rangeCount) {
    return null
  }

  const text = selection.toString().trim()

  if (!text || !selectionIntersectsElement(selection, viewport)) {
    return null
  }

  return {
    range: selection.getRangeAt(0),
    selection,
    text,
    viewport
  }
}
