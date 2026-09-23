/**
 * What a right-click landed on, resolved from the DOM.
 *
 * One resolver so every surface agrees on ownership. Order encodes priority:
 * an editable wins over the link wrapping it (the caret is where the user is
 * working), a link wins over the image inside it for the LINK section — the
 * image section still appears because the target carries both.
 */

export interface ContextMenuDomTarget {
  /** The enclosing dialog content node, when the click landed inside one. */
  dialogPortalContainer: HTMLElement | null
  /** The clicked editable, when the click landed in one. */
  editable: HTMLElement | null
  /** `href` of the enclosing anchor, as written (never absolutized). */
  linkUrl: string
  /** Source URL of the clicked image, when the click landed on one. */
  imageUrl: string
  /** True when the click landed on an `<img>` (imageUrl may still be empty
   *  for a broken image; Copy image works through coordinates either way). */
  onImage: boolean
  /** The live selection's text at the moment of the click. */
  selectionText: string
  /** The filesystem path the gesture landed on: the selection when the whole
   *  selection is one path, else the token under the caret. Empty when the
   *  click was not on a path (or the token's shape is unknowable without a
   *  filesystem — existence is checked by the reveal itself). */
  path: string
}

/** Path-shaped tokens in running prose: `~/x`, `/Users/…`, `C:\…`. The
 *  lookbehind is what keeps URLs out — the `/` in `https://…` follows a `:`,
 *  and a path never starts mid-word — so `https://x.com/a` yields nothing
 *  instead of a bogus `//x.com/a`. Brackets and CJK punctuation end a token,
 *  which is how paths get written inside parentheses or followed by `，`. */
const PATH_TOKEN_RE = /(?<![\w:.@/\\-])(?:~\/|\/|[a-z]:[\\/])[^\s"'`<>|()[\]{}，。；：、）】]*/gi
/** A whole-selection path: the same shape, anchored to both ends. */
const PATH_ONLY_RE = /^(?:~\/|\/|[a-z]:[\\/])[^\s"'`<>|()[\]{}，。；：、）】]*$/
/** Sentence punctuation that abuts a path without belonging to it. */
const PATH_TRAILING_RE = /[.,;:!?]+$/
/** `report.py:42:9` names a line in the path, not a path. */
const PATH_LINE_REF_RE = /^(.*\.[A-Za-z0-9_]+):\d+(?::\d+)?$/

/** Form fields and `contenteditable` hosts. Mirrors the keybind helper, but
 *  returns the element so the menu can act on it. */
function editableFrom(element: Element | null): HTMLElement | null {
  if (!element) {
    return null
  }

  if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
    return element.disabled || element.readOnly ? null : element
  }

  const host = element.closest('[contenteditable]')

  return host instanceof HTMLElement && host.isContentEditable ? host : null
}

/** Strip what abuts a path in prose without being part of it, and fold a line
 *  reference back to the file it names. */
export function cleanPathToken(raw: string): string {
  const trimmed = raw.replace(PATH_TRAILING_RE, '')
  const lineRef = PATH_LINE_REF_RE.exec(trimmed)

  return lineRef ? lineRef[1] : trimmed
}

/** The path token containing `offset` in `text`, or ''. The offset may sit one
 *  past the token's end: clicking the last character puts the caret there. */
export function pathTokenAt(text: string, offset: number): string {
  for (const match of text.matchAll(PATH_TOKEN_RE)) {
    const start = match.index ?? 0

    if (offset >= start && offset <= start + match[0].length) {
      const token = cleanPathToken(match[0])

      // `C:` and `/` alone are punctuation, not a path to reveal.
      return token.length > 1 ? token : ''
    }
  }

  return ''
}

/** A selection counts as a path only when the WHOLE selection is one — a
 *  sentence the user swept up is prose, not a target. */
export function pathFromSelection(text: string): string {
  const trimmed = text.trim()
  const token = trimmed && !/\s/.test(trimmed) && PATH_ONLY_RE.test(trimmed) ? cleanPathToken(trimmed) : ''

  return token.length > 1 ? token : ''
}

/** The path the gesture landed on. `caretRangeFromPoint` is the only way to
 *  learn where inside a TEXT NODE the click landed — the event's target is the
 *  whole paragraph. Absent (jsdom, older engines) means no caret knowledge, so
 *  only an exact selection can name a path. */
function pathUnderPoint(point: null | { x: number; y: number }): string {
  if (!point || typeof document.caretRangeFromPoint !== 'function') {
    return ''
  }

  try {
    const range = document.caretRangeFromPoint(point.x, point.y)
    const node = range?.startContainer

    return node && node.nodeType === Node.TEXT_NODE ? pathTokenAt(node.textContent ?? '', range.startOffset) : ''
  } catch {
    return ''
  }
}

export function resolveDomTarget(
  element: Element | null,
  point: null | { x: number; y: number } = null
): ContextMenuDomTarget {
  const anchor = element?.closest('a[href]')
  const dialogContent = element?.closest('[data-slot="dialog-content"]')
  const image = element?.closest('img')
  const linkUrl = anchor?.getAttribute('href')?.trim() ?? ''
  const selectionText = window.getSelection()?.toString().trim() ?? ''

  return {
    dialogPortalContainer: dialogContent instanceof HTMLElement ? dialogContent : null,
    editable: editableFrom(element),
    // A placeholder anchor is not a link the menu can act on.
    linkUrl: linkUrl === '#' ? '' : linkUrl,
    imageUrl: image instanceof HTMLImageElement ? image.currentSrc || image.src : '',
    onImage: Boolean(image),
    selectionText,
    // A selection wins over the caret: it is the deliberate, unambiguous act,
    // and Chromium keeps it alive across the gesture.
    path: pathFromSelection(selectionText) || pathUnderPoint(point)
  }
}

/** True when `url` is something the in-app browser can render. */
export function isWebUrl(url: string): boolean {
  return /^https?:\/\//i.test(url)
}
