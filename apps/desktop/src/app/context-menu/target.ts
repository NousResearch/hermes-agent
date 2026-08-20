/**
 * What a right-click landed on, resolved from the DOM.
 *
 * One resolver so every surface agrees on ownership. Order encodes priority:
 * an editable wins over the link wrapping it (the caret is where the user is
 * working), a link wins over the image inside it for the LINK section — the
 * image section still appears because the target carries both.
 */

export interface ContextMenuDomTarget {
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
}

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

export function resolveDomTarget(element: Element | null): ContextMenuDomTarget {
  const anchor = element?.closest('a[href]')
  const image = element?.closest('img')
  const linkUrl = anchor?.getAttribute('href')?.trim() ?? ''

  return {
    editable: editableFrom(element),
    // A placeholder anchor is not a link the menu can act on.
    linkUrl: linkUrl === '#' ? '' : linkUrl,
    imageUrl: image instanceof HTMLImageElement ? image.currentSrc || image.src : '',
    onImage: Boolean(image),
    selectionText: window.getSelection()?.toString().trim() ?? ''
  }
}

/** True when `url` is something the in-app browser can render. */
export function isWebUrl(url: string): boolean {
  return /^https?:\/\//i.test(url)
}
