/**
 * What a right-click landed on, resolved from the DOM.
 *
 * One resolver so every surface agrees on ownership. Order encodes priority:
 * an editable wins over the link wrapping it (the caret is where the user is
 * working), a link wins over the image inside it for the LINK section — the
 * image section still appears because the target carries both.
 */

import { captureComposerMenuTarget, type ComposerMenuTarget } from './composer-menu'

export interface ContextMenuDomTarget {
  /** Set only when the click landed in a composer: what the composer
   *  context-menu area may see and act on (see `composer-menu.ts`). */
  composer: ComposerMenuTarget | null
  /** The enclosing dialog content node, when the click landed inside one. */
  dialogPortalContainer: HTMLElement | null
  /** The clicked editable, when the click landed in one. */
  editable: HTMLElement | null
  /** Selection or insertion point inside a contenteditable before the menu takes focus. */
  editableRange: Range | null
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
  const dialogContent = element?.closest('[data-slot="dialog-content"]')
  const image = element?.closest('img')
  const linkUrl = anchor?.getAttribute('href')?.trim() ?? ''
  const editable = editableFrom(element)
  const selection = window.getSelection()
  const range = editable?.isContentEditable && selection?.rangeCount ? selection.getRangeAt(0) : null
  const editableRange = range && editable?.contains(range.commonAncestorContainer) ? range.cloneRange() : null

  return {
    composer: captureComposerMenuTarget(editable, editableRange),
    dialogPortalContainer: dialogContent instanceof HTMLElement ? dialogContent : null,
    editable,
    editableRange,
    // A placeholder anchor is not a link the menu can act on.
    linkUrl: linkUrl === '#' ? '' : linkUrl,
    imageUrl: image instanceof HTMLImageElement ? image.currentSrc || image.src : '',
    onImage: Boolean(image),
    selectionText: selection?.toString().trim() ?? ''
  }
}

/** True when `url` is something the in-app browser can render. */
export function isWebUrl(url: string): boolean {
  return /^https?:\/\//i.test(url)
}
