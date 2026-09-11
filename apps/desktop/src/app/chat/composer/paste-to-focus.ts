/**
 * Paste-to-focus: the paste twin of type-to-focus. A ⌘V/Ctrl+V landing on
 * non-editable chrome (the transcript, empty chat, the window body) routes the
 * clipboard into the active composer instead of dying on the body — images
 * attach, text inserts (links/paths chipping exactly like a focused paste),
 * and the composer takes focus so the user keeps typing.
 *
 * Clipboard data is only readable synchronously inside the paste event, so the
 * window listener extracts everything here and the payload rides the composer
 * bus. Editable targets keep their own paste handlers, and the same surfaces
 * that block type-to-focus (dialogs, menus, terminal, full pages) block this.
 */

import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { isEditableTarget } from '@/lib/keybinds/combo'
import { composerFocusBlockedBySurface } from '@/lib/keybinds/composer-focus-keys'
import { MAX_COMPOSER_TEXT_CHARS } from '@/store/composer'

import { requestComposerAttachImages, requestComposerFocus, requestComposerInsert } from './focus'
import { pathifyRefs } from './path-refs'
import { extractClipboardImageBlobs } from './text-utils'
import { linkifyUrls } from './url-refs'

/** Prepare pasted text for composer insertion (#98562): links/paths chip as
 *  usual, but a paste over MAX_COMPOSER_TEXT_CHARS is clamped to a plain-text
 *  prefix — linkify/pathify/DOM-chipping a multi-megabyte blob synchronously
 *  on the UI thread is what froze the app. Both paste entry points (the
 *  focused editor's handlePaste and the window-level route below) share this
 *  so neither can bypass the ceiling. */
export function preparePastedText(text: string): string {
  if (text.length > MAX_COMPOSER_TEXT_CHARS) {
    return text.slice(0, MAX_COMPOSER_TEXT_CHARS)
  }

  return pathifyRefs(linkifyUrls(text))
}

/** Route clipboard contents to the active composer. True when it carried
 *  something a composer can take (the caller should swallow the event). */
export function routeClipboardToComposer(clipboard: DataTransfer): boolean {
  const blobs = extractClipboardImageBlobs(clipboard)
  const text = sanitizeComposerInput(clipboard.getData('text').trim())

  if (blobs.length > 0) {
    requestComposerAttachImages(blobs)
  }

  // A bare `data:` URL IS the image attached above — not text to insert.
  if (text && !DATA_IMAGE_URL_RE.test(text)) {
    // Same chipping the focused paste path applies: links land as `@url:`
    // chips, bare `@path` tokens promote. The insert focuses the composer.
    requestComposerInsert(preparePastedText(text), { mode: 'inline' })

    return true
  }

  if (blobs.length > 0) {
    // Image-only paste: pull focus so the attach lands somewhere visible.
    requestComposerFocus('active')

    return true
  }

  return false
}

/** The window-level `paste` dispatcher (use-keybinds registers it beside the
 *  keydown listener). Yields to editables — they own their own paste — and to
 *  every surface that owns its keys per type-to-focus. */
export function handleWindowPaste(event: ClipboardEvent) {
  if (
    event.defaultPrevented ||
    !event.clipboardData ||
    isEditableTarget(event.target) ||
    composerFocusBlockedBySurface()
  ) {
    return
  }

  if (routeClipboardToComposer(event.clipboardData)) {
    event.preventDefault()
  }
}
