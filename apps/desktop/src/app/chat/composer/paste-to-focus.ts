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

import { requestComposerAttachImages, requestComposerAttachPaths, requestComposerFocus, requestComposerInsert } from './focus'
import { pathifyRefs } from './path-refs'
import { extractClipboardImageBlobs, localPathFromFileUrl, pastedFileClipboardPaths } from './text-utils'
import { linkifyUrls } from './url-refs'

/** Route clipboard contents to the active composer. True when it carried
 *  something a composer can take (the caller should swallow the event). */
export function routeClipboardToComposer(clipboard: DataTransfer): boolean {
  const blobs = extractClipboardImageBlobs(clipboard)
  const text = sanitizeComposerInput(clipboard.getData('text').trim())
  // A file copied from the OS file manager pastes as file:// URL text only —
  // decode it here and route the absolute path to the composer, where it rides
  // the same upload/staging pipeline as an OS drop. Consumed so the URL text
  // is not ALSO inserted as a message.
  const filePaths = pastedFileClipboardPaths(clipboard)

  if (blobs.length > 0) {
    requestComposerAttachImages(blobs)
  }

  if (filePaths.length > 0) {
    requestComposerAttachPaths(filePaths)
  }

  // A bare `data:` URL IS the image attached above — not text to insert.
  // A pure file:// copy likewise becomes an attachment, not text.
  const residualText =
    filePaths.length === 0
      ? text
      : sanitizeComposerInput(
          clipboard
            .getData('text')
            .split(/\r?\n/)
            .filter(line => localPathFromFileUrl(line) === null)
            .join('\n')
            .trim()
        )

  if (residualText && !DATA_IMAGE_URL_RE.test(residualText)) {
    // Same chipping the focused paste path applies: links land as `@url:`
    // chips, bare `@path` tokens promote. The insert focuses the composer.
    requestComposerInsert(pathifyRefs(linkifyUrls(residualText)), { mode: 'inline' })

    return true
  }

  if (blobs.length > 0 || filePaths.length > 0) {
    // Image-only / file-only paste: pull focus so the attach lands somewhere visible.
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
