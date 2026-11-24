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

import {
  requestComposerAttachImages,
  requestComposerAttachPastedText,
  requestComposerFocus,
  requestComposerInsert
} from './focus'
import { shouldConvertPasteToAttachment } from './large-paste'
import { pathifyRefs } from './path-refs'
import { extractClipboardImageBlobs } from './text-utils'
import { linkifyUrls } from './url-refs'

/** Inline insertion for text that reached here (short pastes, and the fallback
 *  when a large paste could not be attached). Links land as `@url:` chips and
 *  bare `@path` tokens promote, exactly like a focused paste. */
function insertInline(text: string) {
  requestComposerInsert(pathifyRefs(linkifyUrls(text)), { mode: 'inline' })
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
    // One policy with the focused paste handler (ChatBar's own `handlePaste`):
    // past the large-paste threshold the payload becomes a `.txt` attachment
    // chip instead of composer text. The unfocused path used to insert it
    // raw, so a copied whole reply pasted onto the transcript became the
    // draft — text the user never typed (#1090-class surprise prompt).
    if (shouldConvertPasteToAttachment(text)) {
      void requestComposerAttachPastedText(text).then(attached => {
        // No surface took it (no bridge, messaging chat, write failure): keep
        // the paste rather than dropping it, as the focused path does.
        if (!attached) {
          insertInline(text)
        }
      })
    } else {
      insertInline(text)
    }

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
