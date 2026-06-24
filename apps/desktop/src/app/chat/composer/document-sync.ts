/**
 * DOM is still the edit surface — derive MessageDocument from composerPlainText
 * on each input/compositionend until inserts write tokens directly.
 */
import { importFromWireText, type MessageDocument, normalizeDocument } from '@/lib/message-document'

import { composerPlainText } from './rich-editor'

export function syncDocumentFromDom(editor: HTMLElement): MessageDocument {
  return normalizeDocument(importFromWireText(composerPlainText(editor)))
}
