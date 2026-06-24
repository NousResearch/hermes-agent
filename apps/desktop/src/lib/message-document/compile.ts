/**
 * Document order is wire order (no pill prepend). Paths are always quoted via
 * `quoteRefValue` so the compiler never emits `@kind: value` (space after colon).
 */
import { quoteRefValue } from '@/app/chat/composer/rich-editor'

import { type MessageDocument, isAttachmentToken } from './types'

export function compileToWireText(document: MessageDocument): string {
  let out = ''

  for (const token of document) {
    if (isAttachmentToken(token)) {
      out += `@${token.kind}:${quoteRefValue(token.path)}`
    } else {
      out += token.value
    }
  }

  return out
}
