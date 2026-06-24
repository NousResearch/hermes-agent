/**
 * Parse wire text into a MessageDocument. Used for legacy/history messages and
 * when the composer document atom is empty. Only file/folder/image become
 * attachment tokens; other ref kinds stay in text for round-trip. Tolerates
 * whitespace after the colon so malformed `@file: path` imports cleanly.
 */
import { unquoteRef } from '@/app/chat/composer/rich-editor'

import { createTokenId } from './ids'
import { type AttachmentToken, type MessageDocument, type Token } from './types'

const IMPORT_REF_RE =
  /@(file|folder|url|image|tool|line|terminal|session):\s*((?:`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+))/g

const ATTACHMENT_KINDS = new Set<string>(['file', 'folder', 'image'])

function displayNameForPath(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() || path
}

export function importFromWireText(text: string): MessageDocument {
  const tokens: Token[] = []
  let cursor = 0

  IMPORT_REF_RE.lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = IMPORT_REF_RE.exec(text)) !== null) {
    const kind = match[1]

    // Non-attachment kinds stay as text — leave the cursor so the surrounding
    // slice swallows the verbatim ref on the next attachment or at the end.
    if (!ATTACHMENT_KINDS.has(kind)) {
      continue
    }

    const start = match.index

    if (start > cursor) {
      tokens.push({ type: 'text', value: text.slice(cursor, start) })
    }

    const path = unquoteRef((match[2] || '').trim())

    tokens.push({
      type: 'attachment',
      id: createTokenId('attachment'),
      kind: kind as AttachmentToken['kind'],
      path,
      displayName: displayNameForPath(path)
    })

    cursor = start + match[0].length
  }

  if (cursor < text.length) {
    tokens.push({ type: 'text', value: text.slice(cursor) })
  }

  return tokens
}
