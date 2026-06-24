/**
 * Document invariants: adjacent text tokens are merged and empty text tokens at
 * the document boundaries are dropped (empties between attachments are kept so
 * the caret can sit there during editing). Returns a new array — never mutates.
 */
import { attachmentPathsEquivalent, preferCanonicalAttachmentPath } from './paths'
import { type AttachmentToken, type MessageDocument, type Token, isAttachmentToken, isTextToken } from './types'

/** Drop duplicate file/folder/image tokens with the same kind+path (pill + inline). */
export function dedupeAttachmentTokens(doc: MessageDocument, cwd = ''): MessageDocument {
  const next: Token[] = []

  for (const token of doc) {
    if (!isAttachmentToken(token)) {
      next.push(token)
      continue
    }

    const existingIdx = next.findIndex(
      candidate =>
        isAttachmentToken(candidate) &&
        candidate.kind === token.kind &&
        attachmentPathsEquivalent(candidate.path, token.path, cwd)
    )

    if (existingIdx === -1) {
      next.push({ ...token })
      continue
    }

    const existing = next[existingIdx] as AttachmentToken
    const canonicalPath = preferCanonicalAttachmentPath(existing.path, token.path)

    next[existingIdx] = {
      ...existing,
      path: canonicalPath,
      displayName: token.displayName || existing.displayName,
      ...(token.previewUrl && !existing.previewUrl ? { previewUrl: token.previewUrl } : {})
    }
  }

  return next
}

export function normalizeDocument(doc: MessageDocument, cwd = ''): MessageDocument {
  const merged: Token[] = []

  for (const token of doc) {
    const prev = merged[merged.length - 1]

    if (isTextToken(token) && prev && isTextToken(prev)) {
      merged[merged.length - 1] = { type: 'text', value: prev.value + token.value }
    } else if (isTextToken(token)) {
      merged.push({ type: 'text', value: token.value })
    } else {
      merged.push({ ...token })
    }
  }

  while (merged.length > 0) {
    const first = merged[0]
    if (isTextToken(first) && first.value === '') {
      merged.shift()
    } else {
      break
    }
  }

  while (merged.length > 0) {
    const last = merged[merged.length - 1]
    if (isTextToken(last) && last.value === '') {
      merged.pop()
    } else {
      break
    }
  }

  return dedupeAttachmentTokens(merged, cwd)
}
