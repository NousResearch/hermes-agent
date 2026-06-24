/**
 * Legacy path prepended pills to gateway text. When the composer flag is on,
 * represent pills as leading attachment tokens when not already inline.
 */
import type { ComposerAttachment } from '@/store/composer'

import { createTokenId } from './ids'
import { attachmentPathsEquivalent } from './paths'
import { type AttachmentToken, type MessageDocument, isAttachmentToken } from './types'
import { normalizeDocument } from './validate'

const PILL_KINDS = new Set<ComposerAttachment['kind']>(['file', 'folder', 'image'])

function attachmentTokenFromPill(attachment: ComposerAttachment): AttachmentToken | null {
  if (!PILL_KINDS.has(attachment.kind)) {
    return null
  }

  const path = attachment.path || attachment.refText?.replace(/^@(file|folder|image):/, '') || ''

  if (!path) {
    return null
  }

  return {
    type: 'attachment',
    id: attachment.id || createTokenId('attachment'),
    kind: attachment.kind as AttachmentToken['kind'],
    path,
    displayName: attachment.label || path.split(/[\\/]/).filter(Boolean).pop() || path,
    ...(attachment.previewUrl ? { previewUrl: attachment.previewUrl } : {})
  }
}

function findEquivalentInlineToken(
  document: MessageDocument,
  token: AttachmentToken,
  cwd: string
): AttachmentToken | null {
  for (const inline of document) {
    if (!isAttachmentToken(inline)) {
      continue
    }

    if (inline.kind !== token.kind) {
      continue
    }

    if (attachmentPathsEquivalent(inline.path, token.path, cwd)) {
      return inline
    }
  }

  return null
}

function upgradeInlineTokens(
  document: MessageDocument,
  canonical: AttachmentToken,
  cwd: string
): MessageDocument {
  return document.map(token => {
    if (!isAttachmentToken(token)) {
      return token
    }

    if (token.kind !== canonical.kind || !attachmentPathsEquivalent(token.path, canonical.path, cwd)) {
      return token
    }

    return {
      ...token,
      path: canonical.path,
      displayName: canonical.displayName || token.displayName,
      ...(canonical.previewUrl ? { previewUrl: canonical.previewUrl } : {})
    }
  })
}

export function mergeAttachmentPillsIntoDocument(
  document: MessageDocument,
  attachments: ComposerAttachment[],
  cwd = ''
): MessageDocument {
  let upgradedDoc = document
  const leading: AttachmentToken[] = []

  for (const attachment of attachments) {
    const token = attachmentTokenFromPill(attachment)

    if (!token) {
      continue
    }

    if (findEquivalentInlineToken(upgradedDoc, token, cwd)) {
      upgradedDoc = upgradeInlineTokens(upgradedDoc, token, cwd)
      continue
    }

    const alreadyLeading = leading.some(
      leadingToken => leadingToken.kind === token.kind && attachmentPathsEquivalent(leadingToken.path, token.path, cwd)
    )

    if (alreadyLeading) {
      continue
    }

    leading.push(token)
  }

  if (leading.length === 0) {
    return normalizeDocument(upgradedDoc, cwd)
  }

  return normalizeDocument([...leading, ...upgradedDoc], cwd)
}
