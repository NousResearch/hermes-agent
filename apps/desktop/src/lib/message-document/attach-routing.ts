/**
 * Routes attachment tokens that must be staged through the gateway before send.
 * Absolute OS image paths cannot be read from prompt text by a remote gateway,
 * so they go through `image.attach`; `updateTokenPath` rewrites the token path
 * (by stable id) once the gateway returns the staged workspace path.
 */
import { type AttachmentToken, type MessageDocument, isAttachmentToken } from './types'

const ABSOLUTE_OS_PATH_RE = /^([A-Za-z]:[\\/]|\/)/

function isAbsoluteOsPath(path: string): boolean {
  return ABSOLUTE_OS_PATH_RE.test(path.trim())
}

export function collectAttachCandidates(document: MessageDocument): AttachmentToken[] {
  return document.filter(
    (token): token is AttachmentToken =>
      isAttachmentToken(token) && token.kind === 'image' && isAbsoluteOsPath(token.path)
  )
}

export function updateTokenPath(document: MessageDocument, tokenId: string, newPath: string): MessageDocument {
  return document.map(token =>
    isAttachmentToken(token) && token.id === tokenId ? { ...token, path: newPath } : token
  )
}
