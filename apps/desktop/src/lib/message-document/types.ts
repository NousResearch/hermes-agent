/**
 * Flat ordered token list (prose + attachments) for composer compile/render.
 * Wire text (`@file:\`path\``) is compiled output — not the editing format.
 */

export const DOCUMENT_VERSION = 1 as const

export type MessageDocument = Token[]

export type Token = TextToken | AttachmentToken

export interface TextToken {
  type: 'text'
  /** UTF-16 string segment; may be empty only between attachments during editing. */
  value: string
}

export interface AttachmentToken {
  type: 'attachment'
  /** Stable id for sync/rewrite after file.attach (survives path changes). */
  id: string
  kind: 'file' | 'folder' | 'image'
  path: string
  displayName: string
  previewUrl?: string
}

export function isTextToken(token: Token): token is TextToken {
  return token.type === 'text'
}

export function isAttachmentToken(token: Token): token is AttachmentToken {
  return token.type === 'attachment'
}
