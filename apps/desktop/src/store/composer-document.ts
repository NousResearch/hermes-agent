/**
 * Live MessageDocument while the composer flag is on. Memory-only; legacy
 * `$composerDraft` stays authoritative when the flag is off.
 */
import { atom } from 'nanostores'

import {
  type AttachmentToken,
  createTokenId,
  type MessageDocument,
  normalizeDocument,
  type Token
} from '@/lib/message-document'

export const $composerDocument = atom<MessageDocument>([])

export function setComposerDocument(document: MessageDocument): void {
  $composerDocument.set(normalizeDocument(document))
}

export function clearComposerDocument(): void {
  $composerDocument.set([])
}

export function replaceComposerDocument(updater: (document: MessageDocument) => MessageDocument): void {
  $composerDocument.set(normalizeDocument(updater($composerDocument.get())))
}

export function appendTextToken(value: string): void {
  if (!value) {
    return
  }

  replaceComposerDocument(document => [...document, { type: 'text', value }])
}

export function insertAttachmentToken(token: AttachmentToken, index?: number): void {
  replaceComposerDocument(document => {
    const next: Token[] = [...document]
    const at = index === undefined ? next.length : Math.max(0, Math.min(index, next.length))
    next.splice(at, 0, token)

    return next
  })
}

/** Build an AttachmentToken with a fresh id from the minimal fields. */
export function createAttachmentToken(
  fields: Omit<AttachmentToken, 'type' | 'id'> & { id?: string }
): AttachmentToken {
  return {
    type: 'attachment',
    id: fields.id ?? createTokenId('attachment'),
    kind: fields.kind,
    path: fields.path,
    displayName: fields.displayName,
    ...(fields.previewUrl ? { previewUrl: fields.previewUrl } : {})
  }
}
