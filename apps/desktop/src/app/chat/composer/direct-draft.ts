import type { ComposerAttachment } from '@/store/composer'

import type { RelayAttachmentTarget, RelayedAttachment } from './attachment-relay'

const MAX_EXACT_SOURCE_CHARS = 64_000
const HEX_DIGEST = /^[0-9a-f]{64}$/

export interface DirectDraft {
  text: string
  attachments?: ComposerAttachment[]
}

export interface RelayedDirectDraft {
  text: string
  attachments?: RelayedAttachment[]
}

export interface DirectDraftTarget extends RelayAttachmentTarget {
  composerScope: string
  composerTarget: string
}

export interface DirectDraftTargetQuery {
  connectionId: string
  profile: string
  runtimeSessionId: string
  storedSessionId: string
}

export interface DirectDraftAttachmentManifestEntry {
  id: string
  kind: 'file' | 'image'
  mediaType: string
  name: string
  occurrenceId: null | string
  order: number
  refText?: string
  runtimeSessionId: string
  sha256: string
  size: number
  sourceId: string
  storedName: string
}

export interface DirectDraftAdmission {
  attachmentManifest: DirectDraftAttachmentManifestEntry[]
  contextText: string
  payloadDigest: string
  sourceText: string
}

export type DirectDraftReceipt =
  | { state: 'acknowledgement_uncertain'; submissionId: string; payloadDigest: string }
  | { state: 'durably_accepted'; submissionId: string; payloadDigest: string }
  | {
      state: 'rejected'
      submissionId: string
      reason:
        | 'busy-target'
        | 'invalid-draft'
        | 'invalid-submission-id'
        | 'stale-target'
        | 'target-unavailable'
        | 'unauthorized-attachment'
        | 'unsupported-capability'
    }

export interface DirectDraftSubmitOptions {
  submissionId?: string
}

function attachmentManifest(attachments: RelayedAttachment[]): DirectDraftAttachmentManifestEntry[] {
  return attachments.map((attachment, order) => ({
    id: attachment.id,
    kind: attachment.kind,
    mediaType: attachment.mediaType,
    name: attachment.name,
    occurrenceId: attachment.occurrenceId,
    order,
    ...(attachment.refText ? { refText: attachment.refText } : {}),
    runtimeSessionId: attachment.runtimeSessionId,
    sha256: attachment.sha256,
    size: attachment.size,
    sourceId: attachment.provenance.sourceId,
    storedName: attachment.storedName
  }))
}

function validAttachment(attachment: RelayedAttachment, order: number): boolean {
  return (
    attachment.order === order &&
    attachment.provenance.sourceId === attachment.id &&
    attachment.size > 0 &&
    Number.isSafeInteger(attachment.size) &&
    HEX_DIGEST.test(attachment.sha256) &&
    Boolean(attachment.runtimeSessionId) &&
    Boolean(attachment.name) &&
    Boolean(attachment.mediaType) &&
    Boolean(attachment.storedName)
  )
}

export function directDraftIsValid(draft: RelayedDirectDraft): boolean {
  if (!draft || typeof draft.text !== 'string' || draft.text.length > MAX_EXACT_SOURCE_CHARS) {
    return false
  }

  const attachments = draft.attachments ?? []

  return (
    (draft.text.length > 0 || attachments.length > 0) &&
    attachments.length <= 32 &&
    attachments.every(validAttachment)
  )
}

async function digestJson(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(value))
  const digest = await crypto.subtle.digest('SHA-256', bytes)

  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')
}

export async function createDirectDraftAdmission(draft: RelayedDirectDraft): Promise<DirectDraftAdmission> {
  if (!directDraftIsValid(draft)) {
    throw new Error('invalid exact direct draft')
  }

  const attachments = attachmentManifest(draft.attachments ?? [])
  const refs = attachments.flatMap(attachment => (attachment.refText ? [attachment.refText] : []))
  const contextText = refs.length
    ? `${refs.join('\n')}\n\n${draft.text || 'What do you see in this image?'}`
    : draft.text || 'What do you see in this image?'
  const payloadDigest = await digestJson({ text: draft.text, contextText, attachments })

  return { attachmentManifest: attachments, contextText, payloadDigest, sourceText: draft.text }
}
