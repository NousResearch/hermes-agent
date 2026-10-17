import { attachmentReferenceMatches, attachmentReferenceName } from '@/app/chat/composer/attachment-references'
import type { ComposerAttachment } from '@/store/composer'

/** Keep positional aliases meaningful when staging changes a gateway filename. */
export function attachmentContextRef(attachment: ComposerAttachment, text: string) {
  const name = attachmentReferenceName(attachment)

  if (!name || !attachmentReferenceMatches(text, [name]).length) {
    return attachment.refText
  }

  const destination =
    attachment.refText || (attachment.kind === 'image' && attachment.path ? JSON.stringify(attachment.path) : undefined)

  return destination ? `[${name}] = ${destination}` : undefined
}
