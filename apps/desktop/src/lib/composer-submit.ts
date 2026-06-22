/** Submit-time ref normalization and inline OS image routing for send fidelity. */
import { quoteRefValue, unquoteRef } from '@/app/chat/composer/rich-editor'
import { attachmentId } from '@/lib/chat-runtime'
import type { ComposerAttachment } from '@/store/composer'

/** Matches inline Hermes refs; tolerates optional whitespace after the colon. */
export const COMPOSER_INLINE_REF_RE =
  /@(file|folder|url|image|tool|line|terminal|session):\s*((?:`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+))/g

const ABSOLUTE_OS_PATH_RE = /^([A-Za-z]:[\\/]|\/)/

const IMAGE_EXT_RE = /\.(avif|bmp|gif|heic|heif|jpe?g|png|webp)$/i

export function isAbsoluteOsPath(path: string): boolean {
  return ABSOLUTE_OS_PATH_RE.test(path.trim())
}

function inlineRefPath(kind: string, rawValue: string): string {
  return unquoteRef(rawValue.trim())
}

function isInlineOsImageRef(kind: string, path: string): boolean {
  if (!isAbsoluteOsPath(path)) {
    return false
  }

  if (kind === 'image') {
    return true
  }

  return kind === 'file' && IMAGE_EXT_RE.test(path)
}

/**
 * Inline absolute OS image paths must go through image.attach — the gateway
 * cannot read the local filesystem path from prompt text (especially remote).
 */
export function collectInlineOsImageAttachments(
  text: string,
  existing: ComposerAttachment[]
): ComposerAttachment[] {
  const knownPaths = new Set(existing.map(a => a.path).filter(Boolean))
  const extra: ComposerAttachment[] = []

  for (const match of text.matchAll(COMPOSER_INLINE_REF_RE)) {
    const kind = match[1] || 'file'
    const path = inlineRefPath(kind, match[2] || '')

    if (!path || knownPaths.has(path) || !isInlineOsImageRef(kind, path)) {
      continue
    }

    knownPaths.add(path)
    const label = path.split(/[\\/]/).filter(Boolean).pop() || path

    extra.push({
      id: attachmentId('image', path),
      kind: 'image',
      label,
      path
    })
  }

  return extra
}

/** Remove inline @image/@file refs for paths that were uploaded via image.attach. */
export function stripInlineImageRefs(text: string, attachedPaths: Set<string>): string {
  if (!text || attachedPaths.size === 0) {
    return text
  }

  let next = text

  for (const match of text.matchAll(COMPOSER_INLINE_REF_RE)) {
    const kind = match[1] || 'file'
    const path = inlineRefPath(kind, match[2] || '')

    if (!path || !attachedPaths.has(path)) {
      continue
    }

    next = next.replace(match[0], '')
  }

  return next.replace(/[ \t]{2,}/g, ' ').replace(/\n{3,}/g, '\n\n').trim()
}

/**
 * Fix `@file: path` (space after colon, unquoted path) to `@file:\`path\``.
 * Only rewrites the path token — trailing prose stays intact.
 */
export function normalizeInlineRefWireForm(text: string): string {
  return text.replace(
    /@(file|folder|url|image|tool|line|terminal|session):\s+(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)/g,
    (full, kind: string, value: string) => {
      if (value.startsWith('`') || value.startsWith('"') || value.startsWith("'")) {
        return `@${kind}:${value}`
      }

      return `@${kind}:${quoteRefValue(value)}`
    }
  )
}

/** Keep thumbnail data URLs in the attachment row; omit refs already in bubble body text. */
export function bubbleAttachmentRefsForRow(refs: string[], partsText: string): string[] {
  const body = partsText.trim()

  if (!body) {
    return refs
  }

  return refs.filter(ref => {
    if (ref.startsWith('data:')) {
      return true
    }

    if (body.includes(ref)) {
      return false
    }

    const unquoted = ref.replace(/^@(file|folder|url|image|tool|line|terminal|session):/, '')
    const bare = unquoted.replace(/^[`"']|[`"']$/g, '')

    return !body.includes(bare)
  })
}
