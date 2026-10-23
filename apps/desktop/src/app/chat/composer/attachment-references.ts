import type { ComposerAttachment } from '@/store/composer'

// Editor-owned presentation metadata: a repaint or undo uses the same names,
// without sharing attachments between mounted composers.
const namesByEditor = new WeakMap<HTMLElement, readonly string[]>()

export const attachmentNamesForEditor = (editor: HTMLElement) => namesByEditor.get(editor) ?? []
export const setAttachmentNamesForEditor = (editor: HTMLElement, names: readonly string[]) =>
  namesByEditor.set(editor, names)

export function attachmentReferenceName(attachment: ComposerAttachment): string | null {
  return attachment.kind === 'file' || attachment.kind === 'image'
    ? (attachment.referenceName ?? attachment.label).trim() || null
    : null
}

export function attachmentReferenceMatches(text: string, names: readonly string[]) {
  const choices = [...new Set(names.map(name => name.trim()).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .map(name => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))

  if (!choices.length || !text.includes('[')) {
    return []
  }

  const code = [...text.matchAll(/(`+)[\s\S]*?\1/g)].map(match => [match.index, match.index + match[0].length])
  const pattern = new RegExp(`\\[\\s*(?:${choices.join('|')})\\s*\\]`, 'giu')

  return [...text.matchAll(pattern)].flatMap(match => {
    const start = match.index
    const end = start + match[0].length

    if (
      /[!\\\]]/.test(text[start - 1] ?? '') ||
      /^[(:[]/.test(text.slice(end)) ||
      code.some(([a, b]) => start >= a && start < b)
    ) {
      return []
    }

    return [{ start, end, text: match[0] }]
  })
}

export function attachmentReferenceElement(text: string) {
  const chip = document.createElement('span')
  chip.contentEditable = 'false'
  chip.className = 'ref'
  chip.dataset.ref = 'file'
  chip.dataset.refText = text
  chip.dataset.attachmentReference = ''
  chip.textContent = text

  return chip
}
