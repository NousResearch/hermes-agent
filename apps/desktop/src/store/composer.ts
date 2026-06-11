import { atom } from 'nanostores'

import { triggerHaptic } from '@/lib/haptics'

export interface ComposerAttachment {
  id: string
  kind: 'image' | 'file' | 'folder' | 'terminal' | 'url'
  label: string
  detail?: string
  refText?: string
  previewUrl?: string
  path?: string
  attachedSessionId?: string
  /** Set while the file/image bytes are being staged into the session
   * workspace (remote upload or local stage), and 'error' if that failed.
   * Drives the spinner / error state on the composer attachment card. */
  uploadState?: 'uploading' | 'error'
}

export const $composerDraft = atom('')
export const $composerAttachments = atom<ComposerAttachment[]>([])
export const $composerTerminalSelections = atom<Record<string, string>>({})

const COMPOSER_DRAFT_STORAGE_PREFIX = 'hermes:composer-draft:v1:'
const NEW_SESSION_DRAFT_SCOPE = '__new__'

function storageScope(scope: string | null | undefined): string {
  const trimmed = scope?.trim()

  return trimmed || NEW_SESSION_DRAFT_SCOPE
}

function browserStorage(): Storage | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    return window.localStorage
  } catch {
    return null
  }
}

export function composerDraftStorageKey(scope: string | null | undefined): string {
  return `${COMPOSER_DRAFT_STORAGE_PREFIX}${encodeURIComponent(storageScope(scope))}`
}

export function readPersistedComposerDraft(scope: string | null | undefined): string {
  try {
    return browserStorage()?.getItem(composerDraftStorageKey(scope)) ?? ''
  } catch {
    return ''
  }
}

export function writePersistedComposerDraft(scope: string | null | undefined, value: string) {
  try {
    const storage = browserStorage()

    if (!storage) {
      return
    }

    const key = composerDraftStorageKey(scope)

    if (value.length === 0) {
      storage.removeItem(key)
    } else {
      storage.setItem(key, value)
    }
  } catch {
    // Draft persistence is a safety net only; storage quota/private-mode errors
    // must never break typing or submission.
  }
}

export function clearPersistedComposerDraft(scope: string | null | undefined) {
  try {
    browserStorage()?.removeItem(composerDraftStorageKey(scope))
  } catch {
    // Best-effort only.
  }
}

// Attachments can't ride along in localStorage the way text does — they carry
// live blobs, object URLs, and in-flight upload state that don't serialize and
// are tied to the running app. So we retain them per scope in an in-memory map
// instead: a session switch restores the chips you'd staged, even though they
// (unlike text) cannot survive a full app reload.
const composerAttachmentsByScope = new Map<string, ComposerAttachment[]>()

const cloneComposerAttachments = (attachments: ComposerAttachment[]): ComposerAttachment[] =>
  attachments.map(attachment => ({ ...attachment }))

export function stashComposerAttachments(scope: string | null | undefined, attachments: ComposerAttachment[]) {
  const key = storageScope(scope)

  if (attachments.length === 0) {
    composerAttachmentsByScope.delete(key)

    return
  }

  composerAttachmentsByScope.set(key, cloneComposerAttachments(attachments))
}

export function takeComposerAttachments(scope: string | null | undefined): ComposerAttachment[] {
  const stashed = composerAttachmentsByScope.get(storageScope(scope))

  return stashed ? cloneComposerAttachments(stashed) : []
}

export function clearStashedComposerAttachments(scope: string | null | undefined) {
  composerAttachmentsByScope.delete(storageScope(scope))
}

export function setComposerDraft(value: string) {
  $composerDraft.set(value)
}

export function appendComposerDraft(value: string) {
  const text = value.trim()

  if (!text) {
    return
  }

  const current = $composerDraft.get()
  const separator = current && !current.endsWith('\n') ? '\n\n' : ''

  $composerDraft.set(`${current}${separator}${text}`)
}

export function appendComposerInline(value: string) {
  const text = value.trim()

  if (!text) {
    return
  }

  const current = $composerDraft.get().trimEnd()
  const separator = current ? ' ' : ''

  $composerDraft.set(`${current}${separator}${text}`)
}

export function clearComposerDraft() {
  $composerDraft.set('')
}

export function addComposerAttachment(attachment: ComposerAttachment) {
  const previous = $composerAttachments.get()
  const next = upsertAttachment(previous, attachment)
  $composerAttachments.set(next)

  if (next.length > previous.length && attachment.kind !== 'url') {
    triggerHaptic('selection')
  }
}

export function removeComposerAttachment(id: string): ComposerAttachment | null {
  const current = $composerAttachments.get()
  const removed = current.find(attachment => attachment.id === id) || null
  $composerAttachments.set(current.filter(attachment => attachment.id !== id))

  return removed
}

/** Replace an existing attachment in place by id. No-op (returns false) when the
 * id is gone — e.g. the user removed the chip while an eager upload was still in
 * flight, so a late success must NOT resurrect it. Use this instead of
 * addComposerAttachment for async results that may land after a removal. */
export function updateComposerAttachment(attachment: ComposerAttachment): boolean {
  const current = $composerAttachments.get()
  const index = current.findIndex(item => item.id === attachment.id)

  if (index < 0) {
    return false
  }

  const next = [...current]
  next[index] = attachment
  $composerAttachments.set(next)

  return true
}

export function clearComposerAttachments() {
  $composerAttachments.set([])
}

/** Update only the upload state of an existing attachment (no-op if it's gone,
 * e.g. the user removed it mid-upload). Pass `undefined` to clear it. */
export function setComposerAttachmentUploadState(id: string, uploadState?: ComposerAttachment['uploadState']) {
  const current = $composerAttachments.get()
  const index = current.findIndex(attachment => attachment.id === id)

  if (index < 0) {
    return
  }

  const next = [...current]
  next[index] = { ...next[index]!, uploadState }
  $composerAttachments.set(next)
}

const TERMINAL_REF_RE = /@terminal:(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)/g

function unquoteRefValue(raw: string) {
  const head = raw[0]
  const tail = raw[raw.length - 1]
  const quoted = (head === '`' && tail === '`') || (head === '"' && tail === '"') || (head === "'" && tail === "'")

  return (quoted ? raw.slice(1, -1) : raw).replace(/[,.;!?]+$/, '').trim()
}

function terminalLabelsFromDraft(draft: string) {
  const labels: string[] = []
  const seen = new Set<string>()

  for (const match of draft.matchAll(TERMINAL_REF_RE)) {
    const label = unquoteRefValue(match[1] || '')

    if (!label || seen.has(label)) {
      continue
    }

    seen.add(label)
    labels.push(label)
  }

  return labels
}

export function setComposerTerminalSelection(label: string, text: string) {
  const nextLabel = label.trim()
  const nextText = text.trim()

  if (!nextLabel || !nextText) {
    return
  }

  const current = $composerTerminalSelections.get()

  if (current[nextLabel] === nextText) {
    return
  }

  $composerTerminalSelections.set({
    ...current,
    [nextLabel]: nextText
  })
}

export function reconcileComposerTerminalSelections(draft: string) {
  const current = $composerTerminalSelections.get()
  const labels = new Set(terminalLabelsFromDraft(draft))
  let changed = false
  const next: Record<string, string> = {}

  for (const [label, text] of Object.entries(current)) {
    if (!labels.has(label)) {
      changed = true

      continue
    }

    next[label] = text
  }

  if (changed) {
    $composerTerminalSelections.set(next)
  }
}

export function terminalContextBlocksFromDraft(draft: string) {
  const labels = terminalLabelsFromDraft(draft)

  if (labels.length === 0) {
    return []
  }

  const selections = $composerTerminalSelections.get()

  return labels.flatMap(label => {
    const text = selections[label]?.trim()

    if (!text) {
      return []
    }

    return `\`\`\`terminal\n${text}\n\`\`\``
  })
}

export function clearComposerTerminalSelections() {
  if (Object.keys($composerTerminalSelections.get()).length === 0) {
    return
  }

  $composerTerminalSelections.set({})
}

function upsertAttachment(attachments: ComposerAttachment[], attachment: ComposerAttachment) {
  const index = attachments.findIndex(item => item.id === attachment.id)

  if (index < 0) {
    return [...attachments, attachment]
  }

  const next = [...attachments]
  next[index] = attachment

  return next
}
