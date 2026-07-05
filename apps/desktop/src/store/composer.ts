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

// Per-thread draft stash for the decoupled composer. Session lifecycle never
// touches this — only ChatBar's scope swap reads/writes it. Text mirrors to
// localStorage; attachments are memory-only (blobs, upload state).
export const SESSION_DRAFTS_STORAGE_KEY = 'hermes:composer-drafts:v3'

const NEW_SESSION_DRAFT_KEY = '__new__'
const MAX_PERSISTED_DRAFTS = 50
const EMPTY_SESSION_DRAFT: SessionDraft = { attachments: [], text: '' }

export interface SessionDraft {
  attachments: ComposerAttachment[]
  text: string
}

const draftKey = (scope: string | null | undefined) => scope?.trim() || NEW_SESSION_DRAFT_KEY

const cloneDraft = (draft: SessionDraft): SessionDraft => ({
  attachments: draft.attachments.map(attachment => ({ ...attachment })),
  text: draft.text
})

function loadPersistedDraftTexts(): [string, SessionDraft][] {
  try {
    const raw = window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY)

    if (!raw) {
      return []
    }

    return Object.entries(JSON.parse(raw) as Record<string, string>).map(([key, text]) => [
      key,
      { attachments: [], text }
    ])
  } catch {
    return []
  }
}

const draftsBySession = new Map<string, SessionDraft>(loadPersistedDraftTexts())

function persistDraftTexts() {
  try {
    const entries = [...draftsBySession]
      .filter(([, draft]) => draft.text)
      .slice(-MAX_PERSISTED_DRAFTS)
      .map(([key, draft]) => [key, draft.text] as const)

    if (entries.length === 0) {
      window.localStorage.removeItem(SESSION_DRAFTS_STORAGE_KEY)
    } else {
      window.localStorage.setItem(SESSION_DRAFTS_STORAGE_KEY, JSON.stringify(Object.fromEntries(entries)))
    }
  } catch {
    // Best-effort only — quota/private-mode must never break typing.
  }
}

export function stashSessionDraft(scope: string | null | undefined, text: string, attachments: ComposerAttachment[]) {
  const key = draftKey(scope)

  // Delete-then-set keeps MRU order for MAX_PERSISTED_DRAFTS eviction.
  draftsBySession.delete(key)

  if (text.trim() || attachments.length > 0) {
    draftsBySession.set(key, cloneDraft({ attachments, text }))
  }

  persistDraftTexts()
}

export function takeSessionDraft(scope: string | null | undefined): SessionDraft {
  const stashed = draftsBySession.get(draftKey(scope))

  return stashed ? cloneDraft(stashed) : EMPTY_SESSION_DRAFT
}

export const clearSessionDraft = (scope: string | null | undefined) => stashSessionDraft(scope, '', [])

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
