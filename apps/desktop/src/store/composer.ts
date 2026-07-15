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

// ---------------------------------------------------------------------------
// Composer scopes — one live attachment set PER MOUNTED COMPOSER. The main
// chat's scope wraps the module-level atom above (all existing readers keep
// working); each session tile creates its own so two composers on screen
// never share chips. Draft text needs no scope: it lives in each ChatBar's
// DOM + draftRef and stashes per session key already.
// ---------------------------------------------------------------------------

export interface ComposerAttachmentScope {
  $attachments: ReturnType<typeof atom<ComposerAttachment[]>>
  add(attachment: ComposerAttachment): void
  clear(): void
  remove(id: string): ComposerAttachment | null
  setUploadState(id: string, uploadState?: ComposerAttachment['uploadState']): void
  update(attachment: ComposerAttachment): boolean
}

export function createComposerAttachmentScope(
  $attachments = atom<ComposerAttachment[]>([])
): ComposerAttachmentScope {
  return {
    $attachments,
    add(attachment) {
      const previous = $attachments.get()
      const next = upsertAttachment(previous, attachment)
      $attachments.set(next)

      if (next.length > previous.length && attachment.kind !== 'url') {
        triggerHaptic('selection')
      }
    },
    clear() {
      $attachments.set([])
    },
    remove(id) {
      const current = $attachments.get()
      const removed = current.find(attachment => attachment.id === id) || null
      $attachments.set(current.filter(attachment => attachment.id !== id))

      return removed
    },
    setUploadState(id, uploadState) {
      const current = $attachments.get()
      const index = current.findIndex(attachment => attachment.id === id)

      if (index < 0) {
        return
      }

      const next = [...current]
      next[index] = { ...next[index]!, uploadState }
      $attachments.set(next)
    },
    update(attachment) {
      const current = $attachments.get()
      const index = current.findIndex(item => item.id === attachment.id)

      if (index < 0) {
        return false
      }

      const next = [...current]
      next[index] = attachment
      $attachments.set(next)

      return true
    }
  }
}

/** The main chat's scope — the module-level atom, so every existing
 *  `$composerAttachments` reader/writer IS this scope. */
export const mainComposerScope = createComposerAttachmentScope($composerAttachments)

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

// Main-scope conveniences — the names the app has always used.
export const addComposerAttachment = (attachment: ComposerAttachment) => mainComposerScope.add(attachment)
export const removeComposerAttachment = (id: string) => mainComposerScope.remove(id)

/** Replace an existing attachment in place by id. No-op (returns false) when the
 * id is gone — e.g. the user removed the chip while an eager upload was still in
 * flight, so a late success must NOT resurrect it. Use this instead of
 * addComposerAttachment for async results that may land after a removal. */
export const updateComposerAttachment = (attachment: ComposerAttachment) => mainComposerScope.update(attachment)

export const clearComposerAttachments = () => mainComposerScope.clear()

/** Update only the upload state of an existing attachment (no-op if it's gone,
 * e.g. the user removed it mid-upload). Pass `undefined` to clear it. */
export const setComposerAttachmentUploadState = (id: string, uploadState?: ComposerAttachment['uploadState']) =>
  mainComposerScope.setUploadState(id, uploadState)

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
