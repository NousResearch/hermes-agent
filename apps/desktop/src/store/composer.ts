import { atom } from 'nanostores'

import { deriveDraftTitle } from '@/lib/draft-title'
import { triggerHaptic } from '@/lib/haptics'

export interface ComposerAttachment {
  id: string
  kind: 'file' | 'folder' | 'image' | 'review' | 'terminal' | 'url'
  label: string
  detail?: string
  refText?: string
  previewUrl?: string
  /** Downscaled data URL for the attachment card's <img> pill only. Keeps the
   * full-resolution `previewUrl` for lightbox/download/model bytes. */
  thumbnailUrl?: string
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

// Latched because opening a fresh session may remount the main composer before
// it can start voice. Session-tile composers deliberately never consume this.
export const $voiceConversationStartRequest = atom(0)
let nextVoiceStartRequest = 0
let handledVoiceStartRequest = 0

export const requestVoiceConversationStart = (): void => $voiceConversationStartRequest.set(++nextVoiceStartRequest)

export const takeVoiceConversationStart = (current: number): boolean => {
  if (current <= handledVoiceStartRequest) {
    return false
  }

  handledVoiceStartRequest = current

  return true
}

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

export function createComposerAttachmentScope($attachments = atom<ComposerAttachment[]>([])): ComposerAttachmentScope {
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

/**
 * What each unsent draft would be called, keyed the same way its text is.
 *
 * A draft has no session to carry a title, so the tab showing it reads this
 * instead of the "New session" placeholder. Written from `stashSessionDraft`,
 * the one funnel every composer's text already flows through — the debounce
 * that persists a draft is the same beat that renames its tab, so typing costs
 * nothing extra. Only tabs showing a draft subscribe, and each selects its own
 * key, so a rename repaints one label rather than the strip.
 *
 * Seeded from the persisted texts: a draft left open across a restart comes
 * back already named.
 */
export const $draftTitles = atom<Record<string, string>>(
  Object.fromEntries(
    [...draftsBySession].map(([key, draft]) => [key, deriveDraftTitle(draft.text)]).filter(([, title]) => title)
  )
)

/** Read one draft's title out of the map — for a `useStoreSelector`, so a tab
 *  repaints on its OWN rename rather than on every draft's. */
export const draftTitleIn = (titles: Record<string, string>, scope: string | null | undefined): string =>
  titles[draftKey(scope)] ?? ''

export const draftTitleFor = (scope: string | null | undefined): string => draftTitleIn($draftTitles.get(), scope)

function publishDraftTitle(key: string, title: string): void {
  const current = $draftTitles.get()

  if ((current[key] ?? '') === title) {
    return
  }

  const next = { ...current }

  if (title) {
    next[key] = title
  } else {
    delete next[key]
  }

  $draftTitles.set(next)
}

/**
 * Re-read the persisted drafts written by ANOTHER window into this one's map.
 *
 * Drafts are per-renderer state backed by shared localStorage, and the map
 * above is read exactly once at module load. Two windows on the same session
 * (HUD mode ⇄ the app window) therefore diverge the moment either one types:
 * whichever window mounted first keeps its stale copy forever, so text typed
 * in the HUD is simply gone when you return to the app.
 *
 * Merge, don't clobber — the local map may hold attachments (never persisted)
 * that the incoming text-only snapshot can't know about.
 */
export function reloadPersistedDrafts(): void {
  const incoming = new Map(loadPersistedDraftTexts())

  for (const [key, draft] of incoming) {
    const local = draftsBySession.get(key)
    draftsBySession.set(key, local?.attachments.length ? { ...local, text: draft.text } : draft)
    publishDraftTitle(key, deriveDraftTitle(draft.text))
  }

  // A key that vanished from storage was cleared (sent) in the other window.
  for (const key of [...draftsBySession.keys()]) {
    if (!incoming.has(key)) {
      draftsBySession.delete(key)
      publishDraftTitle(key, '')
    }
  }
}

// localStorage `storage` events fire across Electron BrowserWindows of the
// same origin, so the other window's write is the sync signal.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (event.key === SESSION_DRAFTS_STORAGE_KEY) {
      reloadPersistedDrafts()
    }
  })
}

/**
 * Push a composer's live text into the shared stash (`flush`), or repaint it
 * from the stash (`reload`).
 *
 * Both halves of the HUD handoff need this. The stash is the only draft state
 * two windows share, but a mounted composer only consults it when its session
 * scope changes — so entering HUD mode has to flush the app window's in-editor
 * text down to the stash before the HUD boots and reads it, and leaving has to
 * repaint the app's editor from whatever the HUD left behind (usually empty,
 * because the HUD sent it).
 *
 * Dispatched synchronously, unlike the focus bus: the flush must complete
 * before the HUD window is created.
 */
const DRAFT_SYNC_EVENT = 'hermes:composer-draft-sync'

export type ComposerDraftSyncMode = 'flush' | 'reload'

interface ComposerDraftSyncDetail {
  mode: ComposerDraftSyncMode
  target: string
}

export function requestComposerDraftSync(mode: ComposerDraftSyncMode, target = 'main'): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent<ComposerDraftSyncDetail>(DRAFT_SYNC_EVENT, { detail: { mode, target } }))
  }
}

export function onComposerDraftSyncRequest(handler: (detail: ComposerDraftSyncDetail) => void): () => void {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const listener = (event: Event) => handler((event as CustomEvent<ComposerDraftSyncDetail>).detail)
  window.addEventListener(DRAFT_SYNC_EVENT, listener)

  return () => window.removeEventListener(DRAFT_SYNC_EVENT, listener)
}

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
  publishDraftTitle(key, deriveDraftTitle(text))
}

export function takeSessionDraft(scope: string | null | undefined): SessionDraft {
  const stashed = draftsBySession.get(draftKey(scope))

  return stashed ? cloneDraft(stashed) : EMPTY_SESSION_DRAFT
}

export const clearSessionDraft = (scope: string | null | undefined) => stashSessionDraft(scope, '', [])

/**
 * Move a stashed composer draft from one session key onto another.
 *
 * Auto-compression rotates the live stored tip id (root → continuation) while
 * the user may still be typing. Drafts keyed on the obsolete tip would otherwise
 * vanish from the composer when selection follows the new tip. No-op unless both
 * keys resolve, differ, and the source has content. Does not overwrite a
 * non-empty destination draft.
 */
export function migrateSessionDraft(fromKey: string | null | undefined, toKey: string | null | undefined): boolean {
  const from = draftKey(fromKey)
  const to = draftKey(toKey)

  if (!fromKey || !toKey || from === to) {
    return false
  }

  const source = draftsBySession.get(from)

  if (!source || (!source.text.trim() && source.attachments.length === 0)) {
    return false
  }

  const dest = draftsBySession.get(to)

  if (dest && (dest.text.trim() || dest.attachments.length > 0)) {
    return false
  }

  stashSessionDraft(toKey, source.text, source.attachments)
  clearSessionDraft(fromKey)

  return true
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
