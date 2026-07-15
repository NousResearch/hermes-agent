import type { Unstable_TriggerItem } from '@assistant-ui/core'

import type { SlashChipKind } from '@/components/assistant-ui/directive-text'
import type { ComposerAttachment } from '@/store/composer'
import { setSessionPickerOpen } from '@/store/session'

export const COMPOSER_STACK_BREAKPOINT_PX = 320

// A single editor line is ~28px (--composer-input-min-height 1.625rem + 0.5rem
// vertical padding). Anything taller means the text wrapped to a second line,
// which is when the composer should expand to the stacked layout.
export const COMPOSER_SINGLE_LINE_MAX_PX = 36

export const COMPOSER_FADE_BACKGROUND =
  'linear-gradient(to bottom, transparent, color-mix(in srgb, var(--dt-background) 10%, transparent))'

// Quiet period after the last keystroke before persisting the draft;
// unmount/pagehide flushes bypass it.
export const DRAFT_PERSIST_DEBOUNCE_MS = 400

export const pickPlaceholder = (pool: readonly string[]) => pool[Math.floor(Math.random() * pool.length)]

/** Completion items can carry an `action` (set in use-slash-completions) that
 *  runs a side effect on pick instead of inserting a chip — e.g. the session
 *  picker's "Browse all…" entry opens the overlay. Table-driven so new action
 *  items are a registry row, not a composer branch. */
export const COMPLETION_ACTIONS: Record<string, () => void> = {
  'session-picker': () => setSessionPickerOpen(true)
}

/** Map a picked `/` completion to its pill accent. Driven by the completion
 *  group set in use-slash-completions (Skills / Themes / Commands|Options). */
export function slashChipKindForItem(item: Unstable_TriggerItem): SlashChipKind {
  const group = (item.metadata as { group?: unknown } | undefined)?.group

  if (group === 'Skills') {
    return 'skill'
  }

  if (group === 'Themes') {
    return 'theme'
  }

  return 'command'
}

/** A `/` query is at its arg stage once it's past the command name. */
export const slashArgStage = (query: string) => query.includes(' ')

/** The `/command` token of a slash query (`personality x` → `/personality`). */
export const slashCommandToken = (query: string) => `/${query.split(/\s+/, 1)[0]?.toLowerCase() ?? ''}`

export interface QueueEditState {
  attachments: ComposerAttachment[]
  draft: string
  entryId: string
  sessionKey: string
}

export const cloneAttachments = (attachments: ComposerAttachment[]) => attachments.map(a => ({ ...a }))

export interface PendingDraftPersist {
  scope: string | null
  text: string
}

/**
 * Defense-in-depth for #54527: the debounce timer and the `pagehide` flush
 * both write a captured `{ scope, text }` pair some time after it was
 * scheduled. Before either commits the write, this checks the pair is still
 * the one currently on file — i.e. nothing cleared or replaced it in the
 * meantime (a session swap, a newer keystroke). The scope-capture fix
 * upstream (`draftScopeRef`) already makes every captured pair correct by
 * construction; this guard exists so that if a future change reintroduces a
 * stale/live-ref read at one of these call sites, the write is dropped
 * instead of silently filing one session's text under another session's key.
 */
export function isPendingDraftPersistCurrent(
  pending: PendingDraftPersist | null,
  expected: PendingDraftPersist | null
): boolean {
  return (
    pending !== null && expected !== null && pending.scope === expected.scope && pending.text === expected.text
  )
}
