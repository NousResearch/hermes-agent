import type { Unstable_TriggerItem } from '@assistant-ui/core'

import type { SlashChipKind } from '@/components/assistant-ui/directive-text'
import type { ComposerAttachment } from '@/store/composer'
import { setSessionPickerOpen } from '@/store/session'

import type { TriggerState } from './text-utils'

export const COMPOSER_STACK_BREAKPOINT_PX = 320

// Above the stack breakpoint but still cramped: the model pill sheds its label
// for its chevron icon so the controls stop crowding the input before the whole
// row has to stack. Progressive collapse: full pill → icon pill → stacked.
//
// Sized off what the controls actually cost, because guessing put the two
// stages on top of each other. With the full pill the controls take ~284px
// (pill 111 + the icon cluster), so at the old 440 the inline input was ~156px
// — barely over its 128px minimum. A few words wrapped, wrapping is what
// stacks the row, and the pill's chevron arrived at the same moment the row
// gave up, which is the one thing progressive collapse is supposed to avoid.
// At 560 the label goes while the input still has ~276px, and the ~110px the
// chevron frees is spent keeping the row single for another stretch.
export const COMPOSER_COMPACT_PILL_PX = 560

// The ladder keeps going below the stack breakpoint — a pane can be far
// narrower than even the stacked controls row. Both rungs are budgeted
// against that row's real cost: menu ~24 + surface padding 16 + the cluster
// (~190; ~218 mid-turn with the queue button).
//
// At 260 the three voice toggles fold into the one menu HUD mode already
// uses, clearing the mid-turn worst case with margin. Each stage sits clear
// of the floor below it rather than arriving the instant the previous one
// gives out — the mistake COMPOSER_COMPACT_PILL_PX documents.
export const COMPOSER_FOLD_VOICE_PX = 260

// Type and send, nothing else. A pane can be dragged to MIN_PANE_PX (80), and
// even with voice folded the row still costs ~150, so the last rung drops the
// pill AND the voice menu. Both stay reachable — the model by hotkey and the
// full picker, dictation from any wider pane — and Send fits with room to
// spare at any width the layout tree allows (~74 all-in).
export const COMPOSER_MINIMAL_PX = 180

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

/** True for a skill completion — the only kind offered mid-message. */
export const isSkillItem = (item: Unstable_TriggerItem) => slashChipKindForItem(item) === 'skill'

/** A `/` query is at its arg stage once it's past the command name. */
export const slashArgStage = (query: string) => query.includes(' ')

/** The `/command` token of a slash query (`personality x` → `/personality`). */
export const slashCommandToken = (query: string) => `/${query.split(/\s+/, 1)[0]?.toLowerCase() ?? ''}`

/** Typed `/` query or completion text without the leading slash. */
export const slashCompletionToken = (value: string) => value.replace(/^\//, '').trimEnd().toLowerCase()

/**
 * Which row Space/Enter should take. Tab always uses the highlight; this is
 * the "still suggesting, don't steal what I typed" pick.
 *
 * `/com` + a highlighted `/compress` still completes. `/review` while
 * `/compress` is the leftover highlight does not. An exact list match wins
 * so a fully typed command isn't replaced by a longer/fuzzier neighbour.
 */
export function implicitSlashAcceptIndex(
  query: string,
  itemTexts: readonly string[],
  activeIndex: number,
  activeExplicit: boolean
): number | null {
  const typed = slashCompletionToken(query)

  if (!typed) {
    return null
  }

  if (activeExplicit && itemTexts[activeIndex] != null) {
    return activeIndex
  }

  const exact = itemTexts.findIndex(text => slashCompletionToken(text) === typed)

  if (exact >= 0) {
    return exact
  }

  const active = itemTexts[activeIndex]

  if (active != null && slashCompletionToken(active).startsWith(typed)) {
    return activeIndex
  }

  const prefixHits = itemTexts.flatMap((text, index) => (slashCompletionToken(text).startsWith(typed) ? [index] : []))

  return prefixHits.length === 1 ? prefixHits[0] : null
}

export interface TriggerAcceptInput {
  /** The user moved the highlight themselves (arrow keys) rather than
   *  inheriting the list's default first row. */
  activeExplicit: boolean
  /** The trigger is a slash command whose argument is arbitrary prose. */
  freeTextArgStage: boolean
  key: string
  kind: TriggerState['kind']
  query: string
}

/**
 * Whether a keypress accepts the highlighted completion while the popover is
 * open. Tab is always an accept — it has no other meaning in the composer.
 *
 * Enter and Space are conditional, because both mean something else while a
 * free-text argument is being written (`/goal ship the redesign`). Space types
 * a space, and Enter sends the message; letting either take the popover's
 * pre-highlighted row would swap the prose the user is mid-sentence on for a
 * subcommand they never chose. Enter still accepts once the user has arrowed
 * to a row deliberately, so the highlight never lies about what Enter will do.
 */
export function acceptsTriggerCompletion({
  activeExplicit,
  freeTextArgStage,
  key,
  kind,
  query
}: TriggerAcceptInput): boolean {
  if (key === 'Tab') {
    return true
  }

  if (key === 'Enter') {
    return !freeTextArgStage || activeExplicit
  }

  // Space is slash-only (an `@` mention takes a literal space) and gated to a
  // non-empty query so a bare `/ ` still types a space.
  return key === ' ' && kind === '/' && Boolean(query.trim()) && !freeTextArgStage
}

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
  return pending !== null && expected !== null && pending.scope === expected.scope && pending.text === expected.text
}
