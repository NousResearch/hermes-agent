import { SLASH_COMMAND_RE } from '@hermes/shared'
import { atom } from 'nanostores'

import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
import { persistBoolean, storedBoolean } from '@/lib/storage'

// Plan mode: while on, each new turn the user sends goes out as `/plan <text>`,
// so the built-in /plan command writes a plan to `.hermes/plans/` instead of
// executing. Composer-side only: the gateway sees an ordinary /plan
// invocation, so system prompt and toolsets never change mid-conversation.

const KEY = 'hermes.desktop.planMode.v1'

export const $planMode = atom(storedBoolean(KEY, false))

$planMode.subscribe(on => persistBoolean(KEY, on))

export function setPlanMode(on: boolean) {
  $planMode.set(on)
}

export function togglePlanMode() {
  $planMode.set(!$planMode.get())
}

type PlanModeOptions = Pick<SubmitTextOptions, 'attachments' | 'displayKind' | 'displayText' | 'surface'>

/**
 * The text actually sent for a submission, given the plan-mode state at SEND
 * time. Unchanged unless plan mode is on and the text is a user-typed, text-only
 * message: an explicit slash command wins (no `/plan /help`, no double prefix);
 * a message with attachments is left alone because slash commands never carry
 * attachments; hidden notes, pre-expanded skill kickoffs (`displayText`) and
 * voice-live turns are machine text, not a request to plan.
 */
export function applyPlanMode(text: string, options?: PlanModeOptions): string {
  const trimmed = text.trim()

  if (
    !$planMode.get() ||
    !trimmed ||
    SLASH_COMMAND_RE.test(trimmed) ||
    options?.attachments?.length ||
    options?.displayKind ||
    options?.displayText ||
    options?.surface
  ) {
    return text
  }

  return `/plan ${trimmed}`
}

type Submit = (text: string, options?: SubmitTextOptions) => Promise<boolean> | boolean

/** Wrap a submit function so every send through it gets plan-mode routing,
 *  while callers keep (and restore / re-queue) the raw text. */
export const withPlanMode =
  (submit: Submit): Submit =>
  (text, options) =>
    submit(applyPlanMode(text, options), options)
