import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { translateNow } from '@/i18n'
import { registerDraftProvider } from '@/store/composer-suggestions'

/**
 * Recurrence draft provider: the draft reads like a recurring task
 * ("remind me every morning…", "check this daily"), so offer to schedule it
 * as a cron job instead of a one-off message.
 *
 * Invoke prefixes the draft with an explicit scheduling instruction — the
 * agent owns parsing the schedule and creating the job (cronjob tool); the
 * pill only makes the intent unambiguous. Same reversible-edit contract as
 * the skill pill: it touches the draft, nothing else, and stands down once
 * the draft leads with the instruction it inserted.
 */

// Recurrence phrasing, deliberately narrow (VS Code lesson: high-precision
// triggers over statistical ones). Whole-word, unicode boundaries; hyphen
// counts as a word character so "weekly-report.pdf" stays quiet.
// - "every <hour|morning|day|week|weekday|month|monday…>" (+ "every 2 hours")
// - "daily" / "weekly" / "monthly" / "nightly" / "hourly"
// - "each morning/day/week/…"
// - a bare "remind me" is NOT a trigger — one-shot reminders are fine as
//   normal messages.
const RECURRENCE_RE =
  /(?<![\p{L}\p{N}-])(?:(?:every|each)\s+(?:\d+\s+)?(?:second|minute|hour|morning|afternoon|evening|night|day|weekday|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?|daily|weekly|monthly|nightly|hourly)(?![\p{L}\p{N}-])/giu

// A bare frequency adverb immediately followed by a Capitalized word is a
// proper noun ("the Daily Prophet", "Weekly Standup notes"), not a schedule.
// Checked in code because the regex's `i` flag case-folds \p{Lu} inside the
// pattern (making "please" read as capitalized).
const ADVERB_RE = /^(daily|weekly|monthly|nightly|hourly)$/i

/** Pure matcher, exported for tests: the recurrence phrase that fired, or
 *  null. The pill tooltip shows it, same attribution rule as every source. */
export function matchRecurrence(text: string): string | null {
  for (const match of text.matchAll(RECURRENCE_RE)) {
    const phrase = match[0]

    if (ADVERB_RE.test(phrase) && /^\s+\p{Lu}/u.test(text.slice(match.index + phrase.length))) {
      continue
    }

    return phrase
  }

  return null
}

registerDraftProvider('cron', async ({ text }) => {
  const copy = (key: string, ...args: unknown[]) => translateNow(`composer.cronSuggestions.${key}`, ...args)
  const trimmed = text.trimStart()

  // Already a slash command, or already carrying the instruction we insert —
  // stand down.
  if (trimmed.startsWith('/') || trimmed.toLowerCase().startsWith(copy('prefix').toLowerCase())) {
    return []
  }

  const phrase = matchRecurrence(text)

  if (!phrase) {
    return []
  }

  return [
    {
      doneLabel: copy('done'),
      doneTip: copy('doneTip'),
      icon: 'calendar',
      id: 'schedule',
      invoke: async () => {
        requestComposerInsert(copy('prefix'), { mode: 'prefix' })
        requestComposerFocus()
      },
      label: copy('label'),
      provider: 'cron',
      tip: copy('tip', phrase),
      workingLabel: copy('label'),
      workingTip: copy('tip', phrase)
    }
  ]
})
