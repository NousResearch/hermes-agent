/**
 * The one-line "what it's doing" label on the turn's status row, Claude Code
 * style: `[pulse] Checking the adapter's display setting… 40s · 8.5k tokens`.
 *
 * Claude Code builds its spinner as `override ?? inProgressTodo.activeForm ??
 * randomVerb`. Hermes has the same override layer (compaction, provider wait,
 * a tool being drafted — `useStatusHint` in status.tsx) and adds one source
 * Claude Code's terminal can't use: the model's own streamed reasoning. Opus
 * 5.x narrates its summarized thinking in the first person ("I'm checking
 * whether…", "Next I'll grep…"), so the newest sentence of the newest
 * reasoning part, turned into a present-tense phrase, says what the turn is
 * doing right now — with no extra model call and no backend change.
 *
 * Everything here is pure so the wording rules are tested without a renderer.
 */

import type { TodoItem } from '@/lib/todos'

/** Longest label we hand the row. The row truncates to its width anyway; this
 *  only keeps a run-on sentence from carrying a paragraph into the DOM. */
export const SPINNER_LABEL_MAX = 90

// Claude Code's spinner word list is whimsical on purpose: when the turn has
// nothing better to say, a playful verb reads as alive rather than stuck. One is
// picked per turn (seeded by the turn start) so it never strobes mid-turn.
export const SPINNER_VERBS = [
  'Thinking',
  'Pondering',
  'Working',
  'Mulling',
  'Reasoning',
  'Cooking',
  'Noodling',
  'Brewing',
  'Crunching',
  'Figuring',
  'Considering',
  'Computing',
  'Synthesizing',
  'Percolating',
  'Untangling',
  'Connecting dots'
] as const

export function spinnerVerb(seed: number | undefined): string {
  const n = Number.isFinite(seed) ? Math.abs(Math.trunc(seed as number)) : 0

  return SPINNER_VERBS[n % SPINNER_VERBS.length]
}

// Base verbs whose -ing form doubles the final consonant (CVC stress) or is
// otherwise irregular. Anything not listed follows the regular rules below.
const GERUND_EXCEPTIONS: Readonly<Record<string, string>> = {
  admit: 'admitting',
  be: 'being',
  begin: 'beginning',
  chat: 'chatting',
  commit: 'committing',
  cut: 'cutting',
  debug: 'debugging',
  die: 'dying',
  dig: 'digging',
  drag: 'dragging',
  drop: 'dropping',
  flag: 'flagging',
  get: 'getting',
  grab: 'grabbing',
  grep: 'grepping',
  hit: 'hitting',
  let: 'letting',
  lie: 'lying',
  log: 'logging',
  map: 'mapping',
  pin: 'pinning',
  plan: 'planning',
  plug: 'plugging',
  prep: 'prepping',
  put: 'putting',
  rerun: 'rerunning',
  run: 'running',
  scan: 'scanning',
  set: 'setting',
  ship: 'shipping',
  sit: 'sitting',
  skim: 'skimming',
  spin: 'spinning',
  step: 'stepping',
  stop: 'stopping',
  submit: 'submitting',
  swap: 'swapping',
  tag: 'tagging',
  tap: 'tapping',
  tie: 'tying',
  trim: 'trimming',
  wrap: 'wrapping',
  zip: 'zipping'
}

const capitalize = (s: string): string => (s ? s[0].toUpperCase() + s.slice(1) : s)

/** "check" -> "checking", "write" -> "writing", "run" -> "running". Keeps the
 *  input's leading case. Words already in -ing form pass through. */
export function toGerund(verb: string): string {
  const lower = verb.toLowerCase()

  if (!/^[a-z]+$/.test(lower) || lower.endsWith('ing')) {
    return verb
  }

  let out = GERUND_EXCEPTIONS[lower]

  if (!out) {
    if (/(?:ee|ye|oe)$/.test(lower)) {
      out = `${lower}ing`
    } else if (lower.endsWith('ie')) {
      out = `${lower.slice(0, -2)}ying`
    } else if (lower.length > 2 && lower.endsWith('e')) {
      out = `${lower.slice(0, -1)}ing`
    } else {
      out = `${lower}ing`
    }
  }

  return verb[0] === verb[0].toUpperCase() ? capitalize(out) : out
}

// First words a todo title can open with that are NOT an imperative verb —
// "New sidebar section" must not become "Newing sidebar section".
const NOT_A_VERB = new Set([
  'a',
  'all',
  'an',
  'another',
  'each',
  'every',
  'final',
  'first',
  'my',
  'new',
  'next',
  'no',
  'one',
  'our',
  'some',
  'the',
  'these',
  'this',
  'those',
  'two'
])

/** Imperative title -> present-tense label: "Build the row" -> "Building the
 *  row" (Claude Code's activeForm, derived instead of asked for). */
export function imperativeToActive(title: string): string {
  const text = title.trim()
  const match = /^([A-Za-z]+)(\b.*)$/s.exec(text)

  if (!match) {
    return text
  }

  const [, first, rest] = match

  // All-caps first word is an acronym ("API audit"), and past tense / -ing
  // already reads as a status.
  if (first.length < 2 || first === first.toUpperCase() || NOT_A_VERB.has(first.toLowerCase())) {
    return text
  }

  if (/(?:ing|ed)$/i.test(first)) {
    return capitalize(text)
  }

  return capitalize(toGerund(first)) + rest
}

const clip = (s: string): string => {
  if (s.length <= SPINNER_LABEL_MAX) {
    return s
  }

  const cut = s.slice(0, SPINNER_LABEL_MAX)
  const space = cut.lastIndexOf(' ')

  return (space > SPINNER_LABEL_MAX * 0.6 ? cut.slice(0, space) : cut).replace(/[\s,;:–—-]+$/, '')
}

const LEADING_FILLER =
  /^(?:(?:ok(?:ay)?|so|now|next|first|then|alright|right|also|finally|good|separately|meanwhile|in the meantime)[,:]?\s+)+/i

// "Nothing has replied yet, so I'll search the channel" — the clause after the
// "so" is what the turn is doing NOW; the one before it is the reason.
const SO_ACTION =
  /^.{0,140}?,?\s+so (?=(?:I(?:'ll|’ll| will| want to| need to| should|'m going to|’m going to| am going to)|let me)\s)/i

// Where a long label can stop without losing its point: the first clause break
// past a short head (", since a" trailing off at the clip reads as broken).
const CLAUSE_BREAK =
  /(?:[,;:]|\s[—–]|\s\()\s|\s(?:since|because|so that|which|so I|and|then|before|while|to see whether|to see if|to confirm)\s/g

const SHORT_ENOUGH = 56
const MIN_HEAD = 22

function shortenAtClause(text: string): string {
  if (text.length <= SHORT_ENOUGH) {
    return text
  }

  for (const match of text.matchAll(CLAUSE_BREAK)) {
    if (match.index !== undefined && match.index >= MIN_HEAD) {
      return text.slice(0, match.index)
    }
  }

  return text
}

// Each rule maps one first-person narration opener to the label's verb form.
// Order matters: the specific "I'm going to X" must win over "I'm X".
const NARRATION_RULES: ReadonlyArray<[RegExp, (m: RegExpExecArray) => string]> = [
  // "I'm going to grep X" / "I'm about to grep X" -> "Grepping X"
  [/^I(?:'m|’m| am) (?:going to|about to|now going to) ([A-Za-z]+)(.*)$/s, m => capitalize(toGerund(m[1])) + m[2]],
  // "I'm checking X" -> "Checking X"; "I'm now checking X" -> "Checking X"
  [/^I(?:'m|’m| am) (?:now |still |also |just )?([A-Za-z]+ing)\b(.*)$/s, m => capitalize(m[1]) + m[2]],
  // "I'll check X", "Let me check X", "I need to check X" -> "Checking X"
  [
    /^(?:I(?:'ll|’ll| will)(?: now| also| first| then| next| quickly| just)?|Let me(?: now| also| just| quickly)?|I(?: also| still| now| first)? (?:need to|want to|should|have to|must|could|can)|I'd like to|Time to|My next step is to|The next step is to) ([A-Za-z]+)(.*)$/is,
    m => capitalize(toGerund(m[1])) + m[2]
  ],
  // "I've found X" / "I have confirmed X" -> "Found X"
  [/^I(?:'ve|’ve| have) (?:now |just |already |also )?([A-Za-z]+)(.*)$/s, m => capitalize(m[1]) + m[2]],
  // "I confirmed X" -> "Confirmed X"
  [/^I ([A-Za-z]+ed)\b(.*)$/s, m => capitalize(m[1]) + m[2]]
]

/** One narrated sentence -> a short present-tense status phrase. Sentences that
 *  are not first-person narration ("Confirmed, so …") pass through untouched. */
export function statusFromSentence(sentence: string): string {
  // Underscores stay: they are part of identifiers (`execute_code`), not
  // emphasis, in this model's narration.
  let text = sentence
    .replace(/\*\*|`/g, '')
    .replace(/^\s*(?:#+|>|[-•*]|\d+[.)])\s+/, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(SO_ACTION, '')
    .replace(LEADING_FILLER, '')

  for (const [rule, render] of NARRATION_RULES) {
    const match = rule.exec(text)

    if (match) {
      text = render(match)

      break
    }
  }

  return clip(shortenAtClause(capitalize(text.trim())).replace(/[\s.:;,!?…]+$/, ''))
}

// A sentence ends at terminal punctuation FOLLOWED BY WHITESPACE, so a file
// name or version ("status.tsx", "v2.1") never splits one in two.
const SENTENCE_BREAK = /(?<=[.!?])\s+/

/** Newest COMPLETE sentence of a streamed reasoning text, else '' — a sentence
 *  still arriving word by word would make the label jitter on every token. A
 *  paragraph with no finished sentence yet defers to the paragraph before it. */
export function latestSentence(reasoning: string): string {
  const paragraphs = reasoning
    .split(/\n\s*\n/)
    .map(p => p.trim())
    .filter(Boolean)

  for (let i = paragraphs.length - 1; i >= 0; i -= 1) {
    const isLast = i === paragraphs.length - 1

    const sentences = paragraphs[i]
      .split(SENTENCE_BREAK)
      .map(s => s.trim())
      .filter(Boolean)

    // In the newest paragraph the final piece may be mid-stream: count it only
    // once it ends with terminal punctuation.
    const finished = isLast ? sentences.filter(s => /[.!?]$/.test(s)) : sentences

    if (finished.length > 0) {
      return finished[finished.length - 1]
    }
  }

  return ''
}

const REASONING_TAIL_CHARS = 4000

export interface SpinnerPart {
  text?: unknown
  type: string
}

/**
 * The label the newest reasoning gives, or '' when there is none worth showing.
 *
 * Only reasoning that comes AFTER the latest tool call counts: once a call has
 * run, the reasoning before it described that call, and the turn has moved on.
 */
export function latestReasoningStatus(content: readonly SpinnerPart[]): string {
  for (let i = content.length - 1; i >= 0; i -= 1) {
    const part = content[i]

    if (part.type === 'tool-call') {
      return ''
    }

    if (part.type === 'reasoning' && typeof part.text === 'string' && part.text.trim()) {
      // Runs on every streamed token: only the tail can hold the newest
      // sentence, so a long reasoning block costs the same as a short one.
      const sentence = latestSentence(part.text.slice(-REASONING_TAIL_CHARS))

      if (sentence) {
        return statusFromSentence(sentence)
      }
    }
  }

  return ''
}

/** Claude Code's activeForm fallback: the in-progress todo, as a status. */
export function todoStatus(todos: readonly TodoItem[] | undefined): string {
  const active = todos?.find(todo => todo.status === 'in_progress')

  return active ? clip(imperativeToActive(active.content)) : ''
}

/** "812 tokens", "8.5k tokens", "42k tokens"; '' for none. */
export function formatTurnTokens(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens <= 0) {
    return ''
  }

  if (tokens < 1000) {
    return `${Math.round(tokens)} tokens`
  }

  const k = tokens / 1000
  const shown = k < 10 ? k.toFixed(1).replace(/\.0$/, '') : String(Math.round(k))

  return `${shown}k tokens`
}

/** Precedence: reasoning (freshest, most specific) > in-progress todo > verb. */
export function pickSpinnerLabel(reasoning: string, todo: string, seed: number | undefined): string {
  return reasoning || todo || spinnerVerb(seed)
}
