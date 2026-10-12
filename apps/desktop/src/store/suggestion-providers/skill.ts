import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { getSkills } from '@/hermes'
import { translateNow } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { type ComposerSuggestion, registerDraftProvider } from '@/store/composer-suggestions'
import { $activeSessionId, $currentCwd, $messages } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

/**
 * Skill-match draft provider: the draft names a skill the user has, so offer
 * to load it with the message. Highest value/annoyance ratio of the ranked
 * sources — the skill index is local, the trigger is the user literally
 * typing the skill's name, and the action is reversible (it edits the draft,
 * nothing else).
 *
 * Invoke prefixes the draft with the skill's `/name` command; the pill then
 * self-limits naturally: the next draft sample sees a leading slash and the
 * provider stands down. Loading actually happens when the user sends —
 * the pill never sends on their behalf.
 */

const SKILLS_TTL_MS = 5 * 60_000
const MIN_NAME_LENGTH = 4
/** Bound on declared trigger phrases honoured per skill (the backend caps its side too): each entry
 *  becomes a regex tested against every draft sample, and the provider runs on each keystroke. */
const MAX_TRIGGERS = 12

interface SkillIndexEntry {
  name: string
  /** null when the name is below the pill's length floor — it never offers on its own. */
  pattern: RegExp | null
  /** Patterns for the frontmatter's declared triggers — the skill's reach in languages the user
   *  actually types. Empty when a skill declares none. */
  triggerPatterns: RegExp[]
}

let index: SkillIndexEntry[] | null = null
let indexAt = 0

/** Drop the cached skill index (skill_manage created/deleted a skill). */
export function invalidateSkillSuggestionIndex(): void {
  index = null
  indexAt = 0
}

const escape = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Whole-word pattern for a skill name, exported for tests. Hyphens and
 *  underscores in the name also match spaces — people type "pr ready", the
 *  skill is `pr-ready` — while "already" can never match a skill `read`. */
export function skillPattern(name: string): RegExp {
  const flexible = name.toLowerCase().split(/[-_]/).map(escape).join('[-_ ]')

  return new RegExp(`(?<![\\p{L}\\p{N}])${flexible}(?![\\p{L}\\p{N}-])`, 'gu')
}

/** Fold the Arabic spelling variants people type interchangeably, so a declared phrase meets the way
 *  the user actually writes rather than the one spelling its author happened to pick:
 *
 *    - alef variants `أ إ آ ٱ` (and their decomposed forms) become `ا`
 *    - diacritics (`U+064B–U+065F`, `U+0670`) and the tatweel/kashida `U+0640` are removed
 *
 *  Folded on BOTH sides — the declared phrase and the draft. Folding one side only matches nothing,
 *  which is why this is one function called twice rather than a rule stated in two places.
 *
 *  Deliberately NOT folded, each because it merges two DISTINCT words instead of two spellings of one:
 *  `ة/ه` (كتابة/كتابه), `ى/ي` (مصطفى/مصطفي), `ؤ/ئ`→`و/ي` (رؤية/روية). Trading a miss for a wrong match
 *  is worse than the miss, so a phrase containing them is declared the way it will be typed, and the
 *  tests assert the exclusions as well as the inclusions — an unbounded fold is how this feature would
 *  turn into noise.
 *
 *  `text.trim()` is not folded: trimming is the caller's job, and a fold that quietly ate leading
 *  whitespace would hide a phrase the author wrote by accident. */
export function foldArabic(text: string): string {
  return (
    text
      // NFC FIRST, and not as decoration: a decomposed spelling writes ؤ as (و + U+0654) and أ as
      // (ا + U+0654). Folding the marks off before composing would turn رؤية into روية — merging two
      // different words in the decomposed spelling while keeping them apart in the composed one, i.e.
      // the same input behaving differently by encoding. Composing first makes the two spellings fold
      // to the same string, which is what the exclusions below depend on.
      .normalize('NFC')
      .replace(/[\u0622\u0623\u0625\u0671]/g, '\u0627')
      .replace(/[\u064B-\u065F\u0670]/g, '')
      .replace(/\u0640/g, '')
  )
}

/** A declared trigger phrase, matched by the same whole-word rule as a name.
 *
 *  Why this exists: the provider's only signal used to be the skill's NAME, so a skill whose name the
 *  user never types — because they write in another language, or because the name is a common English
 *  word that carries no intent — could never be offered. A declared phrase list is authored in the
 *  user's own words, so `ملف pdf` or `دمج ملفين` reaches the skill that `pdf-editing` does not.
 *
 *  Whitespace inside a phrase is flexible (`دمج   ملفين` still matches), the phrase is escaped so
 *  punctuation in a trigger is literal, and Arabic orthography is folded by `foldArabic` — `الأرقام`
 *  declared and `الارقام` typed are the same phrase to a reader and to this matcher. An empty phrase
 *  yields a pattern that never matches rather than one that matches everywhere.
 *
 *  Sibling matcher, and why this one is not it: `mcp.ts` has `keywordHit` for the MCP catalog's
 *  `suggest.keywords`, the same whole-word job. The two differ deliberately in one place — the
 *  whitespace between words. MCP keywords are declared in a reviewed catalog manifest, where exact
 *  spacing is meaningful; skill triggers are hand-authored by whoever writes the skill, in their own
 *  language, and a user who types two spaces means the same phrase. Unifying them would either
 *  loosen the catalog's contract or break a real trigger, so the divergence is documented here and
 *  a shared `wholeWordPattern(phrase, { flexibleWhitespace })` is the follow-up that would remove
 *  the duplication without changing either caller's semantics. */
export function triggerPattern(phrase: string): RegExp {
  const parts = foldArabic(phrase.trim().toLowerCase()).split(/\s+/).filter(Boolean).map(escape)

  if (parts.length === 0) {
    // A blank entry is not a wildcard. Built from a string, not written as a literal, because the
    // `gu` flags must be explicit: skillHit runs matchAll, which throws "called with a non-global
    // RegExp argument" without `g` — measured, not assumed.
    return new RegExp('(?!)', 'gu')
  }

  return new RegExp(`(?<![\\p{L}\\p{N}])${parts.join('\\s+')}(?![\\p{L}\\p{N}-])`, 'gu')
}

/** Declared phrases as a clean string list, exported for tests.
 *
 *  Tolerates a malformed payload: a string where the type says array has a `.length` that passes the
 *  length gate and then no `.map`, which would throw inside the composer's render path. Non-string
 *  entries are dropped rather than stringified, for the same reason the backend refuses a scalar —
 *  `5` becoming the phrase "5" is a phrase nobody wrote. */
export function triggerPhrases(triggers: unknown): string[] {
  if (!Array.isArray(triggers)) {
    return []
  }

  return triggers
    .filter((trigger): trigger is string => typeof trigger === 'string')
    .map(trigger => trigger.trim())
    .filter(Boolean)
    .slice(0, MAX_TRIGGERS)
}

/** Completed whole-word hit, exported for tests: the match only counts once
 *  at least one character follows it. The debounce fires mid-sentence, and a
 *  name still under the caret is a word in progress, not a request — the pill
 *  waiting for the next keystroke is the difference between "offering" and
 *  "reading over your shoulder". */
export function skillHit(pattern: RegExp, haystack: string): boolean {
  pattern.lastIndex = 0

  for (const match of haystack.matchAll(pattern)) {
    if (match.index + match[0].length < haystack.length) {
      return true
    }
  }

  return false
}

/** Strip whitespace-bounded /<token> slash commands the user authored, so a
 *  skill name inside a command ("run /github-auth") does not self-trigger the
 *  prose suggestion pill (#91626). Whitespace-bounded on purpose: a URL like
 *  /api/v1 mid-prose has no leading whitespace and is left intact. Exported so
 *  the provider and its tests exercise the same sanitizer instead of a copy. */
export function stripSlashTokens(text: string): string {
  return text.replace(/\s\/[\w-]+/g, ' ')
}

/** Workspace homonym guard, exported for tests: a skill named like the
 *  working directory is the project's name, not a request for the skill.
 *  Working in `~/www/hermes-agent`, the draft says "hermes-agent" constantly
 *  — a "Use skill: hermes-agent" pill on every mention is noise (the reported
 *  annoyance that prompted this guard). Boundary containment, not exact
 *  segment equality, so worktree dirs (`hermes-agent-suggest`) suppress too. */
export function collidesWithWorkspace(name: string, cwd: string): boolean {
  return new RegExp(`(?<![\\p{L}\\p{N}])${escape(name.toLowerCase())}(?![\\p{L}\\p{N}])`, 'u').test(cwd.toLowerCase())
}

async function loadIndex(): Promise<SkillIndexEntry[]> {
  if (index && Date.now() - indexAt < SKILLS_TTL_MS) {
    return index
  }

  const skills = await getSkills()

  index = skills
    // A short name is unofferable AS a name, but a declared trigger still reaches it — so the length
    // floor applies to the name, not to the skill's presence in the index.
    .filter(
      skill => skill.enabled && (skill.name.length >= MIN_NAME_LENGTH || triggerPhrases(skill.triggers).length > 0)
    )
    .map(skill => ({
      name: skill.name,
      // null below the floor: a short name never offers on its own (unchanged behaviour), while a
      // declared trigger still may. An explicit null beats a sentinel pattern that matches nothing.
      pattern: skill.name.length >= MIN_NAME_LENGTH ? skillPattern(skill.name) : null,
      triggerPatterns: triggerPhrases(skill.triggers).map(triggerPattern)
    }))
  indexAt = Date.now()

  return index
}

// ---------------------------------------------------------------------------
// Already-in-context guard: a skill the session has engaged with must not be
// re-offered. skill_view loads the whole SKILL.md into the conversation and
// skill_manage implies the agent is already working with it — clicking the
// pill would just re-inject content the context already carries (the reported
// annoyance: a "Use skill" pill for a skill loaded at session start).
// ---------------------------------------------------------------------------

const SKILL_TOOL_NAMES = new Set(['skill_manage', 'skill_view'])

/** The `name` argument of a skill tool call, however the transcript stored
 *  it: live rows carry a parsed `args` object, hydrated rows may only have
 *  the JSON `argsText`. */
function skillArgName(part: { args?: unknown; argsText?: unknown }): string {
  const record = part.args && typeof part.args === 'object' ? (part.args as Record<string, unknown>) : null

  if (record && typeof record.name === 'string') {
    return record.name
  }

  if (typeof part.argsText === 'string' && part.argsText) {
    try {
      const parsed: unknown = JSON.parse(part.argsText)

      if (parsed && typeof parsed === 'object' && typeof (parsed as { name?: unknown }).name === 'string') {
        return (parsed as { name: string }).name
      }
    } catch {
      // Non-JSON argsText carries no name.
    }
  }

  return ''
}

/** True when the tool-call arg names this skill — exactly, or as the final
 *  segment of a qualified form (`github/hermes-agent-dev`, `plugin:skill`). */
function argNamesSkill(arg: string, skillName: string): boolean {
  const value = arg.trim().toLowerCase()

  if (!value) {
    return false
  }

  const target = skillName.toLowerCase()

  return value === target || (value.split(/[/:]/).pop() ?? '') === target
}

/** Session already touched this skill: the agent loaded it (`skill_view`),
 *  edited it (`skill_manage`), or the user sent its `/name` command.
 *  Exported for tests. */
export function skillTouchedInMessages(skillName: string, messages: readonly ChatMessage[]): boolean {
  const slash = new RegExp(`^/${escape(skillName.toLowerCase())}(?:\\s|$)`)

  for (const message of messages) {
    for (const part of message.parts) {
      if (
        part.type === 'tool-call' &&
        SKILL_TOOL_NAMES.has(part.toolName) &&
        argNamesSkill(skillArgName(part), skillName)
      ) {
        return true
      }

      if (
        message.role === 'user' &&
        part.type === 'text' &&
        typeof part.text === 'string' &&
        slash.test(part.text.trimStart().toLowerCase())
      ) {
        return true
      }
    }
  }

  return false
}

/** This session's transcript, wherever it currently lives: the per-runtime
 *  mirror ($sessionStates) for any session, falling back to the active view
 *  ($messages) for a session whose mirror hasn't populated yet. */
function sessionTranscript(sessionId: string | null): readonly ChatMessage[] {
  const cached = sessionId ? $sessionStates.get()[sessionId]?.messages : undefined

  if (cached?.length) {
    return cached
  }

  if (sessionId && sessionId === $activeSessionId.get()) {
    return $messages.get()
  }

  return cached ?? []
}

function toSuggestion(name: string): ComposerSuggestion {
  const copy = (key: string, ...args: unknown[]) => translateNow(`composer.skillSuggestions.${key}`, ...args)

  return {
    doneLabel: copy('done', name),
    doneTip: copy('doneTip'),
    icon: 'zap',
    id: name,
    invoke: async () => {
      // Prefix, don't replace: the user's words stay theirs. mode:'prefix'
      // puts the command where slash routing expects it — line start.
      requestComposerInsert(`/${name} `, { mode: 'prefix' })
      requestComposerFocus()
    },
    label: copy('label', name),
    provider: 'skill',
    tip: copy('tip', name),
    workingLabel: copy('label', name),
    workingTip: copy('tip', name)
  }
}

registerDraftProvider('skill', async ({ sessionId, text }) => {
  const trimmed = text.trimStart()

  // Already a slash command (possibly ours from a previous click) — stand
  // down instead of offering to double-prefix.
  if (trimmed.startsWith('/')) {
    return []
  }

  // Folding the haystack here (not only inside triggerPattern) is what makes the fold symmetric: the
  // draft is folded once for every matcher that reads it. It is a no-op for Latin names and for any
  // name that carries no Arabic, so the name match keeps its exact former behaviour.
  const haystack = foldArabic(stripSlashTokens(text).toLowerCase())
  const cwd = $currentCwd.get()
  const skills = await loadIndex()

  const matched = skills.filter(
    skill =>
      ((skill.pattern !== null && skillHit(skill.pattern, haystack)) ||
        skill.triggerPatterns.some(pattern => skillHit(pattern, haystack))) &&
      !collidesWithWorkspace(skill.name, cwd)
  )

  if (matched.length === 0) {
    return []
  }

  // Transcript scan only for actual matches — the common no-match sample
  // never pays for it.
  const transcript = sessionTranscript(sessionId)

  return matched.filter(skill => !skillTouchedInMessages(skill.name, transcript)).map(skill => toSuggestion(skill.name))
})
