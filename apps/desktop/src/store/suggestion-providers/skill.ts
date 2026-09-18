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

interface SkillIndexEntry {
  name: string
  pattern: RegExp
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
    .filter(skill => skill.enabled && skill.name.length >= MIN_NAME_LENGTH)
    .map(skill => ({
      name: skill.name,
      pattern: skillPattern(skill.name)
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
 *  segment of a qualified form (`github/mock-skill`, `plugin:skill`). */
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

  const haystack = text.toLowerCase()
  const cwd = $currentCwd.get()
  const skills = await loadIndex()
  const matched = skills.filter(skill => skillHit(skill.pattern, haystack) && !collidesWithWorkspace(skill.name, cwd))

  if (matched.length === 0) {
    return []
  }

  // Transcript scan only for actual matches — the common no-match sample
  // never pays for it.
  const transcript = sessionTranscript(sessionId)

  return matched.filter(skill => !skillTouchedInMessages(skill.name, transcript)).map(skill => toSuggestion(skill.name))
})
