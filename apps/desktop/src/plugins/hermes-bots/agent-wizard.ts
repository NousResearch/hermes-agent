/**
 * Conversational bot creation — the "describe it and I'll build it" wizard.
 *
 * The Grok-parity gap: Bot Mode's dialogs (name/SOUL/avatar) are powerful but
 * demand the user already know what they want. This module flips it: the user
 * describes the agent in one sentence, a one-shot LLM turn drafts the fields,
 * the user reviews and edits before anything is created. Nothing auto-saves —
 * the dialog stays the authority; the LLM is a pre-filler.
 *
 * The turn runs through the SAME `cli.exec` seam every other bot operation
 * uses (delete, rename), via `hermes -z` (single-shot mode, stdout = the final
 * response). No new RPC, no renderer LLM access beyond what the plugin already
 * has. Output is JSON with a strict schema parsed defensively — a malformed
 * or refused generation degrades to an empty draft, never to a broken dialog.
 */
import type { CliExecResult } from './profile-ops'

export interface AgentDraft {
  name: string
  title: string
  description: string
  soul: string
  color: null | string
  /** Suggested roster members for a group chat, by bot name (may be empty). */
  groupMembers: string[]
  /** True when the description reads like a group request ("um grupo que..."). */
  suggestsGroup: boolean
}

const EMPTY: AgentDraft = {
  color: null,
  description: '',
  groupMembers: [],
  name: '',
  soul: '',
  suggestsGroup: false,
  title: ''
}

const DRAFT_SCHEMA_HINT = `Respond with ONLY a JSON object (no markdown fence, no prose) with exactly these keys:
{
  "name": "short slug, lowercase, hyphens, max 24 chars",
  "title": "human role, max 48 chars",
  "description": "one-sentence mission, max 140 chars",
  "soul": "system-prompt persona, 2-4 sentences, direct tone, first person, defines behavior boundaries",
  "color": "one of: #38bdf8 #8b5cf6 #f97316 #22c55e #ec4899 #eab308 #06b6d4 #ef4444 — best contrast for the role",
  "groupMembers": [ bot names that should join this agent in a group chat, [] if none ],
  "suggestsGroup": true/false
}`

/** Extract the first JSON object from a response that may carry prose or a
 *  fenced block around it. */
export function parseDraftJson(raw: string): Partial<AgentDraft> | null {
  if (!raw) {
    return null
  }

  const start = raw.indexOf('{')
  const end = raw.lastIndexOf('}')

  if (start === -1 || end <= start) {
    return null
  }

  try {
    const parsed = JSON.parse(raw.slice(start, end + 1)) as Record<string, unknown>

    return typeof parsed === 'object' && parsed !== null ? (parsed as Partial<AgentDraft>) : null
  } catch {
    return null
  }
}

const SLUG_RE = /^[a-z0-9][a-z0-9-]{0,23}$/

const STRINGS: readonly (keyof AgentDraft)[] = ['name', 'title', 'description', 'soul']

const COLORS = new Set([
  '#38bdf8', '#8b5cf6', '#f97316', '#22c55e', '#ec4899', '#eab308', '#06b6d4', '#ef4444'
])

/** Coerce an LLM draft into the dialog's shape: truncate to field limits,
 *  slugify the name, whitelist the color, string-array the group members.
 *  Unparseable fields become empty strings — the user edits from there. */
export function coerceDraft(parsed: Partial<AgentDraft> | null): AgentDraft {
  if (!parsed) {
    return { ...EMPTY }
  }

  const rawName = String(parsed.name ?? '').trim().toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-+|-+$/g, '')

  const draft: AgentDraft = {
    color: typeof parsed.color === 'string' && COLORS.has(parsed.color.toLowerCase()) ? parsed.color.toLowerCase() : null,
    description: String(parsed.description ?? '').slice(0, 200),
    groupMembers: Array.isArray(parsed.groupMembers)
      ? parsed.groupMembers.filter((m): m is string => typeof m === 'string').slice(0, 8)
      : [],
    name: SLUG_RE.test(rawName) ? rawName : '',
    soul: String(parsed.soul ?? '').slice(0, 2000),
    suggestsGroup: parsed.suggestsGroup === true,
    title: String(parsed.title ?? '').slice(0, 80)
  }

  for (const key of STRINGS) {
    if (typeof draft[key] !== 'string') {
      ;(draft as unknown as Record<string, unknown>)[key] = ''
    }
  }

  return draft
}

/** Build the one-shot prompt for the draft turn. Kept pure for tests. */
export function draftPrompt(userIntent: string, rosterNames: readonly string[]): string {
  const roster = rosterNames.length > 0 ? `\n\nExisting bots (possible group members): ${rosterNames.join(', ')}` : ''

  return `You are configuring a new AI agent for a desktop app's Bot Mode. The user describes what they want; you draft the agent's configuration.

User description: """${userIntent.trim().slice(0, 500)}"""${roster}

${DRAFT_SCHEMA_HINT}`
}

/** Run the draft turn via the same cli.exec seam as every other bot op.
 *  Returns the coerced draft, or EMPTY on any failure — the dialog then just
 *  stays on the manual form (degradation, never breakage). */
export async function draftAgentFromIntent(
  request: (method: string, params: Record<string, unknown>) => Promise<CliExecResult>,
  userIntent: string,
  rosterNames: readonly string[] = []
): Promise<AgentDraft> {
  if (!userIntent.trim()) {
    return { ...EMPTY }
  }

  let output = ''

  try {
    const result: CliExecResult = await request('cli.exec', {
      argv: ['-z', draftPrompt(userIntent, rosterNames)]
    })

    output = result?.output ?? ''
  } catch {
    return { ...EMPTY }
  }

  return coerceDraft(parseDraftJson(output))
}
