/**
 * Describe-a-bot drafting: the `profiles.draft` wire shape plus the pure
 * helpers the New Agent and Edit Profile dialogs share. The model call lives
 * on the backend; nothing here writes until the dialog's own Create / Save.
 */

export interface DraftSection {
  key: string
  text: string
  title: string
}

export interface ProfileDraftResult {
  description: string
  dropped: { kind: string; name: string }[]
  mcp_servers: string[]
  name: string
  ok: boolean
  previous?: { description: string; sections: DraftSection[]; soul: string; title: string } | null
  reason: string
  sections: DraftSection[]
  skills: string[]
  soul: string
  title: string
  toolsets: string[]
}

export type DraftRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

/** The draft as `profiles.draft` takes it back for a partial regenerate. */
export function draftAsCurrent(draft: ProfileDraftResult): Record<string, unknown> {
  return {
    name: draft.name,
    title: draft.title,
    description: draft.description,
    sections: Object.fromEntries(draft.sections.map(s => [s.key, s.text])),
    skills: draft.skills,
    toolsets: draft.toolsets,
    mcp_servers: draft.mcp_servers
  }
}

/** A Regenerate reply applied to the LATEST draft: only the requested section changes, so a
 *  slower reply never puts back another section's older text. */
export function mergeDraftPart(current: ProfileDraftResult, reply: ProfileDraftResult, only: string): ProfileDraftResult {
  const text = reply.sections.find(s => s.key === only)?.text

  return text === undefined
    ? current
    : { ...current, sections: current.sections.map(s => (s.key === only ? { ...s, text } : s)) }
}

/** SOUL.md from the reviewed sections — same layout the backend composes. */
export function composeDraftSoul(heading: string, sections: DraftSection[]): string {
  const parts = heading.trim() ? [`# ${heading.trim()}`] : []

  for (const s of sections) {
    if (s.text.trim()) {
      parts.push(`## ${s.title}\n\n${s.text.trim()}`)
    }
  }

  return parts.length ? parts.join('\n\n') + '\n' : ''
}

/** The SOUL a new bot is created with: a hand-written one wins, else the
 *  reviewed draft's sections, else '' (the default identity template). */
export function reviewedSoul(custom: string, draft: null | ProfileDraftResult, title: string, slug: string): string {
  return custom || (draft ? composeDraftSoul(title || slug, draft.sections) : '')
}

/** A manual field (Title / Description) and the value it held before a proposal was accepted. */
export interface DecidedField {
  held?: string
  value: string
}

/** The field after a "Describe a change" decision. Only that field's own decision touches it:
 *  accepting writes the proposal and remembers what the field held; un-accepting puts that back
 *  unless the field was edited since. A decision on another part leaves typed text alone. */
export function decideField(field: DecidedField, prev: string | undefined, next: string | undefined): DecidedField {
  if (next === prev) {
    return field
  }

  if (next !== undefined) {
    return { held: prev === undefined ? field.value : field.held, value: next }
  }

  return { value: field.value === prev ? (field.held ?? field.value) : field.value }
}

const escapeRe = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Apply accepted sections to a saved SOUL.md in place: a section that exists
 *  keeps its position, a new one is appended, everything else (the heading,
 *  custom sections, the teammate protocol) is left byte-for-byte. */
export function mergeSoulSections(soul: string, accepted: DraftSection[]): string {
  let out = soul

  for (const s of accepted) {
    // Same heading syntax the backend's split_soul reads, closing hashes included.
    const heading = new RegExp(`^##[ \\t]+${escapeRe(s.title)}[ \\t]*#*[ \\t]*$`, 'im')
    const match = heading.exec(out)
    const body = s.text.trim()

    if (!match) {
      out = body ? `${out.trimEnd()}\n\n## ${s.title}\n\n${body}\n` : out

      continue
    }

    const start = match.index + match[0].length
    const next = /^##[ \t]+/m.exec(out.slice(start))
    const end = next ? start + next.index : out.length
    const tail = out.slice(end)

    out = body
      ? `${out.slice(0, start)}\n\n${body}\n${tail ? '\n' + tail : ''}`
      : `${out.slice(0, match.index)}${tail}`
  }

  return out
}

interface DescribedCaps {
  mcp_servers?: { enabled: boolean; name: string }[]
  skills?: { enabled: boolean; name: string }[]
  toolsets?: { enabled: boolean; name: string }[]
}

const leaf = (name: string) => name.split('/').pop() || name

/** Turn the draft's suggestions ON for a just-created profile without turning
 *  anything else off: `profiles.configure` replaces whole lists, so the new
 *  lists are the profile's current state plus the picks. */
export async function applyDraftCapabilities(
  request: DraftRequest,
  name: string,
  picks: null | Pick<ProfileDraftResult, 'mcp_servers' | 'skills' | 'toolsets'>
): Promise<void> {
  if (!picks || (!picks.skills.length && !picks.toolsets.length && !picks.mcp_servers.length)) {
    return
  }

  const caps = await request<DescribedCaps>('profiles.describe', { name })
  const skillPicks = new Set(picks.skills.map(leaf))

  const on = (rows: { enabled: boolean; name: string }[] | undefined, extra: string[]) => [
    ...new Set([...(rows || []).filter(r => r.enabled).map(r => r.name), ...extra])
  ]

  await request('profiles.configure', {
    name,
    ...(picks.skills.length
      ? { disabled_skills: (caps.skills || []).filter(s => !s.enabled && !skillPicks.has(s.name)).map(s => s.name) }
      : {}),
    ...(picks.toolsets.length ? { enabled_toolsets: on(caps.toolsets, picks.toolsets) } : {}),
    ...(picks.mcp_servers.length ? { enabled_mcp_servers: on(caps.mcp_servers, picks.mcp_servers) } : {})
  })
}
