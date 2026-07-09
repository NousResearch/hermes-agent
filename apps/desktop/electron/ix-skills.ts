/**
 * IX Agency user-level skills — create SKILL.md playbooks right from the
 * desktop, then publish them to the whole org:
 *
 *   1. DRAFT (user-level): saved as ~/.hermes/skills/ix-user/<slug>/SKILL.md.
 *      Local Hermes picks the folder up like any other skill, and the native
 *      Copilot tab can inject it as an active skill — no restart, no portal.
 *   2. PUBLISH (global): POST to the admin portal's /api/admin/skills using
 *      the signed-in OTP session (main process cookie jar). The portal stores
 *      team skills in Sanity, so the skill immediately shows up on the web
 *      admin Skills.md page and in every teammate's copilot.
 *
 * Ships starter TEMPLATES so the team sees what a good skill looks like
 * (goal / when to use / tools to call / output format) before writing one.
 *
 * Pure logic + node fs with injectable fetch — unit-tested in
 * ix-skills.test.ts without electron.
 */
import fs from 'node:fs'
import path from 'node:path'

export interface IxUserSkill {
  /** Folder slug under skills/ix-user/ — doubles as the skill name. */
  id: string
  title: string
  description: string
  /** Full SKILL.md body (markdown, without the frontmatter). */
  content: string
  updatedAt: number
  /** Portal document id (adminSkill.<slug>) once published — global. */
  publishedId: null | string
}

export interface IxSkillTemplate {
  id: string
  title: string
  description: string
  content: string
}

export const USER_SKILL_DIRNAME = 'ix-user'

const SKILL_TITLE_MAX = 120

const SKILL_CONTENT_MAX = 20_000

export function userSkillsDir(hermesHome: string): string {
  return path.join(hermesHome, 'skills', USER_SKILL_DIRNAME)
}

/* ── slug + frontmatter helpers ────────────────────────────────────────────── */

export function skillSlug(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
}

function yamlQuote(value: string): string {
  return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, ' ')}"`
}

function renderSkillMd(skill: IxUserSkill): string {
  const lines = [
    '---',
    `name: ${skill.id}`,
    `description: ${yamlQuote(skill.description || skill.title)}`,
    'metadata:',
    '  ix:',
    `    title: ${yamlQuote(skill.title)}`,
    `    updatedAt: ${skill.updatedAt}`,
    ...(skill.publishedId ? [`    publishedId: ${yamlQuote(skill.publishedId)}`] : []),
    '---'
  ]

  return `${lines.join('\n')}\n\n${skill.content.trim()}\n`
}

function unquote(raw: string): string {
  const trimmed = raw.trim()

  if (trimmed.startsWith('"') && trimmed.endsWith('"') && trimmed.length >= 2) {
    return trimmed.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\')
  }

  return trimmed
}

/** Tolerant line-based parse of the frontmatter this module writes. */
export function parseSkillMd(id: string, raw: string): IxUserSkill {
  const match = /^---\n([\s\S]*?)\n---\n?/.exec(raw)

  const skill: IxUserSkill = {
    id,
    title: id,
    description: '',
    content: raw.trim(),
    updatedAt: 0,
    publishedId: null
  }

  if (!match) {
    return skill
  }

  skill.content = raw.slice(match[0].length).trim()

  for (const line of match[1].split('\n')) {
    const kv = /^\s*([A-Za-z]+):\s*(.*)$/.exec(line)

    if (!kv) {
      continue
    }

    const [, key, value] = kv

    if (key === 'description') {
      skill.description = unquote(value)
    } else if (key === 'title') {
      skill.title = unquote(value)
    } else if (key === 'publishedId') {
      skill.publishedId = unquote(value) || null
    } else if (key === 'updatedAt') {
      skill.updatedAt = Number(value) || 0
    }
  }

  return skill
}

/* ── user-level store (folders under ~/.hermes/skills/ix-user/) ───────────── */

export function listUserSkills(dir: string): IxUserSkill[] {
  let entries: fs.Dirent[]

  try {
    entries = fs.readdirSync(dir, { withFileTypes: true })
  } catch {
    return []
  }

  const skills: IxUserSkill[] = []

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue
    }

    try {
      const raw = fs.readFileSync(path.join(dir, entry.name, 'SKILL.md'), 'utf8')

      skills.push(parseSkillMd(entry.name, raw))
    } catch {
      // Folder without a readable SKILL.md — not a skill.
    }
  }

  return skills.sort((a, b) => b.updatedAt - a.updatedAt)
}

export function saveUserSkill(
  dir: string,
  input: { id?: null | string; title: string; description?: string; content: string }
): IxUserSkill {
  const title = String(input.title || '').trim()
  const content = String(input.content || '').trim()

  if (!title) {
    throw new Error('A skill needs a title.')
  }

  if (title.length > SKILL_TITLE_MAX) {
    throw new Error(`Title is too long (max ${SKILL_TITLE_MAX} characters).`)
  }

  if (!content) {
    throw new Error('A skill needs SKILL.md content — start from a template.')
  }

  if (content.length > SKILL_CONTENT_MAX) {
    throw new Error(`Content is too long (max ${SKILL_CONTENT_MAX} characters).`)
  }

  const id = input.id?.trim() || skillSlug(title)

  if (!id) {
    throw new Error('Could not derive a folder name from the title.')
  }

  // Editing keeps whatever publish state the skill already had.
  const existing = readUserSkill(dir, id)

  const skill: IxUserSkill = {
    id,
    title,
    description: String(input.description || '').trim(),
    content,
    updatedAt: Date.now(),
    publishedId: existing?.publishedId ?? null
  }

  fs.mkdirSync(path.join(dir, id), { recursive: true })
  fs.writeFileSync(path.join(dir, id, 'SKILL.md'), renderSkillMd(skill), 'utf8')

  return skill
}

export function readUserSkill(dir: string, id: string): IxUserSkill | null {
  try {
    return parseSkillMd(id, fs.readFileSync(path.join(dir, id, 'SKILL.md'), 'utf8'))
  } catch {
    return null
  }
}

export function deleteUserSkill(dir: string, id: string): boolean {
  const target = path.join(dir, id)

  if (!id || !fs.existsSync(path.join(target, 'SKILL.md'))) {
    return false
  }

  fs.rmSync(target, { recursive: true, force: true })

  return true
}

function markPublished(dir: string, skill: IxUserSkill, publishedId: string): IxUserSkill {
  const updated: IxUserSkill = { ...skill, publishedId, updatedAt: Date.now() }

  fs.mkdirSync(path.join(dir, skill.id), { recursive: true })
  fs.writeFileSync(path.join(dir, skill.id, 'SKILL.md'), renderSkillMd(updated), 'utf8')

  return updated
}

/* ── publish to the portal's global Skills.md API ──────────────────────────── */

/** POST the skill to <portalUrl>/api/admin/skills through the signed-in
 *  portal session. Re-publishing an already-published skill updates the same
 *  portal document (its adminSkill.<slug> id is kept in the frontmatter). */
export async function publishUserSkill(
  dir: string,
  id: string,
  portalUrl: string,
  fetchImpl: typeof fetch,
  updatedBy?: string
): Promise<IxUserSkill> {
  const skill = readUserSkill(dir, id)

  if (!skill) {
    throw new Error(`Skill "${id}" not found under skills/${USER_SKILL_DIRNAME}/.`)
  }

  const url = new URL('/api/admin/skills', portalUrl).toString()

  const response = await fetchImpl(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...(skill.publishedId ? { id: skill.publishedId } : {}),
      label: skill.title,
      blurb: skill.description || undefined,
      content: skill.content,
      ...(updatedBy ? { updatedBy } : {})
    }),
    signal: AbortSignal.timeout(15_000)
  })

  let payload: { error?: string; id?: string; ok?: boolean } = {}

  try {
    payload = (await response.json()) as typeof payload
  } catch {
    payload = {}
  }

  if (!response.ok || !payload.ok || !payload.id) {
    throw new Error(payload.error || `Portal rejected the skill (HTTP ${response.status}).`)
  }

  return markPublished(dir, skill, payload.id)
}

/* ── starter templates (pre-populated prompts the team can build on) ───────── */

export const IX_SKILL_TEMPLATES: IxSkillTemplate[] = [
  {
    id: 'blank',
    title: 'Blank skill',
    description: 'The bare SKILL.md shape — goal, when to use, steps, output.',
    content: `# Skill: <what this does in one line>

## Goal
One sentence: the outcome this skill produces.

## When to use
- The situations where the copilot should reach for this skill.

## Steps
1. Which MCP tools to call, in order (e.g. \`admin_call_mcp\` tileId "notifuse").
2. What to check or compute from the results.
3. What to do when something looks wrong.

## Output format
Describe the exact shape of the answer (table, bullets, one-line summary).
`
  },
  {
    id: 'weekly-service-report',
    title: 'Weekly service report',
    description: 'Pull one service’s key numbers into a Monday-morning summary.',
    content: `# Skill: Weekly <service> report

## Goal
A Monday-morning summary of <service> health for the team channel.

## When to use
When asked for "the weekly <service> report" or "how did <service> do last week".

## MCPs used
notifuse (email volume/opens), chatwoot (support load), stripe (revenue),
grafana (alerts) — swap in the service you're reporting on.

## Steps
1. Call each service MCP (directly, or via \`admin_call_mcp\`) for last week's
   headline metrics — volume, failures, spend.
2. Compare to the previous week; flag anything ±20%.
3. Pull open incidents/alerts from grafana for every service touched.

## Output format
- **Headline:** one sentence (up / flat / down and why).
- Table: metric | last week | prev week | Δ.
- Bullets: anything that needs a human this week.
`
  },
  {
    id: 'campaign-launch-runbook',
    title: 'Campaign launch runbook',
    description: 'Coordinated email + WhatsApp + social send with checks between steps.',
    content: `# Skill: Campaign launch runbook

## Goal
Launch a coordinated announcement across email (notifuse), WhatsApp (openbsp)
and social (postiz) — with a human confirmation before anything sends.

## When to use
When asked to "announce", "launch a campaign" or "send the update everywhere".

## Steps
1. Draft the copy once; adapt per channel (email long, WhatsApp short, social hooky).
2. SHOW the drafts and target audiences and STOP for confirmation — sending is a
   write action and always needs the confirm gate.
3. After approval: notifuse campaign → openbsp broadcast → postiz scheduled posts.
4. Report back with each channel's id/link and scheduled times.

## Output format
Checklist with per-channel status (draft → approved → sent/scheduled) and links.
`
  },
  {
    id: 'incident-triage',
    title: 'Incident triage',
    description: 'From an alert to a first diagnosis using grafana + service MCPs.',
    content: `# Skill: Incident triage

## Goal
Turn "something is wrong with <service>" into a first diagnosis and a
recommended next step in under five minutes.

## When to use
When an alert fires or someone reports errors/slowness on a platform service.

## MCPs used
grafana (alerts + PromQL) plus the affected service's own MCP — e.g.
nakama (game ops), telnyx (SMS delivery), n8n (workflow runs).

## Steps
1. grafana: list firing alerts, then query the service's error-rate and latency
   panels for the last 6h.
2. The service's own MCP: recent failures/logs if it exposes them.
3. Correlate: deploy? traffic spike? dependency down?
4. Recommend ONE next action (rollback, scale, page owner) — do not execute
   writes without confirmation.

## Output format
- **What's happening:** one sentence.
- **Evidence:** 2-4 bullets with numbers.
- **Recommended action:** one bullet, tagged read-only or needs-approval.
`
  }
]
