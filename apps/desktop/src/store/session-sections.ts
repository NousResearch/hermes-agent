/**
 * SESSION SECTIONS — folders the user makes, for the Sessions list.
 *
 * The sidebar already has ONE axis: the topology's. Workspace grouping
 * (`agentsGroupedByWorkspace`) answers "where did this session run", and pins
 * answer "is this one of the few I keep coming back to". Neither answers the
 * question you are asking when you want six sessions filed under "Billing
 * work" regardless of which repo each ran in.
 *
 * So this is a second axis that composes with the first, and it is the same
 * model the Bot roster already ships (`plugins/hermes-bots/user-sections.ts`):
 *
 *   * Membership lives on the SESSION (its id in `sections`/`membership`),
 *     never as a member list on the section. A session can only be in one
 *     place, deleting a section cannot orphan anybody, and a session deleted
 *     out from under a section leaves nothing dangling to render.
 *   * "Unassigned" is not a section. It is whatever is left, always drawn
 *     last, and it is where members of a deleted section land. With no
 *     sections at all the list renders exactly as it did before this existed.
 *
 * Scope: sections are per PROFILE (the key below), because the session ids in
 * them only mean anything to the backend that served them. Moving a profile's
 * sections to another connection would point them at ids that were never
 * theirs. Both the rename and drop families are exported for
 * `store/session-states.ts` to call alongside its own profile-keyed state.
 *
 * Pure model + session atoms. No JSX — the sidebar composes it.
 */

import { atom } from 'nanostores'

import { readJson, writeKey } from '@/lib/storage'

import { $activeGatewayProfile } from '@/store/profile'

/** Where members of a deleted section land, and where every session starts. */
export const UNASSIGNED_SESSION_KEY = 'section:unassigned'

const SESSION_SECTIONS_KEY = 'hermes.desktop.sessionSections'

export interface SessionSection {
  id: string
  name: string
}

/** One profile's folders: the records, in display order, plus the membership
 *  map session id → section id. Membership is stored beside the sections so a
 *  rename or delete is ONE write, never a partial state on disk. */
interface ProfileSections {
  sections: SessionSection[]
  membership: Record<string, string>
}

type SectionsByProfile = Record<string, ProfileSections>

const EMPTY: ProfileSections = { sections: [], membership: {} }

/** `[{ id, name }]` in display order, for the active profile. */
export const $sessionSections = atom<SessionSection[]>([])

/** session id → section id, for the active profile. Rows absent from this map
 *  are Unassigned. */
export const $sessionSectionMembership = atom<Record<string, string>>({})

/** Roster key of the session in flight during a drag. Session-only, cleared on
 *  dragend even when the drop lands outside any target — a stuck "dragging"
 *  state outlives the gesture and reads as a broken list. */
export const $draggingSession = atom<null | string>(null)

function profileKey(): string {
  const profile = String($activeGatewayProfile.get() || '').trim()

  return profile || 'default'
}

function readAll(): SectionsByProfile {
  const raw = readJson<unknown>(SESSION_SECTIONS_KEY)

  return raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as SectionsByProfile) : {}
}

export function normalizeSessionSections(value: unknown): SessionSection[] {
  if (!Array.isArray(value)) {
    return []
  }

  const seen = new Set<string>()
  const out: SessionSection[] = []

  for (const entry of value) {
    const id = String((entry as SessionSection)?.id || '').trim()
    const name = String((entry as SessionSection)?.name || '').trim()

    if (!id || !name || seen.has(id)) {
      continue
    }

    seen.add(id)
    out.push({ id, name })
  }

  return out
}

/** Keep only memberships that still point at a live section, and drop blank
 *  ids. A session id may appear once; a later duplicate wins, because the map
 *  it came from cannot hold two values for one key anyway. */
export function normalizeMembership(
  value: unknown,
  sections: SessionSection[]
): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const known = new Set(sections.map(s => s.id))
  const out: Record<string, string> = {}

  for (const [sessionId, sectionId] of Object.entries(value as Record<string, unknown>)) {
    const session = String(sessionId || '').trim()
    const section = String(sectionId || '').trim()

    if (session && section && known.has(section)) {
      out[session] = section
    }
  }

  return out
}

function readProfile(): ProfileSections {
  const all = readAll()
  const entry = all[profileKey()]

  if (!entry || typeof entry !== 'object') {
    return EMPTY
  }

  const sections = normalizeSessionSections(entry.sections)

  return { sections, membership: normalizeMembership(entry.membership, sections) }
}

function persistProfile(next: ProfileSections): void {
  const all = readAll()

  if (next.sections.length === 0 && Object.keys(next.membership).length === 0) {
    delete all[profileKey()]
  } else {
    all[profileKey()] = next
  }

  try {
    writeKey(SESSION_SECTIONS_KEY, JSON.stringify(all))
  } catch {
    // Storage is best-effort (see lib/storage): the atoms below still update,
    // so this window keeps working and the next write tries again.
  }

  $sessionSections.set(next.sections)
  $sessionSectionMembership.set(next.membership)
}

/** Read the persisted folders back for the active profile. Called when the
 *  sidebar mounts and after a profile switch. */
export function loadSessionSections(): void {
  const { sections, membership } = readProfile()

  $sessionSections.set(sections)
  $sessionSectionMembership.set(membership)
}

function newSectionId(): string {
  return `sec-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

/** Create a section and file `sessionIds` into it. Returns the new section, or
 *  null when the name is blank. */
export function createSessionSection(name: string, sessionIds: string[] = []): SessionSection | null {
  const clean = String(name || '').trim()

  if (!clean) {
    return null
  }

  const current = readProfile()
  const section: SessionSection = { id: newSectionId(), name: clean }
  const membership = { ...current.membership }

  for (const id of sessionIds) {
    const session = String(id || '').trim()

    if (session) {
      membership[session] = section.id
    }
  }

  persistProfile({ sections: [...current.sections, section], membership })

  return section
}

export function renameSessionSection(id: string, name: string): void {
  const clean = String(name || '').trim()

  if (!clean) {
    return
  }

  const current = readProfile()

  persistProfile({
    ...current,
    sections: current.sections.map(s => (s.id === id ? { ...s, name: clean } : s))
  })
}

/** File `sessionIds` into `sectionId` (null = Unassigned). Ids not in a live
 *  section land in Unassigned rather than being dropped. */
export function moveSessionsToSection(sessionIds: string[], sectionId: null | string): void {
  const current = readProfile()
  const known = new Set(current.sections.map(s => s.id))
  const target = sectionId && known.has(sectionId) ? sectionId : null
  const membership = { ...current.membership }

  for (const id of sessionIds) {
    const session = String(id || '').trim()

    if (!session) {
      continue
    }

    if (target) {
      membership[session] = target
    } else {
      delete membership[session]
    }
  }

  persistProfile({ ...current, membership })
}

/**
 * Delete the section only. Its sessions are not deleted and not hidden — they
 * fall back to Unassigned, which is the whole reason membership lives on the
 * session rather than on the section. Returns an undo that puts the section
 * back in its slot and refiles the same sessions, so the delete needs no
 * confirmation.
 */
export function deleteSessionSection(id: string): { undo: () => void } {
  const current = readProfile()
  const index = current.sections.findIndex(s => s.id === id)
  const section = current.sections[index]
  const members = Object.entries(current.membership)
    .filter(([, sectionId]) => sectionId === id)
    .map(([sessionId]) => sessionId)

  const membership = { ...current.membership }

  for (const sessionId of members) {
    delete membership[sessionId]
  }

  persistProfile({ sections: current.sections.filter(s => s.id !== id), membership })

  return {
    undo: () => {
      if (!section) {
        return
      }

      const now = readProfile()
      const sections = now.sections.filter(s => s.id !== id)

      sections.splice(Math.min(index, sections.length), 0, section)

      const refiled = { ...now.membership }

      for (const sessionId of members) {
        refiled[sessionId] = id
      }

      persistProfile({ sections, membership: refiled })
    }
  }
}

/** The section a session is filed in, or null for Unassigned. */
export function sessionSectionId(sessionId: string): null | string {
  const id = String(sessionId || '').trim()

  if (!id) {
    return null
  }

  return $sessionSectionMembership.get()[id] ?? null
}

/** Sections in display order, each with the sessions filed into it — plus the
 *  trailing Unassigned bucket when it has any members. Rows are the caller's
 *  own (already sorted, already filtered); this only decides what goes where,
 *  and a session the caller did not hand over is simply not drawn. */
export function groupSessionsBySection<T extends { id: string }>(
  rows: T[],
  sections: SessionSection[],
  membership: Record<string, string>
): { id: null | string; name: null | string; rows: T[] }[] {
  const known = new Set(sections.map(s => s.id))
  const bySection = new Map<string, T[]>()
  const unassigned: T[] = []

  for (const row of rows || []) {
    const sectionId = membership[row?.id]

    if (sectionId && known.has(sectionId)) {
      bySection.set(sectionId, [...(bySection.get(sectionId) || []), row])
    } else {
      unassigned.push(row)
    }
  }

  const out = sections.map(section => ({
    id: section.id as null | string,
    name: section.name as null | string,
    rows: bySection.get(section.id) || []
  }))

  if (unassigned.length > 0) {
    out.push({ id: null, name: null, rows: unassigned })
  }

  return out
}

/** Drop every folder the profile owned. Called when a profile is removed, so
 *  its section ids do not outlive the sessions they named. */
export function dropSessionSectionsForProfile(profile: string): void {
  const key = String(profile || '').trim() || 'default'
  const all = readAll()

  if (!(key in all)) {
    return
  }

  delete all[key]

  try {
    writeKey(SESSION_SECTIONS_KEY, JSON.stringify(all))
  } catch {
    // Best-effort; a stale map entry is inert once its profile is gone.
  }

  if (key === profileKey()) {
    $sessionSections.set([])
    $sessionSectionMembership.set({})
  }
}

/** Carry a profile's folders across a rename, so the list does not come back
 *  empty (and pointed at a backend that no longer exists) after one. */
export function migrateSessionSectionsForProfile(oldProfile: string, newProfile: string): void {
  const from = String(oldProfile || '').trim() || 'default'
  const to = String(newProfile || '').trim() || 'default'

  if (from === to) {
    return
  }

  const all = readAll()
  const entry = all[from]

  if (!entry) {
    return
  }

  delete all[from]
  all[to] = entry

  try {
    writeKey(SESSION_SECTIONS_KEY, JSON.stringify(all))
  } catch {
    return
  }

  if (from === profileKey()) {
    loadSessionSections()
  }
}
