import { atom } from 'nanostores'

import { baseName, isProjectCwd } from '@/app/chat/sidebar/workspace-groups'
import { persistString, storedBoolean, storedString } from '@/lib/storage'
import type { SessionInfo } from '@/types/hermes'

// hc-517 — desktop "project" picker. A project is NOT a new entity: it's the
// working directory (git root) a session runs in, exactly what the sidebar
// already groups sessions by (see workspace-groups.ts). This store only adds a
// persisted MRU list of directories the user has picked/created, so the new-
// conversation chip can offer "recent projects" + search. The chosen path flows
// through the existing changeSessionCwd → session.create({ cwd }) path.

export interface RecentProject {
  path: string
  name: string
  lastUsedAt: number
}

const STORAGE_KEY = 'hermes.desktop.recent-projects'
// Feature flag — default ON. Flip to 'false' in localStorage to fall back to
// the plain new-conversation flow with no chip (task: "异常时回退无 chip").
export const PROJECT_PICKER_FLAG_KEY = 'hermes.desktop.feature.project-picker'
const MAX_RECENT_PROJECTS = 24

/** Whether the project picker chip is enabled (localStorage flag, default on). */
export function isProjectPickerEnabled(): boolean {
  return storedBoolean(PROJECT_PICKER_FLAG_KEY, true)
}

/** Trim + strip trailing separators so `/a/b` and `/a/b/` are one key. Keeps a
 *  lone root ("/" or "C:\") intact. The picker only ever stores absolute paths
 *  the main process resolved, so no `~`/relative handling is needed here. */
export function normalizeProjectPath(path: null | string | undefined): string {
  const trimmed = (path ?? '').trim()

  if (!trimmed) {
    return ''
  }

  const stripped = trimmed.replace(/[/\\]+$/, '')

  return stripped || trimmed
}

/** Human label for a project path — its final path segment. */
export function projectDisplayName(path: string): string {
  return baseName(path) || path
}

/** Parse the persisted JSON blob into a clean, deduped list (bad rows dropped). */
export function parseRecentProjects(raw: null | string): RecentProject[] {
  if (!raw) {
    return []
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(raw)
  } catch {
    return []
  }

  if (!Array.isArray(parsed)) {
    return []
  }

  const out: RecentProject[] = []
  const seen = new Set<string>()

  for (const item of parsed) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const record = item as Record<string, unknown>
    const path = normalizeProjectPath(typeof record.path === 'string' ? record.path : '')

    if (!path || seen.has(path)) {
      continue
    }

    const lastUsedAt =
      typeof record.lastUsedAt === 'number' && Number.isFinite(record.lastUsedAt) ? record.lastUsedAt : 0

    const name =
      typeof record.name === 'string' && record.name.trim() ? record.name.trim() : projectDisplayName(path)

    seen.add(path)
    out.push({ path, name, lastUsedAt })
  }

  return out
}

/** Most-recent-first, name-tiebroken. Pure — returns a new array. */
function sortRecentProjects(list: RecentProject[]): RecentProject[] {
  return [...list].sort((a, b) => b.lastUsedAt - a.lastUsedAt || a.name.localeCompare(b.name))
}

/** Upsert a project (bump to `now`, dedupe by path), capped to the MRU limit. */
export function upsertRecentProject(
  list: RecentProject[],
  path: string,
  now: number,
  name?: string
): RecentProject[] {
  const normalized = normalizeProjectPath(path)

  if (!normalized) {
    return list
  }

  const entry: RecentProject = {
    path: normalized,
    name: name?.trim() || projectDisplayName(normalized),
    lastUsedAt: now
  }

  const rest = list.filter(project => project.path !== normalized)

  return sortRecentProjects([entry, ...rest]).slice(0, MAX_RECENT_PROJECTS)
}

/** Project directories discovered from loaded sessions — the same cwd → project
 *  mapping the sidebar uses. Home-dir / empty cwds are plain chats, not
 *  projects, so they're skipped. Powers a populated list on first run before
 *  the user has explicitly picked anything. */
export function sessionProjectEntries(sessions: SessionInfo[]): RecentProject[] {
  const byPath = new Map<string, RecentProject>()

  for (const session of sessions) {
    const path = normalizeProjectPath(session.cwd)

    if (!path || !isProjectCwd(path)) {
      continue
    }

    const lastUsedAt = typeof session.started_at === 'number' ? session.started_at : 0
    const existing = byPath.get(path)

    if (!existing || lastUsedAt > existing.lastUsedAt) {
      byPath.set(path, { path, name: projectDisplayName(path), lastUsedAt })
    }
  }

  return [...byPath.values()]
}

/** Merge the persisted MRU (authoritative for explicit names) with session-
 *  derived entries (discovery), deduped by path, keeping the freshest use time.
 *  Persisted iterates last so an explicit custom name wins the collision. */
export function mergeRecentProjects(persisted: RecentProject[], derived: RecentProject[]): RecentProject[] {
  const byPath = new Map<string, RecentProject>()

  for (const project of [...derived, ...persisted]) {
    const existing = byPath.get(project.path)

    if (!existing) {
      byPath.set(project.path, project)
    } else {
      byPath.set(project.path, {
        path: project.path,
        name: project.name || existing.name,
        lastUsedAt: Math.max(existing.lastUsedAt, project.lastUsedAt)
      })
    }
  }

  return sortRecentProjects([...byPath.values()])
}

/** Filter by a space-separated query, matching every term against name+path. */
export function filterRecentProjects(list: RecentProject[], query: string): RecentProject[] {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean)

  if (terms.length === 0) {
    return list
  }

  return list.filter(project => {
    const haystack = `${project.name} ${project.path}`.toLowerCase()

    return terms.every(term => haystack.includes(term))
  })
}

function persist(list: RecentProject[]): void {
  persistString(STORAGE_KEY, list.length ? JSON.stringify(list) : null)
}

function loadRecentProjects(): RecentProject[] {
  return sortRecentProjects(parseRecentProjects(storedString(STORAGE_KEY)))
}

// Live store: the persisted MRU only. The picker merges in session-derived
// entries at render time (those are not persisted — sessions are their own
// source of truth), so deleting a session naturally prunes discovery rows while
// explicit picks survive.
export const $recentProjects = atom<RecentProject[]>(loadRecentProjects())

/** Record an explicit pick/create — persists + bumps it to the top. */
export function recordRecentProject(path: string, name?: string, now: number = Date.now()): void {
  const next = upsertRecentProject($recentProjects.get(), path, now, name)
  $recentProjects.set(next)
  persist(next)
}

/** Forget a persisted project (session-derived rows reappear from $sessions). */
export function removeRecentProject(path: string): void {
  const normalized = normalizeProjectPath(path)
  const next = $recentProjects.get().filter(project => project.path !== normalized)
  $recentProjects.set(next)
  persist(next)
}
