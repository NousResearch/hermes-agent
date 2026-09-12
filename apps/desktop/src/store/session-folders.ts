/**
 * Desktop session folders — the user's own grouping of the sidebar's sessions.
 *
 * A folder is desktop-local organization, NOT session state: filing a chat
 * changes nothing about the session (no cwd move, no backend write, no second
 * copy of the row) — the issue's "without changing the sessions themselves".
 * The renderer is therefore the authority for it, held in one localStorage
 * record keyed by PROFILE (the axis the session list itself switches on), so
 * one profile's folders can never group another profile's chats.
 *
 * Membership keys off the session's DURABLE stored id — the identity rows
 * resume with, the same one pins and the persisted drag order use.
 */

import { computed, type ReadableAtom } from 'nanostores'

import type { SessionInfo } from '@/hermes'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { $profileScope, ALL_PROFILES } from '@/store/profile'

export interface SessionFolder {
  collapsed: boolean
  id: string
  name: string
}

/** One profile's folders plus which session is filed where. */
export interface SessionFolderScope {
  /** stored session id -> folder id */
  filed: Record<string, string>
  folders: SessionFolder[]
}

export type SessionFolderStore = Record<string, SessionFolderScope>

const STORAGE_KEY = 'hermes.desktop.sessionFolders'

const EMPTY_SCOPE: SessionFolderScope = { filed: {}, folders: [] }

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/**
 * Coerce a persisted record into folders + membership, dropping what can't be
 * honoured: entries with no id, and — the case that matters — assignments
 * pointing at a folder that no longer exists. A stale assignment would hide the
 * session from BOTH the folder strip and the main list.
 */
function sanitizeScope(value: unknown): SessionFolderScope {
  if (!isRecord(value)) {
    return EMPTY_SCOPE
  }

  const folders: SessionFolder[] = Array.isArray(value.folders)
    ? value.folders.flatMap(entry =>
        isRecord(entry) && typeof entry.id === 'string' && entry.id
          ? [
              {
                collapsed: entry.collapsed === true,
                id: entry.id,
                name: typeof entry.name === 'string' ? entry.name : ''
              }
            ]
          : []
      )
    : []

  const known = new Set(folders.map(folder => folder.id))
  const filed: Record<string, string> = {}

  if (isRecord(value.filed)) {
    for (const [sessionId, folderId] of Object.entries(value.filed)) {
      if (sessionId && typeof folderId === 'string' && known.has(folderId)) {
        filed[sessionId] = folderId
      }
    }
  }

  return { filed, folders }
}

export function sanitizeSessionFolderStore(value: unknown): SessionFolderStore {
  if (!isRecord(value)) {
    return {}
  }

  return Object.fromEntries(Object.entries(value).map(([profile, scope]) => [profile, sanitizeScope(scope)]))
}

export const $sessionFolderStore = persistentAtom<SessionFolderStore>(
  STORAGE_KEY,
  {},
  Codecs.json(sanitizeSessionFolderStore)
)

/** The profile whose folders are live; `null` in the all-profiles view, where
 *  one folder list cannot describe a session list spanning profiles. */
export function sessionFolderProfile(): null | string {
  const scope = $profileScope.get()

  return scope === ALL_PROFILES ? null : scope
}

export const $sessionFolderScope: ReadableAtom<SessionFolderScope> = computed(
  [$sessionFolderStore, $profileScope],
  (store, scope) => (scope === ALL_PROFILES ? EMPTY_SCOPE : (store[scope] ?? EMPTY_SCOPE))
)

export const $sessionFolders: ReadableAtom<SessionFolder[]> = computed($sessionFolderScope, scope => scope.folders)

/** Whether this desktop can group sessions at all (the all-profiles view can't). */
export const $sessionFoldersAvailable: ReadableAtom<boolean> = computed($profileScope, scope => scope !== ALL_PROFILES)

export interface SessionFolderGroup {
  folder: SessionFolder
  sessions: SessionInfo[]
}

export interface SessionsByFolder {
  groups: SessionFolderGroup[]
  unfiled: SessionInfo[]
}

/**
 * Split a session list into its folders (in folder order) plus everything else
 * (in list order). With no folders this hands the list straight back, so the
 * ordinary flat rendering keeps its exact array identity.
 */
export function groupSessionsByFolder(sessions: SessionInfo[], scope: SessionFolderScope): SessionsByFolder {
  const { filed, folders } = scope

  if (!folders.length) {
    return { groups: [], unfiled: sessions }
  }

  const byFolder = new Map<string, SessionInfo[]>(folders.map(folder => [folder.id, []]))
  const unfiled: SessionInfo[] = []

  for (const session of sessions) {
    const bucket = byFolder.get(filed[session.id] ?? '')

    if (bucket) {
      bucket.push(session)
    } else {
      unfiled.push(session)
    }
  }

  return {
    groups: folders.map(folder => ({ folder, sessions: byFolder.get(folder.id) ?? [] })),
    unfiled
  }
}

let folderSeq = 0

const nextFolderId = () => `f_${Date.now().toString(36)}-${(folderSeq++).toString(36)}`

type ScopeMutation = (scope: SessionFolderScope) => SessionFolderScope

/** Apply a mutation to the live profile's scope. Writes only when the mutation
 *  actually changed something, so a no-op drop can't churn localStorage. */
function mutateScope(mutate: ScopeMutation): null | SessionFolderScope {
  const profile = sessionFolderProfile()

  if (!profile) {
    return null
  }

  const store = $sessionFolderStore.get()
  const current = store[profile] ?? EMPTY_SCOPE
  const next = mutate(current)

  if (next !== current) {
    $sessionFolderStore.set({ ...store, [profile]: next })
  }

  return next
}

export function createSessionFolder(name: string): null | SessionFolder {
  const trimmed = name.trim()

  if (!trimmed) {
    return null
  }

  const folder: SessionFolder = { collapsed: false, id: nextFolderId(), name: trimmed }

  return mutateScope(scope => ({ ...scope, folders: [...scope.folders, folder] })) ? folder : null
}

export function renameSessionFolder(folderId: string, name: string): void {
  const trimmed = name.trim()

  if (!trimmed) {
    return
  }

  mutateScope(scope => ({
    ...scope,
    folders: scope.folders.map(folder => (folder.id === folderId ? { ...folder, name: trimmed } : folder))
  }))
}

/** Deleting a folder returns its sessions to the main list; nothing about the
 *  sessions themselves is touched. */
export function deleteSessionFolder(folderId: string): void {
  mutateScope(scope => ({
    filed: Object.fromEntries(Object.entries(scope.filed).filter(([, id]) => id !== folderId)),
    folders: scope.folders.filter(folder => folder.id !== folderId)
  }))
}

export function toggleSessionFolderCollapsed(folderId: string): void {
  mutateScope(scope => ({
    ...scope,
    folders: scope.folders.map(folder =>
      folder.id === folderId ? { ...folder, collapsed: !folder.collapsed } : folder
    )
  }))
}

/** File a session into a folder, or back into the main list with `null`. */
export function fileSessionInFolder(sessionId: string, folderId: null | string): void {
  const id = sessionId.trim()

  if (!id) {
    return
  }

  mutateScope(scope => {
    if (folderId === null) {
      if (!(id in scope.filed)) {
        return scope
      }

      const filed = { ...scope.filed }
      delete filed[id]

      return { ...scope, filed }
    }

    if (scope.filed[id] === folderId || !scope.folders.some(folder => folder.id === folderId)) {
      return scope
    }

    return { ...scope, filed: { ...scope.filed, [id]: folderId } }
  })
}
