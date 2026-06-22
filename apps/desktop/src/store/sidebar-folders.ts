import { atom } from 'nanostores'

// Desktop-local "custom folders" for the chat sidebar. Pure client UI state:
// no DB row, no gateway round-trip, no prompt/session-persistence impact. A
// folder just remembers a name + an ordered list of session keys; membership is
// keyed by the same durable id pinning uses (sessionPinId = _lineage_root_id ??
// id) so a folder survives compaction the way a pin does.
export interface SidebarFolder {
  id: string
  name: string
  // Durable session keys (sessionPinId), in display order. Keys for sessions
  // that aren't currently loaded/in-scope are kept verbatim — they simply don't
  // resolve to a visible row until that session loads again.
  sessionIds: string[]
  open: boolean
  // The profile (normalizeProfileKey) this folder belongs to; the folder only
  // shows while that profile is the active scope. Undefined on folders created
  // before scoping existed — those are treated as global (see normalize).
  profileKey?: string
  // Global folder: shows in every profile regardless of profileKey.
  pinned?: boolean
}

const STORAGE_KEY = 'hermes.desktop.sidebarFolders'

function isFolder(value: unknown): value is SidebarFolder {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    typeof r.id === 'string' &&
    r.id.length > 0 &&
    typeof r.name === 'string' &&
    Array.isArray(r.sessionIds) &&
    r.sessionIds.every(id => typeof id === 'string')
  )
}

function normalize(value: SidebarFolder): SidebarFolder {
  return {
    id: value.id,
    name: value.name,
    // Drop blank/duplicate keys defensively so a corrupted store can't render
    // ghost rows or double-count.
    open: typeof value.open === 'boolean' ? value.open : true,
    // Pre-scoping folders have no profileKey: leaving it undefined makes
    // visibleFoldersForScope treat them as global so they don't vanish.
    pinned: typeof value.pinned === 'boolean' ? value.pinned : undefined,
    profileKey: typeof value.profileKey === 'string' && value.profileKey.length > 0 ? value.profileKey : undefined,
    sessionIds: [...new Set(value.sessionIds.filter(id => id.length > 0))]
  }
}

function load(): SidebarFolder[] {
  if (typeof window === 'undefined') {
    return []
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)

    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw) as unknown

    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.filter(isFolder).map(normalize)
  } catch {
    // Treat unparseable persisted state as missing — never throw on boot.
    return []
  }
}

function persist(folders: readonly SidebarFolder[]) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    if (folders.length === 0) {
      window.localStorage.removeItem(STORAGE_KEY)
    } else {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(folders))
    }
  } catch {
    // Folders are a local convenience; restricted storage must not break chat.
  }
}

export const $sidebarFolders = atom<SidebarFolder[]>(load())

$sidebarFolders.subscribe(persist)

// Match the repo's lightweight id convention (see store/composer-queue.ts); this
// app has no uuid dependency and these ids only need to be unique within the
// browser-local folder list.
function createFolderId(): string {
  return `folder-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

// Replace the whole list; nanostores notifies on identity change, so callers
// must hand back a fresh array.
function setFolders(next: SidebarFolder[]) {
  $sidebarFolders.set(next)
}

// `profileKey` scopes the folder to the profile it was created in (the active
// scope at creation time). Omit it to create a global folder.
export function createFolder(name: string, profileKey?: string): string {
  const trimmed = name.trim()
  const id = createFolderId()

  setFolders([...$sidebarFolders.get(), { id, name: trimmed, open: true, profileKey, sessionIds: [] }])

  return id
}

// Toggle a folder between profile-scoped and global (shows in every profile).
export function toggleFolderPinned(id: string) {
  const current = $sidebarFolders.get()

  if (!current.some(f => f.id === id)) {
    return
  }

  setFolders(current.map(f => (f.id === id ? { ...f, pinned: !f.pinned } : f)))
}

export function renameFolder(id: string, name: string) {
  const trimmed = name.trim()

  if (!trimmed) {
    return
  }

  const current = $sidebarFolders.get()

  if (!current.some(f => f.id === id && f.name !== trimmed)) {
    return
  }

  setFolders(current.map(f => (f.id === id ? { ...f, name: trimmed } : f)))
}

// Deleting a folder only drops the folder itself — sessions are never deleted,
// they simply become unfiled (their rows reappear in the main Sessions list).
export function deleteFolder(id: string) {
  const current = $sidebarFolders.get()

  if (!current.some(f => f.id === id)) {
    return
  }

  setFolders(current.filter(f => f.id !== id))
}

// A session lives in at most one folder: clear it from every folder first, then
// add it to the target. No-op if the target folder is gone.
export function moveSessionToFolder(sessionKey: string, folderId: string) {
  if (!sessionKey) {
    return
  }

  const current = $sidebarFolders.get()

  if (!current.some(f => f.id === folderId)) {
    return
  }

  setFolders(
    current.map(f => {
      if (f.id === folderId) {
        return f.sessionIds.includes(sessionKey) ? f : { ...f, sessionIds: [...f.sessionIds, sessionKey] }
      }

      return f.sessionIds.includes(sessionKey)
        ? { ...f, sessionIds: f.sessionIds.filter(id => id !== sessionKey) }
        : f
    })
  )
}

export function removeSessionFromFolder(sessionKey: string) {
  const current = $sidebarFolders.get()

  if (!current.some(f => f.sessionIds.includes(sessionKey))) {
    return
  }

  setFolders(current.map(f => ({ ...f, sessionIds: f.sessionIds.filter(id => id !== sessionKey) })))
}

export function setFolderOpen(id: string, open: boolean) {
  const current = $sidebarFolders.get()

  if (!current.some(f => f.id === id && f.open !== open)) {
    return
  }

  setFolders(current.map(f => (f.id === id ? { ...f, open } : f)))
}

export function toggleFolderOpen(id: string) {
  const folder = $sidebarFolders.get().find(f => f.id === id)

  if (folder) {
    setFolderOpen(id, !folder.open)
  }
}

// Reorder the folder list to match `ids`; any folder not named keeps its
// relative position at the end so a stale id list can't drop folders.
export function reorderFolders(ids: string[]) {
  const current = $sidebarFolders.get()
  const byId = new Map(current.map(f => [f.id, f]))
  const ordered = ids.map(id => byId.get(id)).filter((f): f is SidebarFolder => Boolean(f))
  const seen = new Set(ordered.map(f => f.id))
  const next = [...ordered, ...current.filter(f => !seen.has(f.id))]

  if (next.length === current.length && next.every((f, i) => f === current[i])) {
    return
  }

  setFolders(next)
}

// Reorder sessions within one folder. `sessionKeys` covers only the currently
// visible rows; keys for non-loaded members are appended so a reorder can't
// silently evict hidden sessions.
export function reorderFolderSessions(id: string, sessionKeys: string[]) {
  const current = $sidebarFolders.get()
  const folder = current.find(f => f.id === id)

  if (!folder) {
    return
  }

  const inFolder = new Set(folder.sessionIds)
  const visible = sessionKeys.filter(key => inFolder.has(key))
  const seen = new Set(visible)
  const next = [...visible, ...folder.sessionIds.filter(key => !seen.has(key))]

  if (next.length === folder.sessionIds.length && next.every((key, i) => key === folder.sessionIds[i])) {
    return
  }

  setFolders(current.map(f => (f.id === id ? { ...f, sessionIds: next } : f)))
}

// ---- Pure read helpers (no store access) -----------------------------------

// Every session key claimed by any folder — used to filter the unfiled list.
export function folderKeySet(folders: SidebarFolder[]): Set<string> {
  const out = new Set<string>()

  for (const folder of folders) {
    for (const key of folder.sessionIds) {
      out.add(key)
    }
  }

  return out
}

export function folderForKey(folders: SidebarFolder[], key: string): SidebarFolder | undefined {
  return folders.find(folder => folder.sessionIds.includes(key))
}
