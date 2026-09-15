import { migrateSessionOwnerHintsProfile } from './session'
import { migrateSessionTilesProfile } from './session-states'

const PENDING_RENAME_KEY = 'hermes.desktop.pendingProfileRename.v1'
const TRANSCRIPT_PREFIX = 'hermes.transcript-tail.v2:'
const TRANSCRIPT_INDEX_KEY = 'hermes.transcript-tail.v2-index'
const LAST_SESSION_KEY = 'hermes.desktop.lastSessionId'
const LAST_ROUTE_KEY = 'hermes.desktop.lastRoute'

interface PendingProfileRename {
  newName: string
  oldName: string
}

function normalizedName(name: string): string {
  return name.trim().toLowerCase() || 'default'
}

function readPending(): PendingProfileRename | null {
  try {
    const parsed = JSON.parse(
      window.localStorage.getItem(PENDING_RENAME_KEY) ?? 'null'
    ) as Partial<PendingProfileRename>

    if (typeof parsed?.oldName !== 'string' || typeof parsed?.newName !== 'string') {
      return null
    }

    return { oldName: normalizedName(parsed.oldName), newName: normalizedName(parsed.newName) }
  } catch {
    return null
  }
}

function moveStorageValue(store: Storage, source: string, destination: string): void {
  const value = store.getItem(source)

  if (value !== null && store.getItem(destination) === null) {
    store.setItem(destination, value)
  }

  store.removeItem(source)
}

function migrateRememberedNavigation(store: Storage, oldName: string, newName: string): void {
  const oldScope = `.profile.${encodeURIComponent(oldName)}`
  const newScope = `.profile.${encodeURIComponent(newName)}`
  const keys = Array.from({ length: store.length }, (_, index) => store.key(index)).filter((key): key is string =>
    Boolean(key)
  )

  for (const key of keys) {
    if ((key.startsWith(LAST_SESSION_KEY) || key.startsWith(LAST_ROUTE_KEY)) && key.includes(oldScope)) {
      moveStorageValue(store, key, key.replace(oldScope, newScope))
    }
  }
}

function migrateTranscriptTails(store: Storage, oldName: string, newName: string): void {
  let index: string[]

  try {
    const parsed = JSON.parse(store.getItem(TRANSCRIPT_INDEX_KEY) ?? '[]')
    index = Array.isArray(parsed) ? parsed.filter((entry): entry is string => typeof entry === 'string') : []
  } catch {
    return
  }

  const migrated = index.map(suffix => {
    try {
      const scope = JSON.parse(suffix)

      if (!Array.isArray(scope) || scope.length !== 3 || scope[1] !== oldName) {
        return suffix
      }

      const nextSuffix = JSON.stringify([scope[0], newName, scope[2]])
      moveStorageValue(store, TRANSCRIPT_PREFIX + suffix, TRANSCRIPT_PREFIX + nextSuffix)

      return nextSuffix
    } catch {
      return suffix
    }
  })

  store.setItem(TRANSCRIPT_INDEX_KEY, JSON.stringify([...new Set(migrated)]))
}

function migrateProfileState(oldName: string, newName: string): void {
  const store = window.localStorage

  migrateRememberedNavigation(store, oldName, newName)
  migrateTranscriptTails(store, oldName, newName)
  migrateSessionOwnerHintsProfile(oldName, newName)
  migrateSessionTilesProfile(oldName, newName)
}

/** Record intent before the rename request: a primary-profile rename reloads
 * the renderer before its request promise can settle, so the next boot must be
 * able to finish the presentation-state migration. */
export function stageProfileRenameState(oldName: string, newName: string): void {
  const pending = { oldName: normalizedName(oldName), newName: normalizedName(newName) }

  try {
    window.localStorage.setItem(PENDING_RENAME_KEY, JSON.stringify(pending))
  } catch {
    // A storage-restricted renderer still completes the authoritative backend rename.
  }
}

export function cancelProfileRenameState(oldName: string, newName: string): void {
  const pending = readPending()

  if (pending?.oldName === normalizedName(oldName) && pending.newName === normalizedName(newName)) {
    window.localStorage.removeItem(PENDING_RENAME_KEY)
  }
}

export function completeProfileRenameState(oldName: string, newName: string): void {
  const oldProfile = normalizedName(oldName)
  const newProfile = normalizedName(newName)

  if (oldProfile !== newProfile) {
    migrateProfileState(oldProfile, newProfile)
  }

  try {
    window.localStorage.removeItem(PENDING_RENAME_KEY)
  } catch {
    // Best effort: repeating the idempotent migration on a later boot is safe.
  }
}

/** Complete a rename whose successful primary-backend response reloaded the
 * renderer before RenameProfileDialog resumed. */
export function recoverPendingProfileRenameState(activeProfile: string): boolean {
  const pending = readPending()

  if (!pending || pending.newName !== normalizedName(activeProfile)) {
    return false
  }

  completeProfileRenameState(pending.oldName, pending.newName)

  return true
}
