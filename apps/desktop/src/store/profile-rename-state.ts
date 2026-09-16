const PENDING_RENAME_KEY = 'hermes.desktop.pendingProfileRename.v1'
const TRANSCRIPT_PREFIX = 'hermes.transcript-tail.v2:'
const TRANSCRIPT_INDEX_KEY = 'hermes.transcript-tail.v2-index'
const LAST_SESSION_KEY = 'hermes.desktop.lastSessionId'
const LAST_ROUTE_KEY = 'hermes.desktop.lastRoute'

interface PendingProfileRename {
  connectionId: string
  newName: string
  newNavigationSuffix: null | string
  oldName: string
  oldNavigationSuffix: null | string
}

export interface ProfileRenameStateScope {
  connectionId: string
  newNavigationSuffix: null | string
  oldNavigationSuffix: null | string
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

    return {
      connectionId: typeof parsed.connectionId === 'string' ? parsed.connectionId.trim() || 'local' : 'local',
      oldName: normalizedName(parsed.oldName),
      newName: normalizedName(parsed.newName),
      oldNavigationSuffix:
        parsed.oldNavigationSuffix === null
          ? null
          : typeof parsed.oldNavigationSuffix === 'string'
            ? parsed.oldNavigationSuffix
            : '',
      newNavigationSuffix:
        parsed.newNavigationSuffix === null
          ? null
          : typeof parsed.newNavigationSuffix === 'string'
            ? parsed.newNavigationSuffix
            : ''
    }
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

function migrateRememberedNavigation(
  store: Storage,
  oldName: string,
  newName: string,
  oldSuffix: null | string,
  newSuffix: null | string
): void {
  if (oldSuffix === null || newSuffix === null) {
    return
  }

  const oldScope = `.profile.${encodeURIComponent(oldName)}`
  const newScope = `.profile.${encodeURIComponent(newName)}`

  for (const base of [LAST_SESSION_KEY, LAST_ROUTE_KEY]) {
    moveStorageValue(store, base + oldScope + oldSuffix, base + newScope + newSuffix)
  }
}

function migrateTranscriptTails(store: Storage, oldName: string, newName: string, connectionId: string): void {
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

      if (!Array.isArray(scope) || scope.length !== 3 || scope[0] !== connectionId || scope[1] !== oldName) {
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

function migrateProfileState(oldName: string, newName: string, scope: ProfileRenameStateScope): void {
  const store = window.localStorage

  migrateRememberedNavigation(
    store,
    oldName,
    newName,
    scope.oldNavigationSuffix,
    scope.newNavigationSuffix
  )
  migrateTranscriptTails(store, oldName, newName, scope.connectionId)
  window.dispatchEvent(
    new CustomEvent('hermes:profile-renamed', { detail: { connectionId: scope.connectionId, newName, oldName } })
  )
}

/** Record intent before the rename request: a primary-profile rename reloads
 * the renderer before its request promise can settle, so the next boot must be
 * able to finish the presentation-state migration. */
export function stageProfileRenameState(
  oldName: string,
  newName: string,
  scope: ProfileRenameStateScope = { connectionId: 'local', newNavigationSuffix: '', oldNavigationSuffix: '' }
): void {
  const pending = { ...scope, oldName: normalizedName(oldName), newName: normalizedName(newName) }

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

export function completeProfileRenameState(
  oldName: string,
  newName: string,
  scope: ProfileRenameStateScope = { connectionId: 'local', newNavigationSuffix: '', oldNavigationSuffix: '' }
): void {
  const oldProfile = normalizedName(oldName)
  const newProfile = normalizedName(newName)

  if (oldProfile !== newProfile) {
    migrateProfileState(oldProfile, newProfile, scope)
  }

  try {
    window.localStorage.removeItem(PENDING_RENAME_KEY)
  } catch {
    // Best effort: repeating the idempotent migration on a later boot is safe.
  }
}

/** Complete a rename whose successful primary-backend response reloaded the
 * renderer before RenameProfileDialog resumed. */
export function recoverPendingProfileRenameState(activeProfile: string, activeConnectionId: string): boolean {
  const pending = readPending()

  if (
    !pending ||
    pending.newName !== normalizedName(activeProfile) ||
    pending.connectionId !== activeConnectionId.trim()
  ) {
    return false
  }

  completeProfileRenameState(pending.oldName, pending.newName, pending)

  return true
}
