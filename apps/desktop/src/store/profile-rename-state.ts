const PENDING_RENAME_KEY = 'hermes.desktop.pendingProfileRename.v1'
const PENDING_RENAME_PREFIX = 'hermes.desktop.pendingProfileRename.v2:'
const COMPLETED_RENAME_PREFIX = 'hermes.desktop.completedProfileRename.v1:'
const TRANSCRIPT_PREFIX = 'hermes.transcript-tail.v2:'
const TRANSCRIPT_INDEX_KEY = 'hermes.transcript-tail.v2-index'
const LAST_SESSION_KEY = 'hermes.desktop.lastSessionId'
const LAST_ROUTE_KEY = 'hermes.desktop.lastRoute'

interface PendingProfileRename {
  attemptId: string
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

function parsePending(raw: string | null): PendingProfileRename | null {
  try {
    const parsed = JSON.parse(raw ?? 'null') as Partial<PendingProfileRename>

    if (typeof parsed?.oldName !== 'string' || typeof parsed?.newName !== 'string') {
      return null
    }

    return {
      attemptId: typeof parsed.attemptId === 'string' && parsed.attemptId ? parsed.attemptId : 'legacy',
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

function newAttemptId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function pendingStorageKey(
  pending: Pick<PendingProfileRename, 'attemptId' | 'connectionId' | 'newName' | 'oldName'>
): string {
  return PENDING_RENAME_PREFIX + encodeURIComponent(
    JSON.stringify([pending.connectionId, pending.oldName, pending.newName, pending.attemptId])
  )
}

function readPending(): Array<{ key: string; pending: PendingProfileRename }> {
  try {
    const store = window.localStorage
    const found: Array<{ key: string; pending: PendingProfileRename }> = []

    for (let index = 0; index < store.length; index += 1) {
      const key = store.key(index)

      if (key?.startsWith(PENDING_RENAME_PREFIX)) {
        const pending = parsePending(store.getItem(key))

        if (pending) {
          found.push({ key, pending })
        }
      }
    }

    const legacy = parsePending(store.getItem(PENDING_RENAME_KEY))

    if (legacy) {
      found.push({ key: PENDING_RENAME_KEY, pending: legacy })
    }

    return found
  } catch {
    return []
  }
}

function moveStorageValue(store: Storage, source: string, destination: string): void {
  const value = store.getItem(source)

  if (value !== null) {
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
  try {
    migrateRememberedNavigation(
      window.localStorage,
      oldName,
      newName,
      scope.oldNavigationSuffix,
      scope.newNavigationSuffix
    )
  } catch {
    // Browser storage is presentation-only; the authoritative backend rename already succeeded.
  }

  try {
    migrateTranscriptTails(window.localStorage, oldName, newName, scope.connectionId)
  } catch {
    // Keep the in-memory registries moving even when persistent storage is unavailable.
  }

  try {
    window.dispatchEvent(
      new CustomEvent('hermes:profile-renamed', { detail: { connectionId: scope.connectionId, newName, oldName } })
    )
  } catch {
    // A non-DOM test or restricted renderer has no live registries to notify.
  }
}

function broadcastCompletedRename(pending: PendingProfileRename): void {
  try {
    const key = COMPLETED_RENAME_PREFIX + encodeURIComponent(pending.attemptId)
    window.localStorage.setItem(key, JSON.stringify(pending))
    window.localStorage.removeItem(key)
  } catch {
    // Other windows can reconcile on their next load; never fail the backend rename.
  }
}

function removePending(pending: PendingProfileRename, attemptId?: string): void {
  try {
    for (const entry of readPending()) {
      const sameIdentity =
        entry.pending.connectionId === pending.connectionId &&
        entry.pending.oldName === pending.oldName &&
        entry.pending.newName === pending.newName

      const sameAttempt = attemptId ? entry.pending.attemptId === attemptId : sameIdentity

      if (sameIdentity && sameAttempt) {
        window.localStorage.removeItem(entry.key)
      }
    }
  } catch {
    // A leftover marker only repeats an idempotent migration on a later boot.
  }
}

/** Record intent before the rename request: a primary-profile rename reloads
 * the renderer before its request promise can settle, so the next boot must be
 * able to finish the presentation-state migration. */
export function stageProfileRenameState(
  oldName: string,
  newName: string,
  scope: ProfileRenameStateScope = { connectionId: 'local', newNavigationSuffix: '', oldNavigationSuffix: '' }
): string {
  const connectionId = scope.connectionId.trim() || 'local'
  const normalizedNewName = normalizedName(newName)

  if (
    readPending().some(
      entry => entry.pending.connectionId === connectionId && entry.pending.newName === normalizedNewName
    )
  ) {
    throw new Error(`A rename to ${normalizedNewName} is already in progress on this connection`)
  }

  const pending = {
    ...scope,
    attemptId: newAttemptId(),
    connectionId,
    oldName: normalizedName(oldName),
    newName: normalizedNewName
  }

  try {
    window.localStorage.setItem(pendingStorageKey(pending), JSON.stringify(pending))
  } catch {
    // A storage-restricted renderer still completes the authoritative backend rename.
  }

  return pending.attemptId
}

export function cancelProfileRenameState(
  oldName: string,
  newName: string,
  scope: Pick<ProfileRenameStateScope, 'connectionId'> = { connectionId: 'local' },
  attemptId?: string
): void {
  removePending(
    {
      attemptId: attemptId ?? 'legacy',
      connectionId: scope.connectionId.trim() || 'local',
      oldName: normalizedName(oldName),
      newName: normalizedName(newName),
      oldNavigationSuffix: null,
      newNavigationSuffix: null
    },
    attemptId
  )
}

export function completeProfileRenameState(
  oldName: string,
  newName: string,
  scope: ProfileRenameStateScope = { connectionId: 'local', newNavigationSuffix: '', oldNavigationSuffix: '' },
  attemptId?: string
): void {
  const pending = {
    ...scope,
    attemptId: attemptId ?? 'legacy',
    connectionId: scope.connectionId.trim() || 'local',
    oldName: normalizedName(oldName),
    newName: normalizedName(newName)
  }

  if (pending.oldName !== pending.newName) {
    migrateProfileState(pending.oldName, pending.newName, pending)
    broadcastCompletedRename(pending)
  }

  removePending(pending, attemptId)
}

/** Complete a rename whose successful primary-backend response reloaded the
 * renderer before RenameProfileDialog resumed. */
export function recoverPendingProfileRenameState(activeProfile: string, activeConnectionId: string): boolean {
  const matches = readPending().filter(
    entry =>
      entry.pending.newName === normalizedName(activeProfile) &&
      entry.pending.connectionId === (activeConnectionId.trim() || 'local')
  )

  // Ambiguous concurrent targets need backend evidence to identify the winner.
  // Fail closed rather than migrating presentation state from the wrong profile.
  if (matches.length !== 1) {
    return false
  }

  const pending = matches[0].pending
  completeProfileRenameState(pending.oldName, pending.newName, pending, pending.attemptId)

  return true
}

if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (!event.key?.startsWith(COMPLETED_RENAME_PREFIX) || event.newValue === null) {
      return
    }

    const pending = parsePending(event.newValue)

    if (pending) {
      migrateProfileState(pending.oldName, pending.newName, pending)
    }
  })
}
