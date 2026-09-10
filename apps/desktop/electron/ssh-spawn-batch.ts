import crypto from 'node:crypto'

const BATCH_ID_RE = /^[0-9a-f]{32}$/
const MAX_SCOPE_PART_LENGTH = 256

function normalizeScope(value) {
  if (typeof value !== 'string') {
    return null
  }

  const normalized = value.trim()

  // connectionScopeKey() intentionally represents the registry primary with
  // an empty string. It is still a real scoped backend that must join its
  // profile siblings, not an absent identity.
  return normalized.length <= MAX_SCOPE_PART_LENGTH ? normalized : null
}

function normalizeConnectionId(value) {
  const normalized = typeof value === 'string' ? value.trim() : ''

  return normalized && normalized.length <= MAX_SCOPE_PART_LENGTH ? normalized : null
}

function mintSpawnBatchId() {
  return crypto.randomBytes(16).toString('hex')
}

function createSshSpawnBatchCoordinator({ createId = mintSpawnBatchId } = {}) {
  const batchesByConnection = new Map<string, { id: string; scopes: Set<string> }>()
  const connectionByScope = new Map<string, string>()

  function release(scope) {
    const normalizedScope = normalizeScope(scope)

    if (normalizedScope === null) {
      return
    }

    const connectionId = connectionByScope.get(normalizedScope)

    if (!connectionId) {
      return
    }

    connectionByScope.delete(normalizedScope)
    const batch = batchesByConnection.get(connectionId)

    if (!batch) {
      return
    }

    batch.scopes.delete(normalizedScope)

    if (batch.scopes.size === 0) {
      batchesByConnection.delete(connectionId)
    }
  }

  function acquire(scope, connectionId) {
    const normalizedScope = normalizeScope(scope)
    const normalizedConnectionId = normalizeConnectionId(connectionId)

    if (normalizedScope === null || normalizedConnectionId === null) {
      release(normalizedScope)

      return null
    }

    if (connectionByScope.get(normalizedScope) !== normalizedConnectionId) {
      release(normalizedScope)
    }

    let batch = batchesByConnection.get(normalizedConnectionId)

    if (!batch) {
      const id = String(createId())

      if (!BATCH_ID_RE.test(id)) {
        throw new Error('SSH spawn-batch generator returned an invalid ID.')
      }

      batch = { id, scopes: new Set() }
      batchesByConnection.set(normalizedConnectionId, batch)
    }

    batch.scopes.add(normalizedScope)
    connectionByScope.set(normalizedScope, normalizedConnectionId)

    return batch.id
  }

  return { acquire, release }
}

export { createSshSpawnBatchCoordinator, mintSpawnBatchId }
