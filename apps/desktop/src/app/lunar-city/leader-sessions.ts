import { getSession } from '@/api/sessions'
import { readJson, writeJson } from '@/lib/storage'
import { retainGatewayForAgent } from '@/store/gateway'
import { normalizeProfileKey } from '@/store/profile'
import { setSessionOwnerHint } from '@/store/session'
import { requestForSessionProfile } from '@/store/session-request-router'

export interface LeaderOwner {
  connectionId: string
  profile: string
}

export interface LeaderSession {
  storedId: string
  runtimeId: string
}

export interface LeaderSessionPersistence {
  version: 1
  leaders: Record<string, { storedId: string }>
}

export interface LeaderStoredSessionRow {
  connection_id?: string
  id: string
  profile?: string
}

export interface LeaderSessionDependencies {
  findStoredSession(owner: LeaderOwner, storedId: string): Promise<LeaderStoredSessionRow | null>
  readPersistence(): unknown
  recordOwnerHint(storedId: string, owner: LeaderOwner): void
  retainOwner(owner: LeaderOwner): Promise<() => void>
  request(
    owner: LeaderOwner,
    method: 'session.create' | 'session.resume',
    params: Record<string, unknown>
  ): Promise<unknown>
  writePersistence(value: LeaderSessionPersistence): void
}

const LEADER_SESSIONS_STORAGE_KEY = 'lunar-city.leader-sessions.v1'

function rejectAmbientRequest<T>(): Promise<T> {
  return Promise.reject(new Error('Leader sessions require an exact owner route; ambient routing is forbidden'))
}

function canonicalOwner(owner: LeaderOwner): LeaderOwner {
  const connectionId = owner.connectionId.trim()

  if (!connectionId) {
    throw new Error('Leader session owner is missing connectionId')
  }

  return { connectionId, profile: normalizeProfileKey(owner.profile) }
}

export function leaderOwnerKey(owner: LeaderOwner): string {
  const canonical = canonicalOwner(owner)

  return `${encodeURIComponent(canonical.connectionId)}::${encodeURIComponent(canonical.profile)}`
}

function emptyPersistence(): LeaderSessionPersistence {
  return { leaders: {}, version: 1 }
}

function parsePersistence(value: unknown): LeaderSessionPersistence {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return emptyPersistence()
  }

  const root = value as Record<string, unknown>

  if (root.version !== 1 || !root.leaders || typeof root.leaders !== 'object' || Array.isArray(root.leaders)) {
    return emptyPersistence()
  }

  if (Object.keys(root).some(key => key !== 'version' && key !== 'leaders')) {
    return emptyPersistence()
  }

  const leaders: Record<string, { storedId: string }> = {}

  for (const [key, entry] of Object.entries(root.leaders as Record<string, unknown>)) {
    if (!key || !entry || typeof entry !== 'object' || Array.isArray(entry)) {
      return emptyPersistence()
    }

    const candidate = entry as Record<string, unknown>
    const storedId = typeof candidate.storedId === 'string' ? candidate.storedId.trim() : ''

    if (!storedId || Object.keys(candidate).some(field => field !== 'storedId')) {
      return emptyPersistence()
    }

    leaders[key] = { storedId }
  }

  return { leaders, version: 1 }
}

function exactOwnerRow(row: LeaderStoredSessionRow | null, owner: LeaderOwner, storedId: string): boolean {
  if (!row || row.id.trim() !== storedId) {
    return false
  }

  if (normalizeProfileKey(row.profile) !== owner.profile) {
    return false
  }

  const taggedConnection = row.connection_id?.trim()

  // The read itself is pinned to owner.connectionId. A returned connection tag
  // is additional evidence and must agree; older/local rows legitimately omit
  // it because the backend does not know the Desktop registry id.
  return !taggedConnection || taggedConnection === owner.connectionId
}

function responseString(response: unknown, field: string): string {
  if (!response || typeof response !== 'object') {
    return ''
  }

  const value = (response as Record<string, unknown>)[field]

  return typeof value === 'string' ? value.trim() : ''
}

function writeOwnerSession(
  dependencies: LeaderSessionDependencies,
  owner: LeaderOwner,
  session: LeaderSession
): LeaderSession {
  // Re-read at the synchronous write boundary. Different leaders may resolve
  // concurrently; carrying either resolution's pre-await snapshot here would
  // let the last response erase the first leader's durable mapping.
  const persistence = parsePersistence(dependencies.readPersistence())

  const next: LeaderSessionPersistence = {
    leaders: { ...persistence.leaders, [leaderOwnerKey(owner)]: { storedId: session.storedId } },
    version: 1
  }

  dependencies.recordOwnerHint(session.storedId, owner)
  dependencies.writePersistence(next)

  return session
}

async function createOnOwner(dependencies: LeaderSessionDependencies, owner: LeaderOwner): Promise<LeaderSession> {
  const releaseOwner = await dependencies.retainOwner(owner)

  try {
    const created = await dependencies.request(owner, 'session.create', {
      cols: 96,
      profile: owner.profile,
      source: 'desktop'
    })

    const runtimeId = responseString(created, 'session_id')
    const storedId = responseString(created, 'stored_session_id')

    if (!runtimeId) {
      throw new Error('Leader session creation did not return a runtime session id')
    }

    if (!storedId) {
      throw new Error('Leader session creation did not return a durable stored session id')
    }

    return writeOwnerSession(dependencies, owner, { runtimeId, storedId })
  } finally {
    releaseOwner()
  }
}

async function resolveOnOwner(dependencies: LeaderSessionDependencies, rawOwner: LeaderOwner): Promise<LeaderSession> {
  const owner = canonicalOwner(rawOwner)
  const persistence = parsePersistence(dependencies.readPersistence())
  const storedId = persistence.leaders[leaderOwnerKey(owner)]?.storedId

  if (!storedId) {
    return createOnOwner(dependencies, owner)
  }

  const row = await dependencies.findStoredSession(owner, storedId)

  if (!exactOwnerRow(row, owner, storedId)) {
    return createOnOwner(dependencies, owner)
  }

  let resumed: unknown

  try {
    resumed = await dependencies.request(owner, 'session.resume', {
      cols: 96,
      profile: owner.profile,
      session_id: storedId,
      source: 'desktop'
    })
  } catch (error) {
    if (isMissingStoredSession(error)) {
      return createOnOwner(dependencies, owner)
    }

    throw error
  }

  const resumeError = responseString(resumed, 'error')

  if (resumeError) {
    if (isMissingStoredSession(resumeError)) {
      return createOnOwner(dependencies, owner)
    }

    throw new Error(resumeError)
  }

  const runtimeId = responseString(resumed, 'session_id')

  if (!runtimeId) {
    return createOnOwner(dependencies, owner)
  }

  return writeOwnerSession(dependencies, owner, { runtimeId, storedId })
}

export function createLeaderSessionResolver(dependencies: LeaderSessionDependencies) {
  const inFlight = new Map<string, Promise<LeaderSession>>()

  return (rawOwner: LeaderOwner): Promise<LeaderSession> => {
    const owner = canonicalOwner(rawOwner)
    const key = leaderOwnerKey(owner)
    const active = inFlight.get(key)

    if (active) {
      return active
    }

    const pending = resolveOnOwner(dependencies, owner).finally(() => {
      if (inFlight.get(key) === pending) {
        inFlight.delete(key)
      }
    })

    inFlight.set(key, pending)

    return pending
  }
}

function isMissingStoredSession(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /(?:^|error:\s*)404\b/i.test(message) || /session[^\n]*not found/i.test(message)
}

const defaultDependencies: LeaderSessionDependencies = {
  async findStoredSession(owner, storedId) {
    try {
      return await getSession(storedId, owner)
    } catch (error) {
      if (isMissingStoredSession(error)) {
        return null
      }

      throw error
    }
  },
  readPersistence: () => readJson<unknown>(LEADER_SESSIONS_STORAGE_KEY),
  recordOwnerHint: (storedId, owner) => setSessionOwnerHint(storedId, owner),
  retainOwner: owner => retainGatewayForAgent(owner.connectionId, owner.profile),
  request: (owner, method, params) => requestForSessionProfile<unknown>(owner, rejectAmbientRequest, method, params),
  writePersistence: value => writeJson(LEADER_SESSIONS_STORAGE_KEY, value)
}

const resolveDefaultLeaderSession = createLeaderSessionResolver(defaultDependencies)

export function resolveLeaderSession(owner: LeaderOwner): Promise<LeaderSession> {
  return resolveDefaultLeaderSession(owner)
}
