export interface OverviewSession {
  id: string
  profile: string
  resolved_id?: string
  runtime_id?: string
  title?: string
  preview?: string
  model?: string
  provider?: string
  source?: string
  started_at?: number
  last_active?: number
  message_count?: number
  unread?: boolean
  status?: 'waiting' | 'starting' | 'working' | 'idle'
}

export interface OverviewProfile {
  name: string
  display_name?: string
  description?: string
  model?: string
  provider?: string
}

export interface OverviewSource {
  connectionId: string
  label: string
  kind: string
  state: 'ready' | 'on-demand' | 'offline' | 'unsupported' | 'partial'
  error?: string
  sessions: OverviewSession[]
  canonical: OverviewSession[]
  live: OverviewSession[]
  profiles: OverviewProfile[]
  /** Stored visible history total, excludes canonical + ephemeral live rows. */
  total: number
  /** Number of history entries read; no hidden cap or truncated success. */
  offset: number
  complete: boolean
  liveCoverage: 'process' | 'unavailable'
  errors: Array<{ profile?: string; error: string }>
}

export interface AgentOverview {
  sources: OverviewSource[]
  fetchedAt: number
}

interface Source {
  id: string
  label: string
  kind: string
}

interface OverviewPage {
  sessions: OverviewSession[]
  canonical: OverviewSession[]
  live?: OverviewSession[]
  profiles: OverviewProfile[]
  total: number
  errors: Array<{ profile?: string; error: string }>
  live_coverage?: 'process' | 'unavailable'
}

interface HistoryCacheEntry {
  at: number
  signature: string
  source: OverviewSource
}

interface CollectorOptions<T> {
  cache?: Map<string, HistoryCacheEntry>
  force?: boolean
  sources: Source[]
  pooled: Map<string, T[]>
  /** Credential-free profile names only; never start a parked runtime. */
  discoverParked?: (connectionId: string) => Promise<OverviewProfile[]>
  /** Called ONLY for URL/cloud, never local/SSH. Must not activate a profile. */
  connect: (connectionId: string) => Promise<T[]>
  fetch: (descriptor: T, path: string) => Promise<unknown>
}

function publicFields(raw: unknown, fields: readonly string[]): Record<string, unknown> {
  const value = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}

  return Object.fromEntries(
    fields.filter(key => ['string', 'number', 'boolean'].includes(typeof value[key])).map(key => [key, value[key]])
  )
}

function sessionRow(raw: unknown, profile?: string, live = false): OverviewSession {
  const row = publicFields(raw, [
    'id',
    'profile',
    'resolved_id',
    'title',
    'preview',
    'model',
    'provider',
    'source',
    'started_at',
    'last_active',
    'message_count',
    'unread',
    ...(live ? ['runtime_id', 'status'] : [])
  ])

  if (profile) {
    row.profile = profile
  }

  if (typeof row.id !== 'string' || typeof row.profile !== 'string') {
    throw new Error('invalid-row')
  }

  const billing = publicFields(raw, ['billing_provider']).billing_provider

  if (!row.provider && typeof billing === 'string') {
    row.provider = billing
  }

  return row as unknown as OverviewSession
}

function profileRow(raw: unknown): OverviewProfile {
  const row = publicFields(raw, ['name', 'display_name', 'description', 'model', 'provider'])

  if (typeof row.name !== 'string') {
    throw new Error('invalid-profile')
  }

  return row as unknown as OverviewProfile
}

function pageBody(raw: unknown): OverviewPage {
  const body = raw as OverviewPage

  if (!body || !Array.isArray(body.sessions) || !Number.isFinite(body.total) || body.total < 0) {
    throw new Error('invalid-response')
  }

  return {
    sessions: body.sessions.map(row => sessionRow(row)),
    total: body.total,
    canonical: (body.canonical ?? []).map(row => sessionRow(row)),
    live: (body.live ?? []).map(row => sessionRow(row, undefined, true)),
    profiles: (body.profiles ?? []).map(profileRow),
    live_coverage: body.live_coverage,
    errors: (body.errors ?? []).map(error => ({
      ...publicFields(error, ['profile']),
      error: 'session-store-unavailable'
    }))
  }
}

function missingCapability(error: unknown): boolean {
  const status = (error as { statusCode?: number })?.statusCode

  return status === 404 || status === 405 || status === 501
}

async function legacyInventory<T>(
  descriptor: T,
  fetch: CollectorOptions<T>['fetch'],
  result: OverviewSource
): Promise<void> {
  // Old REST session lists archive/heal stores. profiles.list / session.list
  // RPCs can resurrect canonical chats and/or acquire writable DB handles.
  // There is no side-effect-free, uncapped history fallback on those backends.
  const raw = (await fetch(descriptor, '/api/profiles')) as { profiles?: unknown[] }

  if (!Array.isArray(raw?.profiles)) {
    throw new Error('invalid-response')
  }

  result.profiles = raw.profiles.map(profileRow)
  result.state = 'unsupported'
  result.error = 'read-only-inventory-unavailable'
  result.complete = false
}

interface HistoryRead {
  total: number
  offset: number
  /** False when activity-ordered offset paging shifted under the read. */
  stable: boolean
}

const HISTORY_PATH = '/api/profiles/agent-overview?limit=100&offset='

async function readHistory<T>(
  descriptor: T,
  readPage: CollectorOptions<T>['fetch'],
  first: OverviewPage,
  sink: Pick<OverviewSource, 'sessions' | 'errors'>
): Promise<HistoryRead> {
  // The backend orders by activity with LIMIT/OFFSET and exposes no cursor.
  // A session becoming active between pages moves to the front, so a later
  // page re-serves a seen row while the moved row is never returned. A seen
  // (profile, id) is the only client-visible signal that a row was skipped.
  // Rows land in the sink as they arrive so a timeout keeps what was read.
  const seen = new Set<string>()
  const read: HistoryRead = { total: first.total, offset: 0, stable: true }
  let page = first

  for (;;) {
    for (const row of page.sessions) {
      const key = `${row.profile}\0${row.id}`

      if (seen.has(key)) {
        return { ...read, stable: false }
      }

      seen.add(key)
      sink.sessions.push(row)
      read.offset += 1
    }

    if (read.offset >= read.total) {
      return read
    }

    page = pageBody(await readPage(descriptor, `${HISTORY_PATH}${read.offset}`))
    sink.errors.push(...page.errors)

    if (!page.sessions.length) {
      throw new Error('incomplete-page')
    }

    read.total = page.total
  }
}

async function bounded<T>(work: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      work,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error('read-timeout')), ms)
      })
    ])
  } finally {
    clearTimeout(timer)
  }
}

export async function gatherOverviewBackends<T>(
  promises: Promise<T>[],
  ownerOf: (descriptor: T) => string | null
): Promise<Map<string, T[]>> {
  const pooled = new Map<string, T[]>()

  for (const result of await Promise.allSettled(promises.map(promise => bounded(promise, 2_000)))) {
    if (result.status !== 'fulfilled') {
      continue
    }

    const descriptor = result.value
    const owner = ownerOf(descriptor)

    if (!owner) {
      continue
    }

    const values = pooled.get(owner) ?? []

    if (!values.includes(descriptor)) {
      values.push(descriptor)
    }

    pooled.set(owner, values)
  }

  return pooled
}

export function createAgentOverviewReader<T>() {
  const cache = new Map<string, HistoryCacheEntry>()

  return (options: CollectorOptions<T>, control: { force?: boolean } = {}) =>
    collectAgentOverview({ ...options, cache, force: control.force })
}

export async function collectAgentOverview<T>({
  sources,
  pooled,
  discoverParked,
  connect,
  fetch,
  cache,
  force
}: CollectorOptions<T>): Promise<AgentOverview> {
  return {
    fetchedAt: Date.now(),
    sources: await Promise.all(
      sources.map(async source => {
        // One budget for discovery, descriptors and every history page. The
        // pool gather uses 2s; leave headroom under the renderer's 20s deadline.
        const expiresAt = Date.now() + 12_000

        const read = <R>(work: () => Promise<R>): Promise<R> => {
          const remaining = expiresAt - Date.now()

          if (remaining <= 0) {
            return Promise.reject(new Error('read-timeout'))
          }

          // Bound the I/O, not a mutating collector task: a late response cannot
          // resume pagination, mutate the published snapshot or populate cache.
          return bounded(work(), remaining)
        }

        const readPage: CollectorOptions<T>['fetch'] = (descriptor, path) => read(() => fetch(descriptor, path))

        const result: OverviewSource = {
          connectionId: source.id,
          label: source.label,
          kind: source.kind,
          state: 'ready',
          sessions: [],
          canonical: [],
          live: [],
          profiles: [],
          total: 0,
          offset: 0,
          complete: false,
          liveCoverage: 'unavailable',
          errors: []
        }

        const signature = JSON.stringify(source)
        const cached = !force ? cache?.get(source.id) : undefined

        const history =
          cached && cached.signature === signature && Date.now() - cached.at < 60_000 ? cached.source : undefined

        if (history) {
          Object.assign(result, {
            sessions: history.sessions,
            canonical: history.canonical,
            profiles: history.profiles,
            total: history.total,
            offset: history.offset,
            complete: history.complete,
            state: history.state,
            error: history.error
          })
        }

        try {
          let descriptors = pooled.get(source.id) ?? []

          if (!descriptors.length && (source.kind === 'remote' || source.kind === 'cloud')) {
            descriptors = await read(() => connect(source.id))
          }

          if (!descriptors.length) {
            result.state = 'on-demand'
            result.complete = false

            if (discoverParked && (source.kind === 'local' || source.kind === 'ssh')) {
              try {
                result.profiles = (await read(() => bounded(discoverParked(source.id), 8_000))).map(profileRow)
              } catch {
                result.error = 'profile-inventory-unavailable'
              }
            }

            return result
          }

          let haveHistory = Boolean(history)

          for (const descriptor of descriptors) {
            let page: OverviewPage

            try {
              page = pageBody(
                await readPage(
                  descriptor,
                  haveHistory ? '/api/profiles/agent-overview?live_only=true' : `${HISTORY_PATH}0`
                )
              )
            } catch (error) {
              if (Date.now() >= expiresAt) {
                throw error
              }

              if (missingCapability(error)) {
                if (!haveHistory && result.state !== 'unsupported') {
                  await legacyInventory(descriptor, readPage, result)
                } else if (haveHistory) {
                  result.errors.push({ error: 'live-capability-unavailable' })
                }
              } else {
                result.errors.push({ error: 'runtime-unavailable' })
              }

              continue
            }

            result.live.push(...(page.live ?? []))
            result.errors.push(...page.errors)

            if (page.live_coverage === 'process') {
              result.liveCoverage = 'process'
            }

            if (haveHistory) {
              continue
            }

            haveHistory = true
            result.state = 'ready'
            delete result.error
            result.profiles = page.profiles
            result.canonical = page.canonical

            let read = await readHistory(descriptor, readPage, page, result)

            // One bounded retry from offset 0 under the same source budget; a
            // second shift leaves the source honestly partial for the next poll.
            if (!read.stable && Date.now() < expiresAt) {
              // Live rows were already taken from the first page.
              const retried = pageBody(await readPage(descriptor, `${HISTORY_PATH}0`))
              result.sessions = []
              result.errors.push(...retried.errors)
              read = await readHistory(descriptor, readPage, retried, result)
            }

            result.total = read.total
            result.offset = read.offset

            if (!read.stable) {
              result.error = 'history-shifted'
            }
          }

          result.complete = haveHistory && !result.error && result.offset >= result.total && result.errors.length === 0

          if (!result.complete && result.state !== 'unsupported') {
            result.state = haveHistory ? 'partial' : 'offline'
          }

          if (result.complete && !history) {
            cache?.set(source.id, { at: Date.now(), signature, source: result })
          }
        } catch {
          result.complete = false
          result.state = result.sessions.length ? 'partial' : 'offline'
          result.error = Date.now() >= expiresAt ? 'read-timeout' : 'read-failed'
        }

        return result
      })
    )
  }
}
