/**
 * Electron-free request leg shared by the IPC REST path and focused routing
 * tests. Backend lifecycle stays in main.ts; this helper makes the selected
 * connection observable all the way through the HTTP request.
 */

export interface BackendConnection {
  authMode?: string
  baseUrl: string
  token?: null | string
}

interface RequestOptions {
  body?: unknown
  method?: string
  timeoutMs: number
}

interface ProfileRequest {
  body?: unknown
  explicitLocal?: boolean
  explicitRemote?: boolean
  gatewayId?: string
  globalRemote?: boolean
  method?: string
  path: string
  profile?: null | string
  primaryProfile: string
  timeoutMs: number
}

interface ProfileRequestDependencies<T> {
  ensureBackend: (profile?: null | string, options?: EnsureBackendOptions) => Promise<BackendConnection>
  requestOauth: (url: string, options: RequestOptions) => Promise<T>
  requestToken: (url: string, token: null | string | undefined, options: RequestOptions) => Promise<T>
}

interface ConnectionRequestDependencies<T> {
  requestOauth: (url: string, options: RequestOptions) => Promise<T>
  requestToken: (url: string, token: null | string | undefined, options: RequestOptions) => Promise<T>
}

interface ProfileSessionAggregationDependencies {
  requestPrimary: (path: string) => Promise<unknown>
  requestRemoteProfile: (profile: string, searchParams: URLSearchParams) => Promise<unknown>
}

export type BackendRouteIdentity = 'local' | 'remote-global' | 'remote-profile'

export type BackendTarget = 'pool' | 'primary'

export interface BackendSelection {
  route: BackendRouteIdentity
  target: BackendTarget
}

export interface EnsureBackendOptions {
  forceLocal?: boolean
  selection?: BackendSelection
}

/**
 * Current profile routing has one defensible gateway identity beyond the
 * ambient connection: `local`. Keep that target narrow until a real gateway
 * registry exists instead of guessing how arbitrary ids map to profiles.
 */
export function gatewayRequestForcesLocal(gatewayId?: string): boolean {
  const target = String(gatewayId || '').trim()

  if (!target) {
    return false
  }

  if (target === 'local') {
    return true
  }

  throw new Error(`Unknown gateway target: ${target}`)
}

interface BackendLifecycleDependencies<T> {
  ensurePool: (profile: string, route: BackendRouteIdentity) => Promise<T>
  resolveCurrentSelection: (profile: string, forceLocal: boolean) => BackendSelection
  runExclusive: (profile: string, operation: () => Promise<T>) => Promise<T>
  startPrimary: () => Promise<T>
}

/**
 * Resolve route identity without touching Electron or minting credentials.
 * This is deliberately the same policy used before backend lifecycle work so
 * a cached pool entry can never silently change from local to remote (or back).
 */
export function selectBackendRoute(options: {
  explicitLocal?: boolean
  explicitRemote?: boolean
  forceLocal?: boolean
  globalRemote?: boolean
}): BackendRouteIdentity {
  if (options.forceLocal || options.explicitLocal) {
    return 'local'
  }

  if (options.explicitRemote) {
    return 'remote-profile'
  }

  return options.globalRemote ? 'remote-global' : 'local'
}

/**
 * Keep normal active-profile traffic on the primary lifecycle. An explicit
 * local request needs an auxiliary profile-pinned backend when that lifecycle
 * is remote, while forceLocal must never call a potentially remote primary.
 */
export function selectBackendTarget(options: {
  explicitLocal?: boolean
  forceLocal?: boolean
  globalRemote?: boolean
  profile: string
  primaryProfile: string
}): BackendTarget {
  if (options.profile !== options.primaryProfile) {
    return 'pool'
  }

  if (options.forceLocal || (options.explicitLocal && options.globalRemote)) {
    return 'pool'
  }

  return 'primary'
}

/** Resolve the complete lifecycle choice used by ensureBackend. */
export function selectBackendSelection(options: {
  explicitLocal?: boolean
  explicitRemote?: boolean
  forceLocal?: boolean
  globalRemote?: boolean
  profile: string
  primaryProfile: string
}): BackendSelection {
  const route = selectBackendRoute(options)

  return {
    route,
    target: selectBackendTarget(options)
  }
}

/**
 * Execute the lifecycle decision used by main.ts. A request-provided selection
 * is authoritative for that queued request; callers without an override keep
 * resolving against the latest applied config inside the lifecycle queue.
 */
export function ensureBackendLifecycle<T>(
  profile: string,
  options: EnsureBackendOptions,
  dependencies: BackendLifecycleDependencies<T>
): Promise<T> {
  return dependencies.runExclusive(profile, async () => {
    const selection = options.selection ?? dependencies.resolveCurrentSelection(profile, Boolean(options.forceLocal))

    return selection.target === 'primary'
      ? dependencies.startPrimary()
      : dependencies.ensurePool(profile, selection.route)
  })
}

export function shouldReuseBackendRoute(existing: BackendRouteIdentity, desired: BackendRouteIdentity): boolean {
  return existing === desired
}

/**
 * Serialize lifecycle work for one profile while allowing unrelated profiles
 * to proceed independently. The queue is intentionally small and Electron-
 * free so the main-process race contract can be tested without importing
 * main.ts.
 */
export function createProfileAsyncQueue() {
  const tails = new Map<string, Promise<void>>()

  return {
    run<T>(profile: string, operation: () => Promise<T> | T): Promise<T> {
      const previous = tails.get(profile) ?? Promise.resolve()
      let release!: () => void

      const current = new Promise<void>(resolve => {
        release = resolve
      })

      tails.set(profile, current)

      return previous.then(operation).finally(() => {
        release()

        if (tails.get(profile) === current) {
          tails.delete(profile)
        }
      })
    }
  }
}

/**
 * Reuse or replace one profile's pool entry without losing a replacement that
 * another caller installed while teardown awaited a starting child.
 */
export async function ensureCompatiblePoolEntry<T extends { routeIdentity: BackendRouteIdentity }>(options: {
  create: () => T
  get: () => T | undefined
  route: BackendRouteIdentity
  teardown: (entry: T) => Promise<void>
  touch?: (entry: T) => void
}): Promise<T> {
  while (true) {
    const existing = options.get()

    if (!existing) {
      return options.create()
    }

    if (shouldReuseBackendRoute(existing.routeIdentity, options.route)) {
      options.touch?.(existing)

      return existing
    }

    await options.teardown(existing)
    // Re-read on the next loop: teardown may have awaited a starting backend,
    // during which another caller could have installed the desired replacement.
  }
}

/** Lifecycle work required before making a pooled profile the primary. */
export function planProfilePromotion(currentPrimary: string, nextPrimary: string) {
  return {
    poolProfileToTeardown: nextPrimary,
    primaryProfileToTeardown: currentPrimary
  }
}

/** Decide which cached pool routes a connection apply invalidates. */
export function connectionApplyAffectsPoolProfile(options: {
  appliedProfile: null | string
  hasExplicitProfileRoute: boolean
  primaryProfile: string
  profile: string
}): boolean {
  if (options.profile === options.primaryProfile) {
    return true
  }

  if (options.appliedProfile) {
    return options.profile === options.appliedProfile
  }

  return !options.hasExplicitProfileRoute
}

export async function requestJsonForProfileRoute<T>(
  request: ProfileRequest,
  dependencies: ProfileRequestDependencies<T>
): Promise<T> {
  const forceLocal = gatewayRequestForcesLocal(request.gatewayId)

  const selection = selectBackendSelection({
    explicitLocal: request.explicitLocal,
    explicitRemote: request.explicitRemote,
    forceLocal,
    globalRemote: request.globalRemote,
    primaryProfile: request.primaryProfile,
    profile: request.profile || request.primaryProfile
  })

  const connection = await dependencies.ensureBackend(request.profile, { selection })

  return requestJsonForBackendConnection(
    connection,
    request.path,
    {
      body: request.body,
      method: request.method,
      timeoutMs: request.timeoutMs
    },
    dependencies
  )
}

/** Send JSON through the auth leg owned by the resolved backend connection. */
export function requestJsonForBackendConnection<T>(
  connection: BackendConnection,
  path: string,
  options: RequestOptions,
  dependencies: ConnectionRequestDependencies<T>
): Promise<T> {
  const url = `${connection.baseUrl}${path}`

  return connection.authMode === 'oauth'
    ? dependencies.requestOauth(url, options)
    : dependencies.requestToken(url, connection.token, options)
}

const sessionRows = (data: any): any[] => (Array.isArray(data?.sessions) ? data.sessions : [])

/**
 * Merge an aggregate from the primary with per-profile remote aggregates.
 * Primary failures are authoritative and propagate; only an optional remote
 * profile may disappear from the aggregate when that peer is unavailable.
 */
export async function mergeRemoteProfileSessionAggregates(
  searchParams: URLSearchParams,
  remoteProfiles: string[],
  dependencies: ProfileSessionAggregationDependencies
) {
  const limit = Math.max(1, Number(searchParams.get('limit')) || 20)
  const offset = Math.max(0, Number(searchParams.get('offset')) || 0)
  const order = searchParams.get('order') === 'created' ? 'started_at' : 'last_active'
  const base = (await dependencies.requestPrimary(`/api/profiles/sessions?${searchParams}`)) as any

  const remoteParams = new URLSearchParams(searchParams)
  remoteParams.set('limit', String(limit + offset))
  remoteParams.set('offset', '0')

  const remoteSet = new Set(remoteProfiles)
  const merged = sessionRows(base).filter(session => !remoteSet.has(session?.profile))
  const profileTotals = { ...(base.profile_totals || {}) }

  let total =
    (Number(base.total) || 0) - remoteProfiles.reduce((sum, profile) => sum + (profileTotals[profile] || 0), 0)

  await Promise.all(
    remoteProfiles.map(async profile => {
      const list = (await dependencies.requestRemoteProfile(profile, remoteParams).catch(() => null)) as any

      if (!list) {
        delete profileTotals[profile]

        return
      }

      const rows = sessionRows(list)
      merged.push(...rows)
      profileTotals[profile] = Number(list.total) || rows.length
      total += profileTotals[profile]
    })
  )

  const recency = session => session?.[order] ?? session?.started_at ?? 0
  merged.sort((left, right) => recency(right) - recency(left))

  return { ...base, sessions: merged.slice(offset, offset + limit), total, profile_totals: profileTotals }
}
