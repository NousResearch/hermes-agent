/** Electron-free profile routing and lifecycle coordination. */

export type BackendRouteIdentity = 'local' | 'remote-global' | 'remote-profile'
export type BackendTarget = 'pool' | 'primary'

export interface BackendSelection {
  route: BackendRouteIdentity
  target: BackendTarget
}

export interface EnsureBackendOptions {
  selection?: BackendSelection
}

/** `local` is the only exact gateway target until Desktop has a registry. */
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

export function selectBackendSelection(options: {
  explicitLocal?: boolean
  explicitRemote?: boolean
  forceLocal?: boolean
  globalRemote?: boolean
  primaryProfile: string
  profile: string
}): BackendSelection {
  const route: BackendRouteIdentity =
    options.forceLocal || options.explicitLocal
      ? 'local'
      : options.explicitRemote
        ? 'remote-profile'
        : options.globalRemote
          ? 'remote-global'
          : 'local'

  const isPrimary = options.profile === options.primaryProfile
  const target: BackendTarget = isPrimary
    ? options.forceLocal
      ? 'pool'
      : 'primary'
    : options.explicitLocal || options.explicitRemote || !options.globalRemote
      ? 'pool'
      : 'primary'

  return { route, target }
}

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

    if (existing.routeIdentity === options.route) {
      options.touch?.(existing)

      return existing
    }

    await options.teardown(existing)
  }
}

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
