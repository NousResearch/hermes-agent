import crypto from 'node:crypto'

export interface ProfileRouteConfig {
  cloudOrg: string
  mode: 'cloud' | 'local' | 'remote' | 'ssh'
  remoteUrl: string
  sshHost: string
  sshPort: null | number
  sshRemoteHermesPath: string
  sshRemoteProfile: string
  sshUser: string
}

export interface EffectiveSshRoute {
  hostname: string
  port: null | number
  user: string
}

export interface OpaqueProfileRoute {
  connectionId: string
  mode: 'local' | 'remote'
  profile: string
  targetProfile: string
}

interface RegistryProfileRouteAgent {
  connectionId: string
  profile: string
}

interface RegistryProfileRouteSource {
  [field: string]: unknown
  id: string
  kind: 'cloud' | 'local' | 'remote' | 'ssh'
  remoteProfile?: string
}

interface BuildRegistryProfileRoutesOptions {
  agents: RegistryProfileRouteAgent[]
  legacyRoutes?: OpaqueProfileRoute[]
  sources: RegistryProfileRouteSource[]
}

interface BuildOpaqueProfileRoutesOptions {
  getProfileConfig: (profile: string) => ProfileRouteConfig | Promise<ProfileRouteConfig>
  globalConfig: ProfileRouteConfig
  installationId: string
  primaryProfile: string
  profileNames: string[]
  resolveSsh: (config: ProfileRouteConfig) => Promise<EffectiveSshRoute>
}

/** A 'connect-on-demand' local enumeration was intentionally deferred, not
 * failed — it must not be treated as a failure or Bot Mode will synthesize
 * cached local rows on remote-only workspaces where local was never dialed. */
export function isLocalEnumerationFailure(error?: string): boolean {
  return Boolean(error) && error !== 'connect-on-demand'
}

/** Return cached local profile names only when the local roster read failed. */
export function localRouteFallbackProfiles(
  agents: RegistryProfileRouteAgent[],
  localConnectionId: string,
  profileNames: string[],
  localEnumerationFailed: boolean
): string[] {
  if (!localEnumerationFailed) {
    return []
  }

  const existing = new Set(
    agents.filter(agent => agent.connectionId === localConnectionId).map(agent => normalizeProfile(agent.profile))
  )

  const fallback: string[] = []

  for (const raw of profileNames) {
    const profile = normalizeProfile(raw)

    if (!existing.has(profile)) {
      existing.add(profile)
      fallback.push(profile)
    }
  }

  return fallback
}

/** Credential-free seed routes let a plugin be the first caller to lazily dial
 * an SSH source. Once a source has roster agents, those live profiles are the
 * inventory and no speculative seed is needed. */
export function undialedSshRouteSeeds(
  agents: RegistryProfileRouteAgent[],
  sources: RegistryProfileRouteSource[]
): Array<{ connectionId: string; profile: string }> {
  const dialed = new Set(agents.map(agent => agent.connectionId))

  return sources
    .filter(source => source.kind === 'ssh' && !dialed.has(source.id))
    .map(source => ({
      connectionId: source.id,
      // remoteProfile is the backend target, not the source-local route. Keep
      // identity stable when the first lazy dial later inventories `default`.
      profile: 'default'
    }))
}

function normalizeProfile(name: null | string | undefined): string {
  return String(name ?? '').trim() || 'default'
}

function normalizeRemoteUrl(raw: string): string {
  const value = String(raw || '').trim()

  try {
    const url = new URL(value)
    url.username = ''
    url.password = ''
    url.search = ''
    url.hash = ''
    url.pathname = url.pathname.replace(/\/+$/, '') || '/'

    return url.toString().replace(/\/$/, '')
  } catch {
    return value.toLowerCase().replace(/\/+$/, '')
  }
}

async function connectionScope(
  config: ProfileRouteConfig,
  resolveSsh: BuildOpaqueProfileRoutesOptions['resolveSsh']
): Promise<{ key: string; mode: 'local' | 'remote' }> {
  if (config.mode === 'ssh') {
    let effective: EffectiveSshRoute

    try {
      effective = await resolveSsh(config)
    } catch {
      // `ssh -G` may time out or fail for one stale alias. Keep route inventory
      // available for every other profile and derive this opaque id from the
      // configured target; the actual lazy dial will still surface its error.
      effective = {
        hostname: config.sshHost,
        port: config.sshPort,
        user: config.sshUser
      }
    }

    // Remote profile is intentionally excluded: profiles mapped into the same
    // remote Hermes home form one interaction scope. Key/identity-file paths are
    // credentials and likewise stay out of the scope material.
    return {
      key: [
        'ssh',
        effective.user.trim(),
        effective.hostname.trim().toLowerCase(),
        effective.port ?? 22,
        config.sshRemoteHermesPath.trim()
      ].join('\0'),
      mode: 'remote'
    }
  }

  if (config.mode === 'cloud') {
    return {
      key: `cloud\0${normalizeRemoteUrl(config.remoteUrl)}\0${config.cloudOrg.trim()}`,
      mode: 'remote'
    }
  }

  if (config.mode === 'remote') {
    return { key: `remote\0${normalizeRemoteUrl(config.remoteUrl)}`, mode: 'remote' }
  }

  return { key: 'local', mode: 'local' }
}

function opaqueConnectionId(scope: string, installationId: string): string {
  const digest = crypto.createHmac('sha256', installationId).update(scope).digest('hex')

  return `connection-${digest.slice(0, 24)}`
}

function backendTargetProfile(scoped: ProfileRouteConfig, globalConfig: ProfileRouteConfig, profile: string): string {
  if (scoped.mode === 'ssh') {
    return normalizeProfile(scoped.sshRemoteProfile || profile)
  }

  // A per-profile URL/cloud override selects a standalone remote backend. It
  // does not forward the Desktop alias as a backend profile scope, so that
  // backend answers as its own root profile.
  if (scoped.mode === 'remote' || scoped.mode === 'cloud') {
    return 'default'
  }

  // An inherited global SSH route may explicitly pin the remote process to a
  // differently named profile. Without that pin, Desktop profile names remain
  // the backend profile scope, like inherited URL/cloud connections.
  if (globalConfig.mode === 'ssh' && globalConfig.sshRemoteProfile) {
    return normalizeProfile(globalConfig.sshRemoteProfile)
  }

  return profile
}

/**
 * Resolve Desktop routing profiles at the Electron boundary and return only
 * keyed, credential-free descriptors to the renderer/plugin runtime.
 */
export async function buildOpaqueProfileRoutes({
  getProfileConfig,
  globalConfig,
  installationId,
  primaryProfile,
  profileNames,
  resolveSsh
}: BuildOpaqueProfileRoutesOptions): Promise<OpaqueProfileRoute[]> {
  const primary = normalizeProfile(primaryProfile)
  const names: string[] = []
  const seen = new Set<string>()

  for (const raw of [primary, ...profileNames]) {
    const name = normalizeProfile(raw)

    if (!seen.has(name)) {
      seen.add(name)
      names.push(name)
    }
  }

  const globalScope = await connectionScope(globalConfig, resolveSsh)

  return Promise.all(
    names.map(async profile => {
      const scoped = await getProfileConfig(profile)
      const scope = scoped.mode === 'local' ? globalScope : await connectionScope(scoped, resolveSsh)

      return {
        connectionId: opaqueConnectionId(scope.key, installationId),
        mode: scope.mode,
        profile,
        targetProfile: backendTargetProfile(scoped, globalConfig, profile)
      }
    })
  )
}

/**
 * Project the union registry roster into the narrow plugin descriptor. Registry
 * ids and profile names are routing identities; endpoint/auth/source fields are
 * deliberately discarded here. A registry source of kind `local` always means
 * the actual local runtime, independently of legacy v1 global/profile routing.
 */
export function buildRegistryProfileRoutes({
  agents,
  sources
}: BuildRegistryProfileRoutesOptions): OpaqueProfileRoute[] {
  const sourceById = new Map(sources.map(source => [source.id, source]))
  const seen = new Set<string>()
  const routes: OpaqueProfileRoute[] = []

  for (const agent of agents) {
    const profile = normalizeProfile(agent.profile)
    const source = sourceById.get(agent.connectionId)
    const key = `${agent.connectionId}\0${profile}`

    if (!source || seen.has(key)) {
      continue
    }

    seen.add(key)

    if (source.kind === 'local') {
      routes.push({
        connectionId: source.id,
        mode: 'local',
        profile,
        targetProfile: profile
      })

      continue
    }

    routes.push({
      connectionId: source.id,
      mode: 'remote',
      profile,
      targetProfile: source.kind === 'ssh' && source.remoteProfile ? normalizeProfile(source.remoteProfile) : profile
    })
  }

  return routes
}

/** Add the backend profile scope only for registry remote/cloud descriptors. */
export function registryGatewayWsUrl(
  connection: { profile?: null | string; sharedRemote?: boolean },
  wsUrl: string
): string {
  if (!connection.sharedRemote) {
    return wsUrl
  }

  const url = new URL(wsUrl)
  url.searchParams.set('profile', normalizeProfile(connection.profile))

  return url.toString()
}
